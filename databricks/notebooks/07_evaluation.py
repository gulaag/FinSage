# Databricks notebook source
# ==============================================================================
# FinSage | 07 — Agent Evaluation
#
# Evaluates the deployed RAG agent (finsage_agent_endpoint) against the curated
# ground-truth set in src/evaluation/ground_test.json using MLflow GenAI eval.
#
# Scorers:
#   - Correctness            : LLM judge, answer vs. expected_answer
#   - RetrievalGroundedness  : LLM judge, response grounded in retrieved context
#   - Guidelines (citations) : custom rule — agent must emit [VERBATIM] / [SUMMARY]
#                              tags and a [Source: ...] line per SYSTEM_PROMPT
#   - numerical_tolerance    : custom @scorer, ±1% tolerance on numerical_lookup
#                              category questions
#
# Output: MLflow evaluation run with per-question traces + aggregate scores.
# ==============================================================================

# COMMAND ----------

# ── 1. Runtime Parameters ─────────────────────────────────────────────────────
dbutils.widgets.text("catalog",         "main",                     "UC catalog")
dbutils.widgets.text("env",             "dev",                      "Environment")
dbutils.widgets.text("agent_endpoint",  "finsage_agent_endpoint",   "Target agent serving endpoint")
dbutils.widgets.text("judge_endpoint",  "databricks-meta-llama-3-3-70b-instruct", "LLM-as-judge endpoint")
dbutils.widgets.text("ground_truth_path", "../../src/evaluation/ground_truth_v2.json", "Ground-truth JSON (workspace-relative)")
dbutils.widgets.text("eval_name",       "finsage_eval_v2",          "MLflow eval run name")
dbutils.widgets.text("experiment_id",   "8c0b194f632349c6bc5ebe8c7a45480c", "MLflow experiment id")

CATALOG          = dbutils.widgets.get("catalog")
ENV              = dbutils.widgets.get("env")
AGENT_ENDPOINT   = dbutils.widgets.get("agent_endpoint")
JUDGE_ENDPOINT   = dbutils.widgets.get("judge_endpoint")
GROUND_TRUTH     = dbutils.widgets.get("ground_truth_path")
EVAL_NAME        = dbutils.widgets.get("eval_name")
EXPERIMENT_ID    = dbutils.widgets.get("experiment_id")

print(f"[CONFIG] agent={AGENT_ENDPOINT} | judge={JUDGE_ENDPOINT} | truth={GROUND_TRUTH}")

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.0" databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ── 2. Imports + re-declare constants (required after restartPython) ─────────
import json
import os
import re
import time
from pathlib import Path

import mlflow
from mlflow.genai.scorers import scorer, Correctness, Guidelines
from databricks.sdk import WorkspaceClient

CATALOG         = dbutils.widgets.get("catalog")
ENV             = dbutils.widgets.get("env")
AGENT_ENDPOINT  = dbutils.widgets.get("agent_endpoint")
JUDGE_ENDPOINT  = dbutils.widgets.get("judge_endpoint")
GROUND_TRUTH    = dbutils.widgets.get("ground_truth_path")
EVAL_NAME       = dbutils.widgets.get("eval_name")
EXPERIMENT_ID   = dbutils.widgets.get("experiment_id")

w = WorkspaceClient()

# COMMAND ----------

# ── 3. Load ground-truth dataset ─────────────────────────────────────────────
notebook_dir = Path(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
).parent

truth_path = (notebook_dir / GROUND_TRUTH).resolve() if not GROUND_TRUTH.startswith("/") else Path(GROUND_TRUTH)
# /Workspace prefix for workspace files
ws_truth_path = Path("/Workspace") / str(truth_path).lstrip("/")

for candidate in (ws_truth_path, truth_path, Path(f"/Workspace/Users/digvijay@arsaga.jp/FinSage/src/evaluation/ground_truth_v2.json")):
    if candidate.exists():
        truth_path = candidate
        break
else:
    raise FileNotFoundError(f"ground-truth JSON not found. Tried: {ws_truth_path}, {truth_path}")

with open(truth_path) as f:
    ground = json.load(f)

print(f"[LOAD] {len(ground)} eval questions from {truth_path}")
print(f"[LOAD] categories: {sorted(set(q['category'] for q in ground))}")

# COMMAND ----------

# ── 4. Build MLflow eval dataset ─────────────────────────────────────────────
# MLflow GenAI eval expects rows with `inputs` (payload passed to predict_fn)
# and `expectations` (ground-truth fields consumed by scorers).

def to_eval_row(q: dict) -> dict:
    expectations = {
        "question_id": q["question_id"],
        "category": q["category"],
        "ticker": q["ticker"],
        "fiscal_year": q.get("fiscal_year"),
        "difficulty": q["difficulty"],
        "source_doc": q["source_doc"],
        "source_section": q.get("source_section", ""),
    }

    # MLflow Correctness scorer requires exactly one of expected_response/expected_facts.
    category = q["category"]
    source_section = q.get("source_section", "")
    use_expected_response = source_section == "metrics" or category in {
        "numerical_lookup",
        "yoy_comparison",
        "multi_company",
        "refusal_test",
    }
    if use_expected_response:
        expectations["expected_response"] = q["expected_answer"]
    else:
        expectations["expected_facts"] = [q["evidence_passage"]]

    return {
        "inputs": {
            "messages": [{"role": "user", "content": q["question"]}]
        },
        "expectations": expectations,
    }

eval_dataset = [to_eval_row(q) for q in ground]
print(f"[DATASET] {len(eval_dataset)} rows prepared")

# COMMAND ----------

# ── 5. predict_fn — queries the deployed agent ───────────────────────────────
# The deployed agent is an mlflow.pyfunc.PythonModel registered with a
# {"messages": [...]} input signature, so it must be invoked via
# `dataframe_records` (pyfunc shape), NOT the chat-protocol `messages=` param.
# Empirically verified against finsage_agent_endpoint v16:
#   dataframe_records=[{"messages":[{"role":"user","content":"ping"}]}]
#   → predictions = {"content": "...", "messages": [...]}
#
# Using w.serving_endpoints.query() (SDK) to avoid the deploy-client URL bug
# documented in CLAUDE.md (cell 12 live test learning).

@mlflow.trace(span_type="AGENT")
def predict_fn(messages: list) -> dict:
    # Prevent hard trace failures: if the endpoint call errors, return an explicit
    # fallback response instead of raising.
    last_error = None
    for attempt in range(2):
        try:
            resp = w.serving_endpoints.query(
                name=AGENT_ENDPOINT,
                dataframe_records=[{"messages": messages}],
            )
            pred = resp.predictions
            if isinstance(pred, list) and pred:
                pred = pred[0]
            content = pred.get("content", "") if isinstance(pred, dict) else str(pred)
            return {"choices": [{"message": {"role": "assistant", "content": content}}]}
        except Exception as e:
            last_error = e
            if attempt == 0:
                time.sleep(0.5)
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": f"Unable to answer due to endpoint error: {type(last_error).__name__}: {last_error}",
                }
            }
        ]
    }

# smoke-test predict_fn before full eval
try:
    test_out = predict_fn([{"role": "user", "content": "ping"}])
    print(f"[SMOKE] predict_fn OK — preview: {test_out['choices'][0]['message']['content'][:200]!r}")
except Exception as e:
    print(f"[SMOKE FAIL] {type(e).__name__}: {e}")
    raise

# COMMAND ----------

# ── 6. Custom scorer: numerical tolerance (±1%) ──────────────────────────────
# For numerical_lookup questions, extract the first large dollar figure from
# both response and expected_answer; pass if within 1%.

_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9])([-+]?\$?\s*\d[\d,]*(?:\.\d+)?)(?:\s*(billion|million|thousand|bn|mn|k|b|m|%|x))?",
    re.IGNORECASE,
)


def _extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for m in _NUM_RE.finditer(text or ""):
        raw_token = m.group(1).replace("$", "").replace(",", "").strip()
        try:
            val = float(raw_token)
        except ValueError:
            continue
        unit = (m.group(2) or "").lower()
        if unit in {"billion", "bn", "b"}:
            val *= 1_000_000_000
        elif unit in {"million", "mn", "m"}:
            val *= 1_000_000
        elif unit in {"thousand", "k"}:
            val *= 1_000
        # "%"/"x"/no unit remain as-is
        values.append(val)
    return values


def _pick_best_predicted_number(pred_candidates: list[float], true_num: float) -> float | None:
    if not pred_candidates:
        return None
    # Ignore obvious fiscal-year tokens when we have alternatives.
    filtered = [v for v in pred_candidates if not (1900 <= abs(v) <= 2100)]
    candidates = filtered if filtered else pred_candidates
    if true_num == 0:
        return candidates[0]
    return min(candidates, key=lambda x: abs(x - true_num) / abs(true_num))

@scorer
def numerical_tolerance(*, outputs=None, expectations=None, **kwargs):
    category = (expectations or {}).get("category")
    if category != "numerical_lookup":
        return None  # skip — not applicable

    response = ""
    if isinstance(outputs, dict):
        try:
            response = outputs["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            response = str(outputs)

    expected_text = (expectations or {}).get("expected_response", "")
    pred_candidates = _extract_numbers(response)
    true_candidates = _extract_numbers(expected_text)
    if not true_candidates:
        return False
    true_num = true_candidates[0]  # dataset contract: canonical number appears first
    pred_num = _pick_best_predicted_number(pred_candidates, true_num)

    if pred_num is None or true_num is None or true_num == 0:
        return False

    rel_err = abs(pred_num - true_num) / abs(true_num)
    return rel_err <= 0.01

# COMMAND ----------

# ── 7. Custom scorer: citation format compliance ─────────────────────────────
# SYSTEM_PROMPT mandates [VERBATIM]/[SUMMARY] labels + [Source: TICKER | FY... | Section]

_CITATION_TAG = re.compile(r"\[(VERBATIM|SUMMARY)\]", re.IGNORECASE)
_SOURCE_LINE  = re.compile(r"\[Source:\s*[A-Z]+\s*\|\s*FY\d{4}", re.IGNORECASE)

@scorer
def citation_format(*, outputs=None, expectations=None, **kwargs):
    category = (expectations or {}).get("category")
    # Only applies where retrieval happens (not pure metrics questions)
    if category not in ("risk_summary", "citation_validation", "yoy_comparison", "numerical_lookup"):
        return None
    if (expectations or {}).get("source_section") == "metrics":
        return None

    response = ""
    if isinstance(outputs, dict):
        try:
            response = outputs["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            response = str(outputs)

    has_tag    = bool(_CITATION_TAG.search(response))
    has_source = bool(_SOURCE_LINE.search(response))
    return has_tag and has_source

# COMMAND ----------

# ── 8. Run evaluation ────────────────────────────────────────────────────────
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_id=EXPERIMENT_ID)

with mlflow.start_run(run_name=EVAL_NAME) as run:
    mlflow.log_params({
        "agent_endpoint": AGENT_ENDPOINT,
        "judge_endpoint": JUDGE_ENDPOINT,
        "num_questions":  len(eval_dataset),
        "categories":     ",".join(sorted(set(q["category"] for q in ground))),
    })

    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_fn,
        scorers=[
            Correctness(),
            # v2: RetrievalGroundedness disabled because current agent traces do not
            # emit RETRIEVER-typed spans; enable once tool spans are instrumented.
            Guidelines(
                name="cites_ticker_and_year",
                guidelines=(
                    "The response must cite both a ticker symbol and a fiscal year "
                    "when discussing any financial figure or filing excerpt."
                ),
            ),
            numerical_tolerance,
            citation_format,
        ],
    )

    print(f"[EVAL] run_id={run.info.run_id}")
    print(f"[EVAL] results.metrics={getattr(results, 'metrics', None)}")

# COMMAND ----------

# ── 9. Simple evaluation summary (human-readable) ────────────────────────────
def _normalize_feedback_value(raw):
    if raw is None:
        return None
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in {"yes", "pass", "true"}:
            return 1.0
        if token in {"no", "fail", "false"}:
            return 0.0
        try:
            return float(token)
        except ValueError:
            return None
    return None


def _print_simple_summary(eval_run_id: str) -> None:
    """Print a concise eval-health summary from MLflow traces."""
    api = w.api_client
    traces_resp = api.do(
        "GET",
        "/api/2.0/mlflow/traces",
        query={"experiment_ids": EXPERIMENT_ID, "max_results": 500},
    )
    traces = traces_resp.get("traces", [])
    run_traces = []
    for trace in traces:
        tags = trace.get("tags", [])
        if any(
            tag.get("key", "").startswith("mlflow.assessment.")
            and eval_run_id in tag.get("value", "")
            for tag in tags
        ):
            run_traces.append(trace)

    scorer_stats = {}
    for trace in run_traces:
        for tag in trace.get("tags", []):
            key = tag.get("key", "")
            if not key.startswith("mlflow.assessment."):
                continue
            payload = json.loads(tag.get("value", "{}"))
            name = payload.get("assessment_name")
            if not name:
                continue
            # ignore expectation echo rows, keep real scorer rows only
            if name in {
                "category",
                "difficulty",
                "expected_facts",
                "expected_response",
                "fiscal_year",
                "question_id",
                "source_doc",
                "source_section",
                "ticker",
            }:
                continue

            stats = scorer_stats.setdefault(
                name,
                {"total": 0, "answered": 0, "passed": 0, "failed": 0, "errors": 0},
            )
            stats["total"] += 1
            feedback = payload.get("feedback") or {}
            if feedback.get("error"):
                stats["errors"] += 1
                continue
            numeric = _normalize_feedback_value(feedback.get("value"))
            if numeric is None:
                continue
            stats["answered"] += 1
            if numeric >= 0.5:
                stats["passed"] += 1
            else:
                stats["failed"] += 1

    print("=" * 86)
    print("EVALUATION SUMMARY")
    print("=" * 86)
    print(f"Run ID: {eval_run_id}")
    print(f"Questions in dataset: {len(eval_dataset)}")
    print(f"Traces found for this run: {len(run_traces)}")
    print("")
    print("Per-scorer outcome:")
    for scorer_name in sorted(scorer_stats):
        s = scorer_stats[scorer_name]
        print(
            f"- {scorer_name}: total={s['total']}, answered={s['answered']}, "
            f"passed={s['passed']}, failed={s['failed']}, errors={s['errors']}"
        )
    print("=" * 86)


try:
    _print_simple_summary(run.info.run_id)
except Exception as e:
    print(f"[SUMMARY] skipped: {type(e).__name__}: {e}")

# COMMAND ----------

# ── 10. Per-category breakdown ───────────────────────────────────────────────
try:
    import pandas as pd
    df = results.tables["eval_results"] if hasattr(results, "tables") else None
    if df is not None:
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        cat_col = None
        for c in df.columns:
            if "category" in c.lower():
                cat_col = c; break
        if cat_col:
            print(df.groupby(cat_col).mean(numeric_only=True).round(3))
        else:
            print(df.head())
    else:
        print("[BREAKDOWN] no results.tables — check MLflow UI for the run")
except Exception as e:
    print(f"[BREAKDOWN] skipped: {type(e).__name__}: {e}")

# COMMAND ----------

# ── 11. How to iterate ───────────────────────────────────────────────────────
# 1. Open the MLflow run linked above, inspect per-row traces for failures.
# 2. Common failure modes:
#    - Wrong fiscal year retrieved  → tighten SYSTEM_PROMPT year-filter directive
#    - Missing citations            → strengthen [VERBATIM]/[SUMMARY] instruction
#    - Numeric drift > 1%           → verify get_company_metrics cache freshness
# 3. Re-run notebook 06 to deploy a new agent version, then re-run this notebook
#    to confirm score regressions/improvements.
