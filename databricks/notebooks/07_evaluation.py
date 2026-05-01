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
dbutils.widgets.text("ground_truth_path", "../../src/evaluation/ground_test.json", "Ground-truth JSON (workspace-relative)")
dbutils.widgets.text("eval_name",       "finsage_eval_v1",          "MLflow eval run name")

CATALOG          = dbutils.widgets.get("catalog")
ENV              = dbutils.widgets.get("env")
AGENT_ENDPOINT   = dbutils.widgets.get("agent_endpoint")
JUDGE_ENDPOINT   = dbutils.widgets.get("judge_endpoint")
GROUND_TRUTH     = dbutils.widgets.get("ground_truth_path")
EVAL_NAME        = dbutils.widgets.get("eval_name")

print(f"[CONFIG] agent={AGENT_ENDPOINT} | judge={JUDGE_ENDPOINT} | truth={GROUND_TRUTH}")

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.0" databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ── 2. Imports + re-declare constants (required after restartPython) ─────────
import json
import os
import re
from pathlib import Path

import mlflow
from mlflow.genai.scorers import scorer, Correctness, RetrievalGroundedness, Guidelines
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

CATALOG         = dbutils.widgets.get("catalog")
ENV             = dbutils.widgets.get("env")
AGENT_ENDPOINT  = dbutils.widgets.get("agent_endpoint")
JUDGE_ENDPOINT  = dbutils.widgets.get("judge_endpoint")
GROUND_TRUTH    = dbutils.widgets.get("ground_truth_path")
EVAL_NAME       = dbutils.widgets.get("eval_name")

w = WorkspaceClient()

# COMMAND ----------

# ── 3. Load ground-truth dataset ─────────────────────────────────────────────
notebook_dir = Path(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
).parent

truth_path = (notebook_dir / GROUND_TRUTH).resolve() if not GROUND_TRUTH.startswith("/") else Path(GROUND_TRUTH)
# /Workspace prefix for workspace files
ws_truth_path = Path("/Workspace") / str(truth_path).lstrip("/")

for candidate in (ws_truth_path, truth_path, Path(f"/Workspace/Users/digvijay@arsaga.jp/FinSage/src/evaluation/ground_test.json")):
    if candidate.exists():
        truth_path = candidate
        break
else:
    raise FileNotFoundError(f"ground_test.json not found. Tried: {ws_truth_path}, {truth_path}")

with open(truth_path) as f:
    ground = json.load(f)

print(f"[LOAD] {len(ground)} eval questions from {truth_path}")
print(f"[LOAD] categories: {sorted(set(q['category'] for q in ground))}")

# COMMAND ----------

# ── 4. Build MLflow eval dataset ─────────────────────────────────────────────
# MLflow GenAI eval expects rows with `inputs` (payload passed to predict_fn)
# and `expectations` (ground-truth fields consumed by scorers).

def to_eval_row(q: dict) -> dict:
    return {
        "inputs": {
            "messages": [{"role": "user", "content": q["question"]}]
        },
        "expectations": {
            "expected_response":     q["expected_answer"],
            "expected_facts":        [q["evidence_passage"]],
            "question_id":           q["question_id"],
            "category":              q["category"],
            "ticker":                q["ticker"],
            "fiscal_year":           q.get("fiscal_year"),
            "difficulty":            q["difficulty"],
            "source_doc":            q["source_doc"],
        }
    }

eval_dataset = [to_eval_row(q) for q in ground]
print(f"[DATASET] {len(eval_dataset)} rows prepared")

# COMMAND ----------

# ── 5. predict_fn — queries the deployed agent ───────────────────────────────
# Uses w.serving_endpoints.query() (SDK) to avoid the deploy-client URL bug
# documented in CLAUDE.md (cell 12 live test learning).

@mlflow.trace(span_type="AGENT")
def predict_fn(messages: list) -> dict:
    chat_messages = [
        ChatMessage(
            role=ChatMessageRole(m["role"].lower() if m["role"].lower() in ("user","system","assistant") else "user"),
            content=m["content"],
        )
        for m in messages
    ]
    resp = w.serving_endpoints.query(
        name=AGENT_ENDPOINT,
        messages=chat_messages,
    )
    # resp.choices is a list[Choice]; Choice.message has role+content
    content = resp.choices[0].message.content if resp.choices else ""
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}

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

_NUM_RE = re.compile(r"\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?", re.IGNORECASE)

def _first_number(text: str) -> float | None:
    for m in _NUM_RE.finditer(text or ""):
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        tail = (text[m.end():m.end()+10] or "").lower()
        if "billion" in tail or tail.strip().startswith("b"):
            val *= 1_000  # normalize to millions
        return val
    return None

@scorer
def numerical_tolerance(outputs, expectations):
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
    pred_num = _first_number(response)
    true_num = _first_number(expected_text)

    if pred_num is None or true_num is None or true_num == 0:
        return {"value": False, "rationale": f"Could not parse numbers (pred={pred_num}, true={true_num})"}

    rel_err = abs(pred_num - true_num) / abs(true_num)
    passed = rel_err <= 0.01
    return {
        "value": passed,
        "rationale": f"pred={pred_num:.2f}, true={true_num:.2f}, rel_err={rel_err:.4f} ({'PASS' if passed else 'FAIL'} @ 1% tol)"
    }

# COMMAND ----------

# ── 7. Custom scorer: citation format compliance ─────────────────────────────
# SYSTEM_PROMPT mandates [VERBATIM]/[SUMMARY] labels + [Source: TICKER | FY... | Section]

_CITATION_TAG = re.compile(r"\[(VERBATIM|SUMMARY)\]", re.IGNORECASE)
_SOURCE_LINE  = re.compile(r"\[Source:\s*[A-Z]+\s*\|\s*FY\d{4}", re.IGNORECASE)

@scorer
def citation_format(outputs, expectations):
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
    passed = has_tag and has_source
    return {
        "value": passed,
        "rationale": f"[VERBATIM]/[SUMMARY] tag: {has_tag} | [Source: ...] line: {has_source}"
    }

# COMMAND ----------

# ── 8. Run evaluation ────────────────────────────────────────────────────────
mlflow.set_registry_uri("databricks-uc")

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
            RetrievalGroundedness(),
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

# ── 9. Per-category breakdown ────────────────────────────────────────────────
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

# ── 10. How to iterate ───────────────────────────────────────────────────────
# 1. Open the MLflow run linked above, inspect per-row traces for failures.
# 2. Common failure modes:
#    - Wrong fiscal year retrieved  → tighten SYSTEM_PROMPT year-filter directive
#    - Missing citations            → strengthen [VERBATIM]/[SUMMARY] instruction
#    - Numeric drift > 1%           → verify get_company_metrics cache freshness
# 3. Re-run notebook 06 to deploy a new agent version, then re-run this notebook
#    to confirm score regressions/improvements.
