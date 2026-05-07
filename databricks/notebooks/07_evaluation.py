# Databricks notebook source
# ==============================================================================
# FinSage | 07 — Agent Evaluation (production-grade)
#
# End-to-end MLflow GenAI evaluation of the deployed RAG agent against the
# 100-question ground-truth dataset. Designed as enterprise infrastructure:
#
#   • Modular scorers       — src/evaluation/scorers.py (5 custom + 1 LLM judge)
#   • Result persistence    — Delta tables under main.finsage_gold:
#                                 eval_run_summaries
#                                 eval_question_outcomes
#                             both idempotent on run_id (MERGE INTO).
#   • Failure analysis      — src/evaluation/analysis.py (per-scorer breakdown,
#                             per-category matrix, regression diff vs. previous
#                             run, question-level flips PASS↔FAIL).
#   • Pre-flight smoke      — 5-question fast sanity before the full 100 run, so
#                             a deploy-time regression fails in 30 s, not 5 min.
#   • Provenance & traceability — every Delta row is linked to (mlflow run_id,
#                             agent endpoint version, dataset hash, git commit).
#
# Scorer suite
# -------------
#   Correctness               — built-in LLM judge (databricks-meta-llama-3-3-70b)
#   cites_ticker_and_year     — Guidelines built-in, must mention TICKER + FY when
#                               discussing financial figures
#   numerical_tolerance       — ±1% on numerical_lookup, B/M/K-aware extraction
#   citation_format           — [VERBATIM]/[SUMMARY] + [Source:] line on retrieval-
#                               heavy questions
#   refusal_correctness       — refusal-test category: declined for the right reason
#   tool_routing_correctness  — agent picked the right tool (annual / quarterly /
#                               metadata / search) for the question
#   derived_metric_match      — numeric extraction + tolerance, decoupled from
#                               natural-language wording (catches LLM-judge over-
#                               strictness on derived ratios)
#
# RetrievalGroundedness IS enabled — we load the registered UC model in-process
# (mlflow.pyfunc.load_model) instead of querying the remote serving endpoint, so
# search_filings's RETRIEVER span and the metrics tools' TOOL spans propagate
# into the eval trace tree. The deployed endpoint is still pinged once as a
# canary so production regressions get noticed.
# ==============================================================================

# COMMAND ----------

# ── 1. Runtime parameters ────────────────────────────────────────────────────
dbutils.widgets.text("catalog",          "main",                                       "UC catalog")
dbutils.widgets.text("env",              "dev",                                        "Environment")
dbutils.widgets.text("agent_endpoint",   "finsage_agent_endpoint",                     "Target agent serving endpoint")
dbutils.widgets.text("judge_endpoint",   "databricks-meta-llama-3-3-70b-instruct",     "LLM-as-judge endpoint")
dbutils.widgets.text("ground_truth_path","../../src/evaluation/ground_truth_v2.json",  "Ground-truth JSON (notebook-relative)")
dbutils.widgets.text("eval_name",        "finsage_eval_v2",                            "MLflow eval run name")
dbutils.widgets.text("experiment_id",    "8c0b194f632349c6bc5ebe8c7a45480c",           "MLflow experiment id")
dbutils.widgets.dropdown("smoke_only",   "false", ["true", "false"],                   "Run the 5-question smoke only")

CATALOG          = dbutils.widgets.get("catalog")
ENV              = dbutils.widgets.get("env")
AGENT_ENDPOINT   = dbutils.widgets.get("agent_endpoint")
JUDGE_ENDPOINT   = dbutils.widgets.get("judge_endpoint")
GROUND_TRUTH     = dbutils.widgets.get("ground_truth_path")
EVAL_NAME        = dbutils.widgets.get("eval_name")
EXPERIMENT_ID    = dbutils.widgets.get("experiment_id")
SMOKE_ONLY       = dbutils.widgets.get("smoke_only").lower() == "true"

print(f"[CONFIG] agent={AGENT_ENDPOINT} | judge={JUDGE_ENDPOINT} | truth={GROUND_TRUTH} | smoke_only={SMOKE_ONLY}")

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow[databricks]>=3.0" databricks-sdk databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ── 2. Imports + put repo on sys.path so we can use src/evaluation/* ─────────
import json
import os
import re
import sys
import time
from pathlib import Path

import mlflow
from mlflow.genai.scorers import Correctness, Guidelines, RetrievalGroundedness
from databricks.sdk import WorkspaceClient

# Re-read widgets after restartPython() (they persist; Python state does not).
CATALOG         = dbutils.widgets.get("catalog")
ENV             = dbutils.widgets.get("env")
AGENT_ENDPOINT  = dbutils.widgets.get("agent_endpoint")
JUDGE_ENDPOINT  = dbutils.widgets.get("judge_endpoint")
GROUND_TRUTH    = dbutils.widgets.get("ground_truth_path")
EVAL_NAME       = dbutils.widgets.get("eval_name")
EXPERIMENT_ID   = dbutils.widgets.get("experiment_id")
SMOKE_ONLY      = dbutils.widgets.get("smoke_only").lower() == "true"

w = WorkspaceClient()

# Locate the repo root (containing /src) so `src.evaluation.*` is importable.
notebook_path = Path(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
notebook_dir = notebook_path.parent
# notebook lives at databricks/notebooks/, repo root is two levels up.
for repo_root_candidate in (notebook_dir.parent.parent, Path("/Workspace/Users/digvijay@arsaga.jp/FinSage")):
    if (Path("/Workspace") / str(repo_root_candidate).lstrip("/") / "src").exists():
        REPO_ROOT_WS = Path("/Workspace") / str(repo_root_candidate).lstrip("/")
        break
    if (repo_root_candidate / "src").exists():
        REPO_ROOT_WS = repo_root_candidate
        break
else:
    REPO_ROOT_WS = Path("/Workspace/Users/digvijay@arsaga.jp/FinSage")

sys.path.insert(0, str(REPO_ROOT_WS))
print(f"[IMPORT] repo_root_ws={REPO_ROOT_WS}")

from src.evaluation.scorers import (  # noqa: E402
    numerical_tolerance,
    citation_format,
    refusal_correctness,
    tool_routing_correctness,
    derived_metric_match,
)
from src.evaluation import persistence as eval_persistence  # noqa: E402
from src.evaluation import analysis as eval_analysis        # noqa: E402

print("[IMPORT] scorers + persistence + analysis loaded")

# COMMAND ----------

# ── 3. Load ground-truth dataset ─────────────────────────────────────────────
truth_path = (notebook_dir / GROUND_TRUTH).resolve() if not GROUND_TRUTH.startswith("/") else Path(GROUND_TRUTH)
ws_truth_path = Path("/Workspace") / str(truth_path).lstrip("/")

for candidate in (ws_truth_path, truth_path,
                  Path("/Workspace/Users/digvijay@arsaga.jp/FinSage/src/evaluation/ground_truth_v2.json")):
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
# and `expectations` (ground-truth fields consumed by scorers). Correctness
# requires exactly one of expected_response / expected_facts.

def to_eval_row(q: dict) -> dict:
    expectations = {
        "question_id":      q["question_id"],
        "category":         q["category"],
        "ticker":           q["ticker"],
        "fiscal_year":      q.get("fiscal_year"),
        "fiscal_quarter":   q.get("fiscal_quarter"),
        "difficulty":       q["difficulty"],
        "source_doc":       q["source_doc"],
        "source_section":   q.get("source_section", ""),
    }
    use_expected_response = (
        q.get("source_section") == "metrics"
        or q["category"] in {"numerical_lookup", "yoy_comparison", "multi_company", "refusal_test"}
    )
    if use_expected_response:
        expectations["expected_response"] = q["expected_answer"]
    else:
        expectations["expected_facts"] = [q["evidence_passage"]]

    return {
        "inputs": {"messages": [{"role": "user", "content": q["question"]}]},
        "expectations": expectations,
    }

eval_dataset = [to_eval_row(q) for q in ground]
DATASET_HASH = eval_persistence.dataset_fingerprint(eval_dataset)
print(f"[DATASET] {len(eval_dataset)} rows | fingerprint={DATASET_HASH}")

# COMMAND ----------

# ── 5. predict_fn — IN-PROCESS model load (RetrievalGroundedness requires this)
# We load the registered UC model directly into the eval notebook process so
# its internal RETRIEVER and TOOL spans propagate into the eval trace tree.
# Querying the remote serving endpoint cuts the spans off at the process
# boundary (num_spans=1) and breaks RetrievalGroundedness with
# "no RETRIEVER span found".
#
# The deployed endpoint is still probed once as a deploy-time canary so we
# notice if production drifts vs. the model registry.

UC_MODEL_NAME = f"{CATALOG}.finsage_gold.finsage_rag_agent"

# Resolve the latest UC model version (avoid hardcoding a number).
from mlflow.tracking import MlflowClient  # noqa: E402
mc = MlflowClient(registry_uri="databricks-uc")
versions = sorted(
    mc.search_model_versions(f"name='{UC_MODEL_NAME}'"),
    key=lambda v: int(v.version), reverse=True,
)
LATEST_MODEL_VERSION = versions[0].version if versions else None
if LATEST_MODEL_VERSION is None:
    raise RuntimeError(f"No registered versions found for {UC_MODEL_NAME}")
print(f"[MODEL] loading {UC_MODEL_NAME}/{LATEST_MODEL_VERSION} in-process")

LOADED_AGENT = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}/{LATEST_MODEL_VERSION}")

@mlflow.trace(span_type="AGENT")
def predict_fn(messages: list) -> dict:
    """Run the agent in-process so its @mlflow.trace tools (RETRIEVER, TOOL)
    emit child spans into the eval trace. Returns chat-completion-shaped dict."""
    try:
        out = LOADED_AGENT.predict({"messages": messages})
        if isinstance(out, list) and out:
            out = out[0]
        if isinstance(out, dict):
            content = out.get("content", "")
        elif isinstance(out, str):
            content = out
        else:
            content = str(out)
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}
    except Exception as e:
        return {"choices": [{"message": {"role": "assistant",
                "content": f"In-process agent error: {type(e).__name__}: {e}"}}]}

# In-process smoke test — also verify trace spans propagated.
try:
    test_out = predict_fn([{"role": "user", "content": "What was Apple's revenue in fiscal year 2024?"}])
    preview = test_out["choices"][0]["message"]["content"][:200]
    print(f"[SMOKE in-process] {preview!r}")
except Exception as e:
    print(f"[SMOKE FAIL] {type(e).__name__}: {e}")
    raise

# Deploy-time canary against the served endpoint (non-fatal — the eval uses
# the in-process model, but we want to know if production is broken).
try:
    canary = w.serving_endpoints.query(
        name=AGENT_ENDPOINT,
        dataframe_records=[{"messages": [{"role": "user", "content": "ping"}]}],
    )
    print(f"[ENDPOINT_CANARY] {AGENT_ENDPOINT} reachable ({type(canary.predictions).__name__})")
except Exception as e:
    print(f"[ENDPOINT_CANARY] non-fatal: {type(e).__name__}: {e}")

# COMMAND ----------

# ── 6. Pre-flight smoke (5 questions across categories) ──────────────────────
# Before paying for 100 LLM-judge calls, run 5 stratified questions to confirm
# every scorer is wiring correctly. Fails fast on regressions.

PREFLIGHT_IDS = ["A001", "B001", "C001", "E001", "F001"]
preflight_dataset = [r for r in eval_dataset if r["expectations"]["question_id"] in PREFLIGHT_IDS]

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_id=EXPERIMENT_ID)

with mlflow.start_run(run_name=f"{EVAL_NAME}_preflight") as preflight_run:
    preflight_results = mlflow.genai.evaluate(
        data=preflight_dataset,
        predict_fn=predict_fn,
        scorers=[
            Correctness(),
            RetrievalGroundedness(),  # works now: in-process load → RETRIEVER spans visible
            Guidelines(
                name="cites_ticker_and_year",
                guidelines=("The response must cite both a ticker symbol and a fiscal year "
                            "when discussing any financial figure or filing excerpt."),
            ),
            numerical_tolerance,
            citation_format,
            refusal_correctness,
            tool_routing_correctness,
            derived_metric_match,
        ],
    )
    print(f"[PREFLIGHT] run_id={preflight_run.info.run_id} metrics={getattr(preflight_results, 'metrics', None)}")

if SMOKE_ONLY:
    print("[SMOKE_ONLY=true] preflight complete; skipping the full 100-question run.")
    dbutils.notebook.exit("preflight_only")

# COMMAND ----------

# ── 7. Full evaluation ───────────────────────────────────────────────────────

# Resolve current agent serving version (logged with the run for traceability).
try:
    ep = w.serving_endpoints.get(AGENT_ENDPOINT)
    served = ep.config.served_entities[0] if ep.config and ep.config.served_entities else None
    AGENT_VERSION = str(served.entity_version) if served else None
except Exception as e:
    print(f"[VERSION] could not read endpoint version: {e}")
    AGENT_VERSION = None

# Try to capture git commit if running from a Databricks Repo.
try:
    from databricks.sdk.runtime import dbutils as _db  # type: ignore[import]
    GIT_COMMIT = (
        _db.notebook.entry_point.getDbutils().notebook().getContext()
          .tags().apply("mlflow.databricks.gitRepoCommit")
    )
except Exception:
    GIT_COMMIT = None

with mlflow.start_run(run_name=EVAL_NAME) as run:
    RUN_ID = run.info.run_id
    mlflow.log_params({
        "agent_endpoint":  AGENT_ENDPOINT,
        "agent_version":   AGENT_VERSION,
        "judge_endpoint":  JUDGE_ENDPOINT,
        "num_questions":   len(eval_dataset),
        "dataset_hash":    DATASET_HASH,
        "categories":      ",".join(sorted(set(q["category"] for q in ground))),
    })
    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_fn,
        scorers=[
            Correctness(),
            RetrievalGroundedness(),  # works now: in-process load → RETRIEVER spans visible
            Guidelines(
                name="cites_ticker_and_year",
                guidelines=("The response must cite both a ticker symbol and a fiscal year "
                            "when discussing any financial figure or filing excerpt."),
            ),
            numerical_tolerance,
            citation_format,
            refusal_correctness,
            tool_routing_correctness,
            derived_metric_match,
        ],
    )
    print(f"[EVAL] run_id={RUN_ID}")
    print(f"[EVAL] metrics={getattr(results, 'metrics', None)}")

# COMMAND ----------

# ── 8. Persist results to Delta (eval_run_summaries + eval_question_outcomes)
# Pulls per-trace assessments via MLflow REST and writes both a one-row run
# summary and one row per (question, scorer) outcome. Idempotent on run_id.

eval_persistence.ensure_tables(spark)

api = w.api_client
trace_resp = api.do(
    "GET",
    "/api/2.0/mlflow/traces",
    query={"experiment_ids": EXPERIMENT_ID, "max_results": 500},
)
all_traces = trace_resp.get("traces", [])

run_traces = []
for t in all_traces:
    for tag in t.get("tags", []):
        if tag.get("key", "").startswith("mlflow.assessment.") and RUN_ID in tag.get("value", ""):
            run_traces.append(t)
            break
print(f"[PERSIST] traces tied to run {RUN_ID[:10]}: {len(run_traces)}")

EXPECTATION_NAMES = {
    "category", "difficulty", "expected_facts", "expected_response",
    "fiscal_year", "fiscal_quarter", "question_id", "source_doc",
    "source_section", "ticker",
}

outcome_rows: list[dict] = []
scorer_metrics: dict[str, dict] = {}

for trace in run_traces:
    expectations: dict = {}
    feedbacks: dict[str, dict] = {}
    request_text  = next((m["value"] for m in trace.get("request_metadata", [])
                          if m["key"] == "mlflow.trace.request"),  "")
    response_text = next((m["value"] for m in trace.get("request_metadata", [])
                          if m["key"] == "mlflow.trace.response"), "")

    for tag in trace.get("tags", []):
        if not tag["key"].startswith("mlflow.assessment."):
            continue
        try:
            payload = json.loads(tag.get("value", "{}"))
        except json.JSONDecodeError:
            continue
        name = payload.get("assessment_name")
        if not name:
            continue
        if "expectation" in payload:
            expectations[name] = payload["expectation"].get("value")
            continue
        meta = payload.get("metadata") or {}
        if meta.get("mlflow.assessment.sourceRunId") != RUN_ID:
            continue
        feedbacks[name] = payload

    qid = expectations.get("question_id")
    if not qid:
        continue

    expected_response = expectations.get("expected_response") or ""
    if isinstance(expectations.get("expected_facts"), list) and not expected_response:
        expected_response = " ".join(expectations["expected_facts"])[:2000]

    for scorer_name, payload in feedbacks.items():
        if scorer_name in EXPECTATION_NAMES:
            continue
        fb = payload.get("feedback") or {}
        err = fb.get("error") or {}
        outcome, numeric = eval_persistence.normalize_outcome(
            fb.get("value"), bool(err.get("error_code"))
        )
        outcome_rows.append({
            "question_id":       qid,
            "scorer_name":       scorer_name,
            "category":          expectations.get("category"),
            "ticker":            expectations.get("ticker"),
            "fiscal_year":       expectations.get("fiscal_year"),
            "fiscal_quarter":    expectations.get("fiscal_quarter"),
            "difficulty":        expectations.get("difficulty"),
            "outcome":           outcome,
            "value_numeric":     numeric,
            "rationale":         payload.get("rationale") or fb.get("rationale"),
            "error_message":     err.get("error_message"),
            "agent_response":    response_text[:4000] if response_text else None,
            "expected_response": expected_response[:4000] if expected_response else None,
        })

        bucket = scorer_metrics.setdefault(scorer_name, {"pass": 0, "fail": 0, "error": 0, "skip": 0})
        bucket[outcome.lower()] = bucket.get(outcome.lower(), 0) + 1

print(f"[PERSIST] outcome rows: {len(outcome_rows)}")

# Run summary aggregates
run_started_ms  = run.info.start_time
run_finished_ms = run.info.end_time or int(time.time() * 1000)
mlflow_url = (
    f"{w.config.host.rstrip('/')}/ml/experiments/{EXPERIMENT_ID}/runs/{RUN_ID}"
    if w.config.host else f"{EXPERIMENT_ID}/{RUN_ID}"
)

eval_persistence.merge_run_summary(
    spark,
    run_id=RUN_ID,
    run_name=EVAL_NAME,
    experiment_id=EXPERIMENT_ID,
    run_started_at_ms=run_started_ms,
    run_finished_at_ms=run_finished_ms,
    agent_endpoint=AGENT_ENDPOINT,
    agent_version=AGENT_VERSION,
    judge_endpoint=JUDGE_ENDPOINT,
    num_questions=len(eval_dataset),
    dataset_path=str(truth_path),
    dataset_hash=DATASET_HASH,
    scorer_metrics=scorer_metrics,
    params={"smoke_only": SMOKE_ONLY},
    mlflow_url=mlflow_url,
    git_commit=GIT_COMMIT,
)
eval_persistence.merge_question_outcomes(spark, RUN_ID, outcome_rows)
print(f"[PERSIST] wrote summary + outcomes for run {RUN_ID[:10]}")

# COMMAND ----------

# ── 9. Pretty summary + per-category breakdown ───────────────────────────────
eval_analysis.print_summary(spark, RUN_ID)

# COMMAND ----------

# ── 10. Failures (drill-down) ────────────────────────────────────────────────
failures = eval_analysis.failure_breakdown(spark, RUN_ID)
print("=" * 86)
print("FAILURES + ERRORS (this run)")
print("=" * 86)
display(failures)  # noqa: F821 — Databricks display()

# COMMAND ----------

# ── 11. Regression diff vs. previous run ─────────────────────────────────────
diff = eval_analysis.regression_diff(spark, RUN_ID)
print("=" * 86)
print("REGRESSION DIFF vs. previous run on same agent endpoint")
print("=" * 86)
display(diff)  # noqa: F821

# COMMAND ----------

# ── 12. Run-level trend (last 10 runs) ───────────────────────────────────────
recent = eval_persistence.fetch_recent_runs(spark, limit=10)
print("=" * 86)
print("RECENT EVAL RUNS")
print("=" * 86)
display(recent)  # noqa: F821

# COMMAND ----------

# ── 13. Iteration playbook ───────────────────────────────────────────────────
# 1. Open the MLflow run linked above; per-row traces include scorer rationale.
# 2. Common failure modes:
#    - Wrong fiscal year retrieved   → tighten SYSTEM_PROMPT year-filter directive
#    - Missing citations             → strengthen [VERBATIM]/[SUMMARY] instruction
#    - Numeric drift > 1%            → audit gold metrics vs. SEC EDGAR
#    - Tool-routing miss             → add explicit example to SYSTEM_PROMPT
#    - Refusal too brief             → expand expected refusal context tokens in
#                                      src/evaluation/scorers.py:_REFUSAL_CONTEXT_TOKENS
# 3. Re-run notebook 06 to deploy a new agent version, then re-run this notebook.
#    The Delta tables auto-track regression deltas via run_id and agent_version.
