# Databricks notebook source
# ==============================================================================
# FinSage | 07 — Agent Evaluation (production-grade)
#
# End-to-end MLflow GenAI evaluation of the deployed RAG agent against the
# 100-question ground-truth dataset. Designed as enterprise infrastructure:
#
#   • Modular scorers       — src/evaluation/scorers.py (6 custom + 2 LLM judges)
#   • Result persistence    — Delta tables under main.finsage_gold:
#                                 eval_run_summaries
#                                 eval_question_outcomes
#                             both idempotent on run_id (MERGE INTO).
#   • Failure analysis      — src/evaluation/analysis.py (per-scorer breakdown,
#                             per-category matrix, regression diff vs. previous
#                             run, question-level flips PASS↔FAIL).
#   • In-notebook failures  — even if Delta persistence fails, every failing
#                             question is printed with agent_response vs. expected,
#                             pulled directly from MLflow trace assessments.
#   • Pre-flight smoke      — 5-question fast sanity before the full 100 run, so
#                             a deploy-time regression fails in ~30 s, not 5 min.
#   • Provenance            — every run row links (mlflow run_id, agent_version,
#                             dataset hash, git commit, mlflow_url).
#
# Scorer suite
# -------------
#   Correctness                  — built-in LLM judge (databricks-meta-llama-3-3-70b)
#   cites_ticker_and_year        — Guidelines built-in, must mention TICKER + FY
#   numerical_tolerance          — ±1% on numerical_lookup, B/M/K-aware extraction
#   citation_format              — [VERBATIM]/[SUMMARY] + [Source:] line
#   refusal_correctness          — refusal-test: declined for the right reason
#   tool_routing_correctness     — agent picked the right tool
#   derived_metric_match         — numeric tolerance, decoupled from wording
#   retrieval_grounded_when_used — replaces MLflow's RetrievalGroundedness; SKIPS
#                                  on traces with no RETRIEVER span instead of
#                                  erroring. Most metrics-only questions are
#                                  answered through the structured-data tools
#                                  (get_company_metrics / get_quarterly_metrics)
#                                  without invoking search_filings, so a sizeable
#                                  share of traces have no RETRIEVER span and
#                                  the built-in scorer would mark them all ERROR.
#
# Why in-process model load (mlflow.pyfunc.load_model + unwrap_python_model):
#   Querying the remote serving endpoint cuts the trace at the process boundary
#   (num_spans=1 always); tool spans only propagate when the agent runs in the
#   eval notebook's process. unwrap_python_model bypasses pyfunc's signature-
#   driven input mangling that converts dict→DataFrame and JSON-stringifies
#   nested fields.
#
# Local pre-flight (run before redeploying notebook to workspace):
#   .venv/bin/python -m pytest tests/unit/test_eval_preflight.py -v
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
from mlflow.genai.scorers import Correctness, Guidelines
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
    retrieval_grounded_when_used,
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
    # MLflow's Expectation(value=...) constructor rejects None, so we only
    # populate keys whose values are non-None. fiscal_quarter is the common
    # offender — null for ~75 of the 100 rows (annual/refusal/multi_company).
    expectations = {
        "question_id":    q["question_id"],
        "category":       q["category"],
        "ticker":         q["ticker"],
        "difficulty":     q["difficulty"],
        "source_doc":     q["source_doc"],
        "source_section": q.get("source_section") or "n/a",
    }
    if q.get("fiscal_year") is not None:
        expectations["fiscal_year"] = q["fiscal_year"]
    if q.get("fiscal_quarter") is not None:
        expectations["fiscal_quarter"] = q["fiscal_quarter"]

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

# Fail-fast assertions — these are the "preflight before the cluster" checks
# that the unit tests in tests/unit/test_eval_preflight.py also enforce. If
# you see one of these tripped, run pytest locally before redeploying.
assert len(eval_dataset) == 100, f"expected 100 rows, got {len(eval_dataset)}"
for r in eval_dataset:
    for k, v in r["expectations"].items():
        assert v is not None, f"None value in expectations for {r['expectations'].get('question_id')!r} key={k!r}"
    has_resp  = "expected_response" in r["expectations"]
    has_facts = "expected_facts"    in r["expectations"]
    assert has_resp ^ has_facts, f"expected_response/expected_facts mutex broken: {r['expectations'].get('question_id')!r}"
print(f"[DATASET] {len(eval_dataset)} rows | fingerprint={DATASET_HASH} | preflight OK")

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

_LOADED_AGENT = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}/{LATEST_MODEL_VERSION}")

# Unwrap to the raw FinSageAgent class instance so we bypass MLflow's pyfunc
# signature-driven input transformation. The wrapper converts {"messages":[...]}
# into a single-row DataFrame and JSON-stringifies the messages list, which
# triggers AttributeError("'str' object has no attribute 'get'") inside the
# agent's predict when it iterates messages expecting dicts. Calling the
# unwrapped class instance directly preserves the dict shape end-to-end.
RAW_AGENT = _LOADED_AGENT.unwrap_python_model()

@mlflow.trace(span_type="AGENT")
def predict_fn(messages: list) -> dict:
    """Run the agent in-process so its @mlflow.trace tools (RETRIEVER, TOOL)
    emit child spans into the eval trace. Returns chat-completion-shaped dict."""
    try:
        out = RAW_AGENT.predict(None, {"messages": messages})
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

# In-process smoke test — must produce a real $ figure, NOT an "agent error".
test_out = predict_fn([{"role": "user", "content": "What was Apple's revenue in fiscal year 2024?"}])
preview = test_out["choices"][0]["message"]["content"]
print(f"[SMOKE in-process] {preview[:240]!r}")
assert "In-process agent error" not in preview, (
    f"Smoke test returned an error fallback — agent didn't actually run.\n"
    f"Preview: {preview[:400]}"
)
assert ("$" in preview or "B" in preview or "%" in preview), (
    f"Smoke test response contains no numerical signal — agent likely produced "
    f"a refusal or hallucination on a deterministic-answerable question.\n"
    f"Preview: {preview[:400]}"
)

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
            retrieval_grounded_when_used,  # SKIPs on traces w/o RETRIEVER span
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
            retrieval_grounded_when_used,  # SKIPs on traces w/o RETRIEVER span
        ],
    )
    print(f"[EVAL] run_id={RUN_ID}")
    print(f"[EVAL] metrics={getattr(results, 'metrics', None)}")

# COMMAND ----------

# ── 8a. Collect per-trace assessments from MLflow (no Spark; cannot fail) ────
# Pulls assessments via REST and assembles in-memory `outcome_rows`. This is
# the canonical eval result, available immediately even if Delta persistence
# (cell 8c) fails. The summary in cell 8b reads from this in-memory list.

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

print(f"[COLLECT] traces tied to run {RUN_ID[:10]}: {len(run_traces)}")
print(f"[COLLECT] outcome rows: {len(outcome_rows)}")
assert len(run_traces) > 0, "No traces found for this run — eval did not produce output. Investigate cell 7."
assert len(outcome_rows) > 0, "No scorer outcomes captured — every scorer either errored or skipped."

# COMMAND ----------

# ── 8b. In-memory summary + failure list (no Spark; always works) ────────────
# This is the cell that gives you the answer even if persistence fails.

from collections import defaultdict, Counter as _Counter  # noqa: E402

print("=" * 86)
print(f"EVAL SUMMARY  run_id={RUN_ID}  agent_version={AGENT_VERSION}  questions={len(eval_dataset)}")
print("=" * 86)
for sname in sorted(scorer_metrics):
    m = scorer_metrics[sname]
    answered = m.get("pass", 0) + m.get("fail", 0)
    rate = m.get("pass", 0) / answered if answered else None
    rate_s = f"{rate:.1%}" if rate is not None else "  n/a"
    print(f"  {sname:30s} pass={m.get('pass',0):3d}  fail={m.get('fail',0):3d}  "
          f"err={m.get('error',0):3d}  skip={m.get('skip',0):3d}  rate={rate_s}")

# Per-category × scorer matrix
print("\nPER-CATEGORY × SCORER:")
matrix: dict = defaultdict(lambda: defaultdict(_Counter))
for r in outcome_rows:
    matrix[r["category"]][r["scorer_name"]][r["outcome"]] += 1
for cat in sorted(matrix):
    print(f"  [{cat}]")
    for sname in sorted(matrix[cat]):
        c = matrix[cat][sname]
        ans = c.get("PASS",0) + c.get("FAIL",0)
        rate = c.get("PASS",0)/ans if ans else None
        rate_s = f"{rate:.1%}" if rate is not None else "  n/a"
        print(f"    {sname:30s} {c.get('PASS',0):3d}/{ans:<3d}  rate={rate_s}  err={c.get('ERROR',0)}  skip={c.get('SKIP',0)}")

# Failures drill-down — even one row per failure with full agent vs expected.
fails = [r for r in outcome_rows if r["outcome"] in {"FAIL", "ERROR"}]
print(f"\nFAILURES + ERRORS ({len(fails)} rows):")
print("=" * 86)
for r in sorted(fails, key=lambda x: (x["scorer_name"], x["question_id"])):
    print(f"\n{r['question_id']} | {r['scorer_name']} | {r['outcome']} | cat={r['category']} | ticker={r['ticker']}")
    if r.get("rationale"):
        print(f"  WHY: {(r['rationale'] or '')[:280]}")
    if r.get("error_message"):
        print(f"  ERR: {(r['error_message'] or '')[:280]}")
    if r.get("agent_response"):
        print(f"  AGENT:    {r['agent_response'][:280]}")
    if r.get("expected_response"):
        print(f"  EXPECTED: {r['expected_response'][:280]}")
print("=" * 86)

# COMMAND ----------

# ── 8c. Persist to Delta (eval_run_summaries + eval_question_outcomes) ───────
# Wrapped in try/except so a Delta failure doesn't suppress the in-memory
# results from cell 8b. Idempotent MERGE INTO on run_id, so safe to re-run.

run_started_ms  = run.info.start_time
run_finished_ms = run.info.end_time or int(time.time() * 1000)
mlflow_url = (
    f"{w.config.host.rstrip('/')}/ml/experiments/{EXPERIMENT_ID}/runs/{RUN_ID}"
    if w.config.host else f"{EXPERIMENT_ID}/{RUN_ID}"
)

try:
    eval_persistence.ensure_tables(spark)
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
except Exception as e:
    print(f"[PERSIST] FAILED: {type(e).__name__}: {str(e)[:300]}")
    print(f"[PERSIST] In-memory results in cell 8b are unaffected. "
          f"Re-run only this cell once persistence is fixed.")

# COMMAND ----------

# ── 9. Delta-backed analysis (only if persistence in 8c succeeded) ───────────
# These cells query main.finsage_gold.eval_* via Spark. If 8c failed they will
# fail too — but the in-memory results in cell 8b have already been printed.

try:
    eval_analysis.print_summary(spark, RUN_ID)
except Exception as e:
    print(f"[ANALYSIS] print_summary skipped: {type(e).__name__}: {str(e)[:200]}")

# COMMAND ----------

# ── 10. Failures (drill-down via Delta) ──────────────────────────────────────
try:
    failures = eval_analysis.failure_breakdown(spark, RUN_ID)
    print("=" * 86); print("FAILURES + ERRORS (this run)"); print("=" * 86)
    display(failures)  # noqa: F821 — Databricks display()
except Exception as e:
    print(f"[ANALYSIS] failure_breakdown skipped: {type(e).__name__}: {str(e)[:200]}")

# COMMAND ----------

# ── 11. Regression diff vs. previous run ─────────────────────────────────────
try:
    diff = eval_analysis.regression_diff(spark, RUN_ID)
    print("=" * 86); print("REGRESSION DIFF vs. previous run on same agent endpoint"); print("=" * 86)
    display(diff)  # noqa: F821
except Exception as e:
    print(f"[ANALYSIS] regression_diff skipped: {type(e).__name__}: {str(e)[:200]}")

# COMMAND ----------

# ── 12. Run-level trend (last 10 runs) ───────────────────────────────────────
try:
    recent = eval_persistence.fetch_recent_runs(spark, limit=10)
    print("=" * 86); print("RECENT EVAL RUNS"); print("=" * 86)
    display(recent)  # noqa: F821
except Exception as e:
    print(f"[ANALYSIS] fetch_recent_runs skipped: {type(e).__name__}: {str(e)[:200]}")

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
