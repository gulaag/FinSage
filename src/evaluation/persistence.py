"""Eval result persistence to Unity Catalog Delta tables.

Two tables, both in `main.finsage_gold`:

    eval_run_summaries
        Run-level snapshot. One row per `mlflow.start_run` invocation. Includes
        aggregate scorer metrics, params, agent endpoint version, dataset
        hash, and a workspace-relative MLflow URL. Idempotent on `run_id`.

    eval_question_outcomes
        Per-question, per-scorer outcomes. One row per
        (run_id, question_id, scorer_name). Drives failure-analysis dashboards
        and regression diffs. Idempotent on the composite key.

Design choices:
    - MERGE INTO is used everywhere so re-running a notebook cell against the
      same MLflow run is safe (no duplicate rows).
    - Schema evolution is enabled via `delta.autoMerge` — adding a new scorer
      to scorers.py picks up automatically without manual ALTER.
    - Tables are created on first use (idempotent CREATE IF NOT EXISTS).
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

EVAL_SUMMARY_TABLE   = "main.finsage_gold.eval_run_summaries"
EVAL_OUTCOMES_TABLE  = "main.finsage_gold.eval_question_outcomes"


# ─────────────────────────────────────────────────────────────────────────────
# Table DDL
# ─────────────────────────────────────────────────────────────────────────────

def ensure_tables(spark) -> None:
    """Create both eval tables if they don't exist. Idempotent."""
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {EVAL_SUMMARY_TABLE} (
            run_id              STRING NOT NULL,
            run_name            STRING,
            experiment_id       STRING,
            run_started_at      TIMESTAMP,
            run_finished_at     TIMESTAMP,
            duration_seconds    DOUBLE,
            agent_endpoint      STRING,
            agent_version       STRING,
            judge_endpoint      STRING,
            num_questions       INT,
            dataset_path        STRING,
            dataset_hash        STRING,
            scorer_metrics_json STRING,
            param_json          STRING,
            mlflow_url          STRING,
            git_commit          STRING,
            recorded_at         TIMESTAMP
        )
        USING DELTA
        TBLPROPERTIES (delta.autoOptimize.optimizeWrite = true,
                       delta.enableChangeDataFeed       = true)
    """)
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {EVAL_OUTCOMES_TABLE} (
            run_id              STRING NOT NULL,
            question_id         STRING NOT NULL,
            scorer_name         STRING NOT NULL,
            category            STRING,
            ticker              STRING,
            fiscal_year         INT,
            fiscal_quarter      INT,
            difficulty          STRING,
            outcome             STRING,    -- PASS | FAIL | ERROR | SKIP
            value_numeric       DOUBLE,    -- 0/1 or numeric value
            rationale           STRING,
            error_message       STRING,
            agent_response      STRING,
            expected_response   STRING,
            recorded_at         TIMESTAMP
        )
        USING DELTA
        PARTITIONED BY (run_id)
        TBLPROPERTIES (delta.autoOptimize.optimizeWrite = true,
                       delta.enableChangeDataFeed       = true)
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def dataset_fingerprint(dataset: list[dict]) -> str:
    """Stable hash of the eval dataset — flags accidental dataset drift."""
    blob = json.dumps(
        [(r.get("expectations", {}).get("question_id"),
          r.get("inputs", {}).get("messages", [{}])[-1].get("content", ""))
         for r in dataset],
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def normalize_outcome(raw_value, has_error: bool) -> tuple[str, float | None]:
    """Map a scorer's raw feedback value to (outcome_label, numeric).

    outcome_label ∈ {PASS, FAIL, ERROR, SKIP}.
    numeric is 1.0 / 0.0 for boolean scorers, the raw float for numeric
    scorers, or NULL for SKIP/ERROR.
    """
    if has_error:
        return "ERROR", None
    if raw_value is None:
        return "SKIP", None
    if isinstance(raw_value, bool):
        return ("PASS" if raw_value else "FAIL"), (1.0 if raw_value else 0.0)
    if isinstance(raw_value, (int, float)):
        v = float(raw_value)
        return ("PASS" if v >= 0.5 else "FAIL"), v
    if isinstance(raw_value, str):
        token = raw_value.strip().lower()
        if token in {"yes", "pass", "true"}:
            return "PASS", 1.0
        if token in {"no", "fail", "false"}:
            return "FAIL", 0.0
        try:
            v = float(token)
            return ("PASS" if v >= 0.5 else "FAIL"), v
        except ValueError:
            return "FAIL", 0.0
    return "FAIL", 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────────────────────────────────────

_RUN_SUMMARY_SCHEMA = """
    run_id              STRING,
    run_name            STRING,
    experiment_id       STRING,
    run_started_at_ms   BIGINT,
    run_finished_at_ms  BIGINT,
    duration_seconds    DOUBLE,
    agent_endpoint      STRING,
    agent_version       STRING,
    judge_endpoint      STRING,
    num_questions       INT,
    dataset_path        STRING,
    dataset_hash        STRING,
    scorer_metrics_json STRING,
    param_json          STRING,
    mlflow_url          STRING,
    git_commit          STRING
"""

_OUTCOMES_SCHEMA = """
    run_id            STRING,
    question_id       STRING,
    scorer_name       STRING,
    category          STRING,
    ticker            STRING,
    fiscal_year       INT,
    fiscal_quarter    INT,
    difficulty        STRING,
    outcome           STRING,
    value_numeric     DOUBLE,
    rationale         STRING,
    error_message     STRING,
    agent_response    STRING,
    expected_response STRING
"""


def merge_run_summary(
    spark,
    *,
    run_id: str,
    run_name: str,
    experiment_id: str,
    run_started_at_ms: int,
    run_finished_at_ms: int,
    agent_endpoint: str,
    agent_version: str | None,
    judge_endpoint: str,
    num_questions: int,
    dataset_path: str,
    dataset_hash: str,
    scorer_metrics: dict,
    params: dict,
    mlflow_url: str,
    git_commit: str | None,
) -> None:
    """Idempotent MERGE into eval_run_summaries on run_id.

    Uses an explicit DDL schema because Spark Connect cannot infer types from
    a single-row tuple containing None values (e.g. agent_version, git_commit).
    """
    from pyspark.sql import functions as F

    duration_seconds = max(0.0, (run_finished_at_ms - run_started_at_ms) / 1000.0)
    tuple_row = [(
        run_id,
        run_name,
        experiment_id,
        int(run_started_at_ms),
        int(run_finished_at_ms),
        float(duration_seconds),
        agent_endpoint,
        agent_version,
        judge_endpoint,
        int(num_questions),
        dataset_path,
        dataset_hash,
        json.dumps(scorer_metrics, sort_keys=True),
        json.dumps(params, sort_keys=True, default=str),
        mlflow_url,
        git_commit,
    )]
    df = (
        spark.createDataFrame(tuple_row, schema=_RUN_SUMMARY_SCHEMA)
        .withColumn("run_started_at",  (F.col("run_started_at_ms")  / 1000).cast("timestamp"))
        .withColumn("run_finished_at", (F.col("run_finished_at_ms") / 1000).cast("timestamp"))
        .drop("run_started_at_ms", "run_finished_at_ms")
        .withColumn("recorded_at", F.current_timestamp())
    )
    df.createOrReplaceTempView("_finsage_eval_summary_src")
    spark.sql(f"""
        MERGE INTO {EVAL_SUMMARY_TABLE} t
        USING _finsage_eval_summary_src s
        ON t.run_id = s.run_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)


def merge_question_outcomes(spark, run_id: str, rows: list[dict]) -> None:
    """Idempotent MERGE into eval_question_outcomes on (run_id, qid, scorer).

    Explicit DDL schema avoids Spark Connect CANNOT_DETERMINE_TYPE failures
    on the many nullable columns (fiscal_year, fiscal_quarter, value_numeric,
    rationale, error_message, agent_response, expected_response).
    """
    if not rows:
        return
    from pyspark.sql import functions as F

    tuples = [
        (
            run_id,
            r["question_id"],
            r["scorer_name"],
            r.get("category"),
            r.get("ticker"),
            int(r["fiscal_year"]) if r.get("fiscal_year") is not None else None,
            int(r["fiscal_quarter"]) if r.get("fiscal_quarter") is not None else None,
            r.get("difficulty"),
            r["outcome"],
            float(r["value_numeric"]) if r.get("value_numeric") is not None else None,
            r.get("rationale"),
            r.get("error_message"),
            r.get("agent_response"),
            r.get("expected_response"),
        )
        for r in rows
    ]
    df = (
        spark.createDataFrame(tuples, schema=_OUTCOMES_SCHEMA)
        .withColumn("recorded_at", F.current_timestamp())
    )
    df.createOrReplaceTempView("_finsage_eval_outcomes_src")
    spark.sql(f"""
        MERGE INTO {EVAL_OUTCOMES_TABLE} t
        USING _finsage_eval_outcomes_src s
        ON t.run_id = s.run_id
           AND t.question_id = s.question_id
           AND t.scorer_name = s.scorer_name
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Readers — used by analysis.py and dashboards
# ─────────────────────────────────────────────────────────────────────────────

def fetch_recent_runs(spark, limit: int = 10):
    return spark.sql(f"""
        SELECT run_id, run_name, run_started_at, duration_seconds,
               agent_version, num_questions, scorer_metrics_json,
               dataset_hash, mlflow_url, git_commit
        FROM {EVAL_SUMMARY_TABLE}
        ORDER BY run_started_at DESC
        LIMIT {int(limit)}
    """)


def fetch_outcomes_for_run(spark, run_id: str):
    return spark.sql(f"""
        SELECT *
        FROM {EVAL_OUTCOMES_TABLE}
        WHERE run_id = '{run_id}'
        ORDER BY scorer_name, question_id
    """)
