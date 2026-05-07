"""Eval failure analysis + cross-run regression detection.

Pure-Spark utilities that read from the persistence Delta tables and produce
human-readable summaries:

    summarize_run         — single-run health: per-scorer pass/fail, errors
    failure_breakdown     — categorize failing questions by scorer + category
                            with the agent's actual response and expected text
    regression_diff       — compare two runs (default: latest vs. previous)
                            and surface scorer deltas + question-level flips

These return Spark DataFrames so they can be displayed via `display(df)` inside
notebooks or `.toPandas()` for ad-hoc analysis.
"""

from __future__ import annotations

import json
from typing import Iterable

from .persistence import EVAL_OUTCOMES_TABLE, EVAL_SUMMARY_TABLE


def summarize_run(spark, run_id: str):
    """One row per scorer with PASS/FAIL/ERROR/SKIP counts and pass-rate."""
    return spark.sql(f"""
        SELECT
            scorer_name,
            COUNT(*) AS total,
            SUM(CASE WHEN outcome = 'PASS'  THEN 1 ELSE 0 END) AS passed,
            SUM(CASE WHEN outcome = 'FAIL'  THEN 1 ELSE 0 END) AS failed,
            SUM(CASE WHEN outcome = 'ERROR' THEN 1 ELSE 0 END) AS errored,
            SUM(CASE WHEN outcome = 'SKIP'  THEN 1 ELSE 0 END) AS skipped,
            ROUND(
                SUM(CASE WHEN outcome = 'PASS' THEN 1 ELSE 0 END)
                / NULLIF(SUM(CASE WHEN outcome IN ('PASS','FAIL') THEN 1 ELSE 0 END), 0),
                4
            ) AS pass_rate
        FROM {EVAL_OUTCOMES_TABLE}
        WHERE run_id = '{run_id}'
        GROUP BY scorer_name
        ORDER BY scorer_name
    """)


def failure_breakdown(spark, run_id: str, scorer_name: str | None = None):
    """List all failing questions for one run with full context."""
    where = f"run_id = '{run_id}' AND outcome IN ('FAIL','ERROR')"
    if scorer_name:
        where += f" AND scorer_name = '{scorer_name}'"
    return spark.sql(f"""
        SELECT
            question_id, scorer_name, outcome, category, ticker, fiscal_year,
            fiscal_quarter, difficulty, rationale,
            SUBSTRING(agent_response,    1, 800) AS agent_response,
            SUBSTRING(expected_response, 1, 800) AS expected_response
        FROM {EVAL_OUTCOMES_TABLE}
        WHERE {where}
        ORDER BY scorer_name, question_id
    """)


def category_matrix(spark, run_id: str):
    """Per-category × per-scorer pass/fail matrix."""
    return spark.sql(f"""
        SELECT
            category,
            scorer_name,
            COUNT(*) AS total,
            SUM(CASE WHEN outcome = 'PASS' THEN 1 ELSE 0 END) AS passed,
            ROUND(
                SUM(CASE WHEN outcome = 'PASS' THEN 1 ELSE 0 END)
                / NULLIF(SUM(CASE WHEN outcome IN ('PASS','FAIL') THEN 1 ELSE 0 END), 0),
                3
            ) AS pass_rate
        FROM {EVAL_OUTCOMES_TABLE}
        WHERE run_id = '{run_id}'
        GROUP BY category, scorer_name
        ORDER BY category, scorer_name
    """)


def regression_diff(spark, current_run_id: str, baseline_run_id: str | None = None):
    """Per-scorer pass-rate delta vs. a baseline run.

    If `baseline_run_id` is None, the most-recent prior run for the same agent
    endpoint is used.
    """
    if baseline_run_id is None:
        prior = spark.sql(f"""
            SELECT run_id FROM {EVAL_SUMMARY_TABLE}
            WHERE run_id != '{current_run_id}'
              AND agent_endpoint = (SELECT agent_endpoint
                                    FROM {EVAL_SUMMARY_TABLE}
                                    WHERE run_id = '{current_run_id}')
            ORDER BY run_started_at DESC
            LIMIT 1
        """).collect()
        if not prior:
            return spark.createDataFrame([], "scorer_name STRING, current_pass_rate DOUBLE, baseline_pass_rate DOUBLE, delta DOUBLE")
        baseline_run_id = prior[0]["run_id"]

    return spark.sql(f"""
        WITH curr AS (
            SELECT scorer_name,
                   SUM(CASE WHEN outcome = 'PASS' THEN 1 ELSE 0 END)
                   / NULLIF(SUM(CASE WHEN outcome IN ('PASS','FAIL') THEN 1 ELSE 0 END), 0) AS pass_rate
            FROM {EVAL_OUTCOMES_TABLE}
            WHERE run_id = '{current_run_id}'
            GROUP BY scorer_name
        ),
        base AS (
            SELECT scorer_name,
                   SUM(CASE WHEN outcome = 'PASS' THEN 1 ELSE 0 END)
                   / NULLIF(SUM(CASE WHEN outcome IN ('PASS','FAIL') THEN 1 ELSE 0 END), 0) AS pass_rate
            FROM {EVAL_OUTCOMES_TABLE}
            WHERE run_id = '{baseline_run_id}'
            GROUP BY scorer_name
        )
        SELECT
            COALESCE(curr.scorer_name, base.scorer_name) AS scorer_name,
            ROUND(curr.pass_rate, 4) AS current_pass_rate,
            ROUND(base.pass_rate, 4) AS baseline_pass_rate,
            ROUND(curr.pass_rate - base.pass_rate, 4) AS delta
        FROM curr FULL OUTER JOIN base USING (scorer_name)
        ORDER BY scorer_name
    """)


def question_flips(spark, current_run_id: str, baseline_run_id: str):
    """Questions whose outcome flipped between the two runs (regressions /
    fixes)."""
    return spark.sql(f"""
        WITH curr AS (
            SELECT question_id, scorer_name, outcome AS curr_outcome,
                   agent_response AS curr_response
            FROM {EVAL_OUTCOMES_TABLE}
            WHERE run_id = '{current_run_id}'
        ),
        base AS (
            SELECT question_id, scorer_name, outcome AS base_outcome
            FROM {EVAL_OUTCOMES_TABLE}
            WHERE run_id = '{baseline_run_id}'
        )
        SELECT
            curr.question_id, curr.scorer_name,
            base.base_outcome, curr.curr_outcome,
            CASE
                WHEN base.base_outcome = 'PASS' AND curr.curr_outcome = 'FAIL' THEN 'REGRESSION'
                WHEN base.base_outcome = 'FAIL' AND curr.curr_outcome = 'PASS' THEN 'FIX'
                ELSE 'NEUTRAL'
            END AS movement,
            SUBSTRING(curr.curr_response, 1, 400) AS agent_response
        FROM curr JOIN base USING (question_id, scorer_name)
        WHERE curr.curr_outcome != base.base_outcome
        ORDER BY movement DESC, scorer_name, question_id
    """)


def print_summary(spark, run_id: str) -> None:
    """One-shot pretty-print used by the eval notebook's final cell."""
    print("=" * 86)
    print(f"EVAL RUN SUMMARY  run_id={run_id}")
    print("=" * 86)
    rows = summarize_run(spark, run_id).collect()
    for r in rows:
        rate = f"{r['pass_rate']:.1%}" if r["pass_rate"] is not None else "  n/a"
        print(
            f"  {r['scorer_name']:30s} pass={r['passed']:3d}  fail={r['failed']:3d}  "
            f"err={r['errored']:3d}  skip={r['skipped']:3d}  rate={rate}"
        )

    print("\nPER-CATEGORY × SCORER:")
    cat_rows = category_matrix(spark, run_id).collect()
    by_cat: dict[str, list] = {}
    for r in cat_rows:
        by_cat.setdefault(r["category"], []).append(r)
    for cat in sorted(by_cat):
        print(f"  [{cat}]")
        for r in by_cat[cat]:
            rate = f"{r['pass_rate']:.1%}" if r["pass_rate"] is not None else "  n/a"
            print(f"    {r['scorer_name']:30s} {r['passed']:3d}/{r['total']:3d}  rate={rate}")
    print("=" * 86)
