"""Pre-flight validation for the eval pipeline.

Catches bugs that would otherwise only surface on a Databricks cluster after
several minutes of pip-install + restartPython + model-load. Run locally:

    .venv/bin/python -m pytest tests/unit/test_eval_preflight.py -v

What it validates (no Spark / no MLflow runtime needed):

  • Ground-truth dataset:
      - Loads cleanly, has 100 rows
      - Every row has the required fields used by to_eval_row + scorers
      - Categories are within the supported set
      - source_section is one of the allowed values
      - expected_answer is non-empty
      - Refusal questions have question_id F001-F005 (matches scorers contract)

  • Eval-row construction (the to_eval_row logic):
      - No None values land in the expectations dict (MLflow Expectation rejects
        None — this is the "fiscal_quarter=None" bug)
      - Exactly one of expected_response / expected_facts is set per row
      - inputs.messages is a non-empty list of dicts with role+content

  • Scorer wiring:
      - All 6 custom scorers are importable
      - Each scorer's signature accepts (*, inputs, outputs, expectations, trace,
        **kwargs) and returns bool / None / Feedback (no raises on standard input)
      - Custom scorer return types are MLflow-compatible

  • Persistence schema:
      - DDL strings parse cleanly
      - Column names match what merge_run_summary / merge_question_outcomes write
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GROUND_TRUTH_PATH = ROOT / "src/evaluation/ground_truth_v2.json"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset shape
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ground_truth():
    assert GROUND_TRUTH_PATH.exists(), f"missing dataset {GROUND_TRUTH_PATH}"
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


REQUIRED_FIELDS = {
    "question_id", "category", "ticker", "difficulty",
    "question", "expected_answer", "evidence_passage",
    "source_doc", "source_section",
}
ALLOWED_CATEGORIES = {
    "numerical_lookup", "yoy_comparison", "multi_company",
    "citation_validation", "risk_summary", "refusal_test",
}
ALLOWED_SECTIONS = {
    "metrics", "10-K Cover Page", "MD&A", "Risk Factors",
    "Risk Factors Updates", "Business",
}


def test_dataset_size(ground_truth):
    assert len(ground_truth) == 100, f"expected 100 questions, got {len(ground_truth)}"


def test_dataset_unique_question_ids(ground_truth):
    ids = [q["question_id"] for q in ground_truth]
    assert len(set(ids)) == len(ids), "duplicate question_ids"


def test_dataset_required_fields(ground_truth):
    for q in ground_truth:
        missing = REQUIRED_FIELDS - set(q.keys())
        assert not missing, f"{q['question_id']} missing fields {missing}"
        assert q["expected_answer"].strip(), f"{q['question_id']} has empty expected_answer"


def test_dataset_categories_valid(ground_truth):
    for q in ground_truth:
        assert q["category"] in ALLOWED_CATEGORIES, \
            f"{q['question_id']} has unknown category {q['category']!r}"


def test_dataset_source_sections_valid(ground_truth):
    for q in ground_truth:
        assert q["source_section"] in ALLOWED_SECTIONS, \
            f"{q['question_id']} has unknown source_section {q['source_section']!r}"


def test_refusal_question_ids(ground_truth):
    refusals = [q["question_id"] for q in ground_truth if q["category"] == "refusal_test"]
    expected = {"F001", "F002", "F003", "F004", "F005"}
    assert set(refusals) == expected, \
        f"refusal IDs drifted; expected {expected}, got {set(refusals)} — keep in sync with scorers._REFUSAL_CONTEXT_TOKENS"


# ─────────────────────────────────────────────────────────────────────────────
# Eval-row construction (replicates to_eval_row from 07_evaluation.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_eval_row(q):
    """Mirror of databricks/notebooks/07_evaluation.py:to_eval_row."""
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


def test_no_none_in_expectations(ground_truth):
    """The bug from run 482dd7a3: Expectation(value=None) raises MlflowException."""
    for q in ground_truth:
        row = _build_eval_row(q)
        for key, value in row["expectations"].items():
            assert value is not None, \
                f"{q['question_id']} produces None expectation for key={key!r} — would crash mlflow.genai.evaluate"


def test_exactly_one_expected_response_or_facts(ground_truth):
    for q in ground_truth:
        exps = _build_eval_row(q)["expectations"]
        has_resp = "expected_response" in exps
        has_facts = "expected_facts" in exps
        assert has_resp ^ has_facts, \
            f"{q['question_id']} has both/neither expected_response and expected_facts"


def test_inputs_shape(ground_truth):
    for q in ground_truth:
        row = _build_eval_row(q)
        msgs = row["inputs"]["messages"]
        assert isinstance(msgs, list) and len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert isinstance(msgs[0]["content"], str) and msgs[0]["content"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Scorer wiring
# ─────────────────────────────────────────────────────────────────────────────

def test_all_custom_scorers_importable():
    from src.evaluation.scorers import ALL_CUSTOM_SCORERS
    assert len(ALL_CUSTOM_SCORERS) >= 6, f"expected ≥6 scorers, got {len(ALL_CUSTOM_SCORERS)}"


def test_every_scorer_handles_minimal_input(ground_truth):
    """Smoke each scorer with a minimal synthetic input. Catches signature
    regressions without needing MLflow's evaluate harness."""
    from src.evaluation.scorers import ALL_CUSTOM_SCORERS

    sample_q = ground_truth[0]  # A001
    row = _build_eval_row(sample_q)
    minimal_outputs = {"choices": [{"message": {"role": "assistant", "content": "$7,387,268,000.00"}}]}
    for s in ALL_CUSTOM_SCORERS:
        # Must not raise
        try:
            result = s(
                inputs=row["inputs"],
                outputs=minimal_outputs,
                expectations=row["expectations"],
                trace=None,
            )
        except Exception as e:
            raise AssertionError(f"scorer {s.__name__ if hasattr(s, '__name__') else s} raised: {e}") from e
        # Result must be None (skip), bool, or Feedback-shaped
        if result is None:
            continue
        if isinstance(result, bool):
            continue
        if hasattr(result, "value"):
            continue
        raise AssertionError(f"scorer {s} returned unexpected type {type(result)}")


# ─────────────────────────────────────────────────────────────────────────────
# Persistence schema
# ─────────────────────────────────────────────────────────────────────────────

def test_persistence_schemas_match_ddl():
    from src.evaluation.persistence import (
        _RUN_SUMMARY_SCHEMA, _OUTCOMES_SCHEMA,
        EVAL_SUMMARY_TABLE, EVAL_OUTCOMES_TABLE,
    )
    # Schemas should be non-empty DDL strings with named columns
    for ddl in (_RUN_SUMMARY_SCHEMA, _OUTCOMES_SCHEMA):
        assert "STRING" in ddl
        assert "," in ddl  # multi-column
    assert EVAL_SUMMARY_TABLE.startswith("main.finsage_gold.")
    assert EVAL_OUTCOMES_TABLE.startswith("main.finsage_gold.")


def test_outcome_row_keys_match_schema():
    """The keys we write in 07_evaluation.py:outcome_rows must match the
    columns the persistence module declares in _OUTCOMES_SCHEMA."""
    from src.evaluation.persistence import _OUTCOMES_SCHEMA
    expected_keys = {
        "run_id", "question_id", "scorer_name", "category", "ticker",
        "fiscal_year", "fiscal_quarter", "difficulty", "outcome",
        "value_numeric", "rationale", "error_message",
        "agent_response", "expected_response",
    }
    # Parse column names from the DDL string
    schema_cols = {
        line.strip().split()[0].lower()
        for line in _OUTCOMES_SCHEMA.strip().split(",")
        if line.strip() and not line.strip().startswith("--")
    }
    missing = expected_keys - schema_cols
    extra   = schema_cols - expected_keys
    assert not missing, f"keys missing from DDL: {missing}"
    # `recorded_at` is added by withColumn after createDataFrame, so allow it
    assert not (extra - {"recorded_at"}), f"DDL has unexpected columns: {extra}"


# ─────────────────────────────────────────────────────────────────────────────
# Stratified-sample integrity
# ─────────────────────────────────────────────────────────────────────────────

def test_dataset_category_distribution(ground_truth):
    """Locked-in distribution from the builder: 30/25/15/15/10/5 by question-id
    prefix (A/B/C/D/E/F)."""
    from collections import Counter
    prefix_counts = Counter(q["question_id"][0] for q in ground_truth)
    assert prefix_counts == {"A": 30, "B": 25, "C": 15, "D": 15, "E": 10, "F": 5}, \
        f"taxonomy drifted: {dict(prefix_counts)}"
