"""Unit tests for FinSage MLflow scorers.

Pytest-style. Runs locally without MLflow installed (the @scorer decorator
gracefully degrades when MLflow is missing, see scorers.py top imports).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src/evaluation/...` importable from this test file.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.scorers import (  # noqa: E402
    extract_numbers,
    best_predicted_number,
    numerical_tolerance,
    citation_format,
    refusal_correctness,
    tool_routing_correctness,
    derived_metric_match,
)


def _outputs(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _expectations(**kwargs) -> dict:
    base = {
        "category": "numerical_lookup",
        "ticker": "AAPL",
        "fiscal_year": 2024,
        "difficulty": "easy",
        "source_doc": "AAPL-2024-10K",
        "source_section": "metrics",
        "expected_response": "$391,035 million. Apple FY2024 revenue.",
    }
    base.update(kwargs)
    return base


def _value(scorer_result):
    """Unwrap Feedback-or-bool to a plain truth value."""
    if scorer_result is None:
        return None
    if hasattr(scorer_result, "value"):
        return scorer_result.value
    return scorer_result


# ── extract_numbers / best_predicted_number ─────────────────────────────────

def test_extract_handles_billion_million_units():
    assert extract_numbers("revenue was $391.04B")[0] == 391_040_000_000.0
    assert extract_numbers("net income $97 billion")[0] == 97_000_000_000.0
    assert extract_numbers("$8,675 million in R&D")[0] == 8_675_000_000.0


def test_extract_handles_percent_and_ratio():
    nums = extract_numbers("gross margin was 43.31%, d/e was 0.5x")
    assert 43.31 in nums
    assert 0.5 in nums


def test_best_predicted_filters_year_tokens():
    # When agent says "FY2024 revenue was $391B", we want $391B not 2024.
    candidates = [2024.0, 391_000_000_000.0]
    assert best_predicted_number(candidates, target=391_035_000_000.0) == 391_000_000_000.0


# ── numerical_tolerance ─────────────────────────────────────────────────────

def test_numerical_tolerance_passes_within_one_pct():
    out = _outputs("Apple's revenue was $391.04B.")
    exp = _expectations(expected_response="$391,035,000,000. Apple FY2024 revenue.")
    assert _value(numerical_tolerance(outputs=out, expectations=exp)) is True


def test_numerical_tolerance_fails_outside_tolerance():
    out = _outputs("Apple's revenue was $400B.")
    exp = _expectations(expected_response="$391,035,000,000. Apple FY2024 revenue.")
    assert _value(numerical_tolerance(outputs=out, expectations=exp)) is False


def test_numerical_tolerance_skips_other_categories():
    out = _outputs("anything")
    exp = _expectations(category="refusal_test")
    assert numerical_tolerance(outputs=out, expectations=exp) is None


# ── citation_format ─────────────────────────────────────────────────────────

def test_citation_format_passes_with_both_tags():
    out = _outputs("[VERBATIM] some text\n[Source: AAPL | FY2024 | 10-K | Risk Factors]")
    exp = _expectations(category="risk_summary", source_section="Risk Factors")
    assert _value(citation_format(outputs=out, expectations=exp)) is True


def test_citation_format_skips_metrics_section():
    out = _outputs("any text")
    exp = _expectations(category="numerical_lookup", source_section="metrics")
    assert citation_format(outputs=out, expectations=exp) is None


def test_citation_format_fails_without_tags():
    out = _outputs("just some text without citations")
    exp = _expectations(category="risk_summary", source_section="Risk Factors")
    assert _value(citation_format(outputs=out, expectations=exp)) is False


# ── refusal_correctness ─────────────────────────────────────────────────────

def test_refusal_correctness_passes_with_phrase_and_context():
    out = _outputs("I cannot answer this question because IBM is not in the FinSage corpus.")
    exp = _expectations(category="refusal_test", question_id="F002")
    assert _value(refusal_correctness(outputs=out, expectations=exp)) is True


def test_refusal_correctness_fails_without_specific_reason():
    out = _outputs("I cannot answer this.")
    exp = _expectations(category="refusal_test", question_id="F002")
    # Generic refusal without naming IBM/corpus → fail
    assert _value(refusal_correctness(outputs=out, expectations=exp)) is False


def test_refusal_correctness_skips_non_refusal_categories():
    out = _outputs("I cannot answer this.")
    exp = _expectations(category="numerical_lookup")
    assert refusal_correctness(outputs=out, expectations=exp) is None


# ── tool_routing_correctness ────────────────────────────────────────────────

def test_routing_passes_for_quarterly_question():
    out = _outputs("Net income $X. [Source: AMZN | FY2024 Q1 | Quarterly financial metrics]")
    exp = _expectations(
        category="numerical_lookup",
        source_section="metrics",
        fiscal_quarter=1,
        question_id="B002",
    )
    assert _value(tool_routing_correctness(outputs=out, expectations=exp)) is True


def test_routing_fails_when_agent_uses_annual_for_quarterly_question():
    out = _outputs("Annual net income. [Source: AMZN | FY2024 | metrics]")
    exp = _expectations(
        category="numerical_lookup",
        source_section="metrics",
        fiscal_quarter=1,
        question_id="B002",
    )
    assert _value(tool_routing_correctness(outputs=out, expectations=exp)) is False


def test_routing_skips_refusal():
    out = _outputs("I cannot answer.")
    exp = _expectations(category="refusal_test")
    assert tool_routing_correctness(outputs=out, expectations=exp) is None


# ── derived_metric_match ────────────────────────────────────────────────────

def test_derived_match_passes_on_pct_close():
    out = _outputs("59.03%. Snowflake FY2021 gross margin.")
    exp = _expectations(
        category="numerical_lookup",
        expected_response="59.03%. Snowflake Inc.'s FY2021 gross margin was $349,461,000.00 / $592,049,000.00.",
    )
    assert _value(derived_metric_match(outputs=out, expectations=exp)) is True


def test_derived_match_passes_on_small_ratio_absolute_tolerance():
    # 0.045 vs 0.043 — 4.6% relative but 0.002 absolute → should pass
    out = _outputs("Tesla d/e was 0.045")
    exp = _expectations(expected_response="0.043. Tesla d/e ratio")
    assert _value(derived_metric_match(outputs=out, expectations=exp)) is True


def test_derived_match_fails_on_wrong_value():
    out = _outputs("Net income was $99B.")
    exp = _expectations(expected_response="$391,035,000,000. Revenue not net income.")
    # Compare 99e9 to 391e9 → way off, fail
    assert _value(derived_metric_match(outputs=out, expectations=exp)) is False
