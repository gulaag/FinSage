"""FinSage custom MLflow GenAI scorers.

Five custom scorers, each producing a binary or categorical signal that
complements the LLM-judge `Correctness` and `Guidelines` scorers:

    numerical_tolerance       — value-vs-expected within ±1%, unit-aware
    citation_format           — [VERBATIM]/[SUMMARY] + [Source: TICKER | FY...] line
    refusal_correctness       — agent declined for the right reason on F-category
    tool_routing_correctness  — agent picked the right tool (annual / quarterly /
                                metadata / search) for the question
    derived_metric_match      — extracted numeric value matches ground-truth
                                provenance value with relative tolerance, decoupled
                                from natural-language wording

All scorers follow MLflow 3.x's scorer contract:
    @scorer
    def name(*, inputs=None, outputs=None, expectations=None, trace=None, **kwargs):
        ...
        return bool          # PASS / FAIL
        return None          # SKIP (not applicable to this row)
        return Feedback(...) # rich rationale (optional)

The scorers are pure Python and importable outside Databricks for unit testing.
"""

from __future__ import annotations

import re
from typing import Iterable

try:
    from mlflow.genai.scorers import scorer
except ImportError:  # pragma: no cover — only happens outside MLflow runtime
    def scorer(fn=None, **_kwargs):  # type: ignore[no-redef]
        if fn is None:
            return lambda f: f
        return fn

try:
    from mlflow.entities import Feedback  # MLflow 3.x rich-feedback object
except ImportError:  # pragma: no cover
    Feedback = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# Number extraction (shared by numerical_tolerance and derived_metric_match)
# ─────────────────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9])([-+]?\$?\s*\d[\d,]*(?:\.\d+)?)\s*(billion|million|thousand|bn|mn|k|b|m|%|x)?",
    re.IGNORECASE,
)

_UNIT_MULTIPLIERS = {
    "billion": 1_000_000_000, "bn": 1_000_000_000, "b": 1_000_000_000,
    "million": 1_000_000,     "mn": 1_000_000,     "m": 1_000_000,
    "thousand": 1_000,        "k": 1_000,
}


def extract_numbers(text: str) -> list[float]:
    """Extract numeric values from free-form text with B/M/K unit awareness.

    Percent ('%') and 'x' (multiple) are kept as-is. Year-like values
    (1900-2100) are kept too — caller is responsible for filtering them out
    when an alternative is available.
    """
    out: list[float] = []
    for m in _NUM_RE.finditer(text or ""):
        token = m.group(1).replace("$", "").replace(",", "").strip()
        try:
            val = float(token)
        except ValueError:
            continue
        unit = (m.group(2) or "").lower()
        if unit in _UNIT_MULTIPLIERS:
            val *= _UNIT_MULTIPLIERS[unit]
        out.append(val)
    return out


def best_predicted_number(candidates: list[float], target: float) -> float | None:
    """Pick the predicted value closest to target by relative error.

    Filters out obvious fiscal-year tokens (1900–2100) when alternatives exist
    to avoid the agent's "FY2024" being mistaken for the answer.
    """
    if not candidates:
        return None
    filtered = [v for v in candidates if not (1900 <= abs(v) <= 2100)]
    pool = filtered if filtered else candidates
    if target == 0:
        return pool[0]
    return min(pool, key=lambda x: abs(x - target) / abs(target))


def _response_text(outputs) -> str:
    """Extract assistant content from a chat-completion-shaped response."""
    if isinstance(outputs, dict):
        try:
            return outputs["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return str(outputs)
    return str(outputs or "")


def _question_text(inputs) -> str:
    """Extract user question from a chat-completion-shaped input."""
    if isinstance(inputs, dict):
        msgs = inputs.get("messages") or []
        if msgs and isinstance(msgs[-1], dict):
            return msgs[-1].get("content", "") or ""
    return str(inputs or "")


def _wrap(value: bool, rationale: str) -> object:
    """Return a Feedback if available, else fall back to bool. MLflow handles
    both; Feedback adds the rationale string to the assessment."""
    if Feedback is not None:
        return Feedback(value=value, rationale=rationale)
    return value


# ─────────────────────────────────────────────────────────────────────────────
# Scorer 1: numerical_tolerance — ±1% on numerical_lookup
# ─────────────────────────────────────────────────────────────────────────────

@scorer
def numerical_tolerance(*, inputs=None, outputs=None, expectations=None, trace=None, **_):
    exp = expectations or {}
    if exp.get("category") != "numerical_lookup":
        return None
    response = _response_text(outputs)
    expected_text = exp.get("expected_response", "")

    pred_candidates = extract_numbers(response)
    true_candidates = extract_numbers(expected_text)
    if not true_candidates:
        return _wrap(False, "Could not parse a number from expected_response.")
    target = true_candidates[0]
    pred = best_predicted_number(pred_candidates, target)
    if pred is None or target == 0:
        return _wrap(False, f"No numeric value extractable from response (target={target}).")

    rel_err = abs(pred - target) / abs(target)
    passed = rel_err <= 0.01
    return _wrap(passed, f"pred={pred:.4f} target={target:.4f} rel_err={rel_err:.4%}")


# ─────────────────────────────────────────────────────────────────────────────
# Scorer 2: citation_format — [VERBATIM]/[SUMMARY] + [Source: ...] line
# ─────────────────────────────────────────────────────────────────────────────

_CITATION_TAG = re.compile(r"\[(VERBATIM|SUMMARY)\]", re.IGNORECASE)
_SOURCE_LINE  = re.compile(r"\[Source:\s*[A-Z]+\s*\|\s*FY\d{4}", re.IGNORECASE)


@scorer
def citation_format(*, inputs=None, outputs=None, expectations=None, trace=None, **_):
    exp = expectations or {}
    cat = exp.get("category")
    if cat not in ("risk_summary", "citation_validation", "yoy_comparison", "numerical_lookup"):
        return None
    if exp.get("source_section") == "metrics":
        return None  # metrics-tool answers don't require [VERBATIM]/[SUMMARY]
    response = _response_text(outputs)
    has_tag    = bool(_CITATION_TAG.search(response))
    has_source = bool(_SOURCE_LINE.search(response))
    passed = has_tag and has_source
    return _wrap(passed, f"verbatim/summary tag={has_tag}, [Source:..] line={has_source}")


# ─────────────────────────────────────────────────────────────────────────────
# Scorer 3: refusal_correctness — agent declined for the right reason
# ─────────────────────────────────────────────────────────────────────────────

_REFUSAL_PHRASES = (
    "cannot", "can't", "cant", "unable", "not available", "not present",
    "not in the corpus", "out of scope", "outside", "no data", "no record",
    "decline", "i'm sorry", "i am sorry", "do not have", "don't have",
    "not supported", "not stored", "not covered",
)

# Per-question expected refusal-context tokens. The agent must mention at
# least one of these to demonstrate it identified the SPECIFIC reason for
# refusal, not just any generic refusal.
_REFUSAL_CONTEXT_TOKENS: dict[str, tuple[str, ...]] = {
    "F001": ("2030", "future", "not yet", "has not"),
    "F002": ("IBM", "corpus", "30 ", "thirty", "not tracked", "not in"),
    "F003": ("Q4", "quarter 4", "fourth quarter", "Q1", "Q2", "Q3", "annual"),
    "F004": ("MCD", "McDonald", "10-K", "section", "extract", "silver", "gap"),
    "F005": ("FB", "Meta", "META", "GOOG", "GOOGL", "ticker", "ambiguous"),
}


@scorer
def refusal_correctness(*, inputs=None, outputs=None, expectations=None, trace=None, **_):
    exp = expectations or {}
    if exp.get("category") != "refusal_test":
        return None
    qid = exp.get("question_id", "")
    response = _response_text(outputs).lower()

    refused = any(phrase in response for phrase in _REFUSAL_PHRASES)
    expected_tokens = _REFUSAL_CONTEXT_TOKENS.get(qid, ())
    cited_reason = (
        not expected_tokens
        or any(tok.lower() in response for tok in expected_tokens)
    )

    passed = refused and cited_reason
    rationale = (
        f"refused={refused}, cited_specific_reason={cited_reason} "
        f"(expected tokens: {expected_tokens or 'any'})"
    )
    return _wrap(passed, rationale)


# ─────────────────────────────────────────────────────────────────────────────
# Scorer 4: tool_routing_correctness — agent used the right tool
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that signal which tool the agent's answer came from.
_TOOL_SIGNALS = {
    "annual_metrics":  re.compile(r"get_company_metrics|FY\d{4}\s*\|\s*metrics|Annual financial", re.IGNORECASE),
    "quarterly_metrics": re.compile(r"get_quarterly_metrics|FY\d{4}\s*Q\d|Quarterly financial", re.IGNORECASE),
    "filing_metadata": re.compile(r"get_filing_metadata|10-K Cover|Cover Page|filing date|shares outstanding|employees", re.IGNORECASE),
    "search_filings":  re.compile(r"search_filings|\[Source:\s*[A-Z]+\s*\|\s*FY\d{4}\s*\|\s*(?:10-K|10-Q|MD&A|Risk|Business)", re.IGNORECASE),
}


def _expected_tools(exp: dict) -> tuple[set[str], str]:
    """Return (set of acceptable tool names, rationale_hint) for a row."""
    cat = exp.get("category")
    src = exp.get("source_section", "")
    fiscal_quarter = exp.get("fiscal_quarter")
    qid = (exp.get("question_id") or "").upper()

    # Refusal test — no tool required (agent should not call any)
    if cat == "refusal_test":
        return set(), "refusal — no tool expected"

    # Quarterly questions
    if fiscal_quarter or qid.startswith("B"):
        return {"quarterly_metrics"}, "quarterly question → get_quarterly_metrics"

    # Filing metadata (Cover Page section)
    if src == "10-K Cover Page" or qid.startswith("E"):
        return {"filing_metadata", "search_filings"}, "metadata question → get_filing_metadata or search_filings"

    # Pure-metrics annual questions
    if src == "metrics":
        return {"annual_metrics"}, "annual metrics question → get_company_metrics"

    # Narrative/qualitative
    if cat in ("risk_summary", "citation_validation"):
        return {"search_filings"}, "narrative question → search_filings"

    # YoY / multi-company → annual metrics typically
    if cat in ("yoy_comparison", "multi_company"):
        return {"annual_metrics"}, "comparison question → get_company_metrics"

    return set(), "no expected tool inferred"


@scorer
def tool_routing_correctness(*, inputs=None, outputs=None, expectations=None, trace=None, **_):
    exp = expectations or {}
    expected, hint = _expected_tools(exp)
    if not expected:
        return None  # skip categories without a deterministic tool expectation

    response = _response_text(outputs)
    detected = {name for name, pat in _TOOL_SIGNALS.items() if pat.search(response)}

    passed = bool(expected & detected)
    rationale = f"{hint} | detected={sorted(detected) or 'none'} | expected={sorted(expected)}"
    return _wrap(passed, rationale)


# ─────────────────────────────────────────────────────────────────────────────
# Scorer 5: derived_metric_match — extract value, compare to provenance with tolerance
# ─────────────────────────────────────────────────────────────────────────────

# Categories whose answer is a single numeric value where natural-language
# formatting differences shouldn't matter.
_DERIVED_CATEGORIES = ("numerical_lookup", "yoy_comparison", "multi_company")


@scorer
def derived_metric_match(*, inputs=None, outputs=None, expectations=None, trace=None, **_):
    """Tolerance-based match decoupled from natural-language wording.

    Reads the canonical numeric value from `expectations.expected_response`
    (the ground-truth row's expected_answer always leads with the canonical
    number) and matches the agent's closest extracted value. Tolerance is
    relative: 1% for absolute values, 0.005 absolute for percentages already
    expressed as decimals (e.g. d/e ratio of 0.5 vs 0.501).
    """
    exp = expectations or {}
    if exp.get("category") not in _DERIVED_CATEGORIES:
        return None
    expected_text = exp.get("expected_response", "")
    if not expected_text:
        return None

    response = _response_text(outputs)
    pred_candidates = extract_numbers(response)
    true_candidates = extract_numbers(expected_text)
    if not true_candidates:
        return None  # nothing to compare against
    target = true_candidates[0]
    pred = best_predicted_number(pred_candidates, target)
    if pred is None:
        return _wrap(False, f"no numeric value found in response (target={target})")

    if target == 0:
        passed = abs(pred) < 1e-9
    else:
        rel_err = abs(pred - target) / abs(target)
        # Looser absolute tolerance for small ratios where 1% relative is
        # noisy (e.g. d/e of 0.04 vs 0.05 is 25% relative but 1pp absolute).
        passed = rel_err <= 0.01 or abs(pred - target) <= 0.005
    rel_err_pct = abs(pred - target) / max(abs(target), 1e-9) * 100
    return _wrap(passed, f"pred={pred:.6g} target={target:.6g} rel_err={rel_err_pct:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Public scorer set — what 07_evaluation.py imports
# ─────────────────────────────────────────────────────────────────────────────

ALL_CUSTOM_SCORERS = (
    numerical_tolerance,
    citation_format,
    refusal_correctness,
    tool_routing_correctness,
    derived_metric_match,
)
