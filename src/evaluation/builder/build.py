"""Deterministic ground-truth builder for FinSage evaluation questions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from evaluation.builder.concepts import (  # noqa: E402
    ANNUAL_METRICS,
    QUARTERLY_METRICS,
    CANONICAL_METRICS,
    get_metric_label,
    metric_is_eligible,
)
from evaluation.builder.sources import (  # noqa: E402
    ChosenValue,
    SecClient,
    YFinanceClient,
    cross_validated_value,
    get_10k_filing_date,
    select_sec_fact,
)
from evaluation.builder.templates import (  # noqa: E402
    annual_question,
    build_annual_expected_answer,
    build_comparison_expected_answer,
    build_cross_validation_suffix,
    build_de_ratio_expected_answer,
    build_employees_expected_answer,
    build_evidence_passage,
    build_filing_date_expected_answer,
    build_growth_expected_answer,
    build_margin_expected_answer,
    build_quarterly_expected_answer,
    build_shares_expected_answer,
    choose_comparison_winner,
    comparison_question,
    debt_to_equity_question,
    employees_question,
    filing_date_question,
    gross_margin_question,
    growth_question,
    quarterly_question,
    shares_question,
)
from evaluation.builder.tickers import ORDERED_TICKERS, TICKER_REGISTRY, TickerRecord  # noqa: E402

QUESTION_TARGETS = {"A": 30, "B": 25, "C": 15, "D": 15, "E": 10, "F": 5}
YEARS = [2021, 2022, 2023, 2024, 2025]
PAIR_CANDIDATES: Sequence[Tuple[str, str]] = (
    ("BAC", "JPM"),
    ("AAPL", "MSFT"),
    ("F", "GM"),
    ("ABBV", "PFE"),
    ("KO", "SBUX"),
    ("NVDA", "AMZN"),
    ("V", "MA"),
    ("CRM", "DDOG"),
    ("NET", "SNOW"),
    ("TSLA", "RIVN"),
    ("MRK", "JNJ"),
    ("GOOGL", "MSFT"),
    ("WMT", "NKE"),
    ("GS", "BAC"),
    ("PLTR", "CRM"),
)


def _difficulty_for_metric(metric: str, quarterly: bool = False) -> str:
    if metric in {"revenue", "net_income"} and not quarterly:
        return "easy"
    if metric in {"total_assets", "total_liabilities", "total_equity", "operating_cash_flow"}:
        return "medium"
    return "hard"


def _provenance_from_chosen(chosen: ChosenValue, seed: int, generation_template: str) -> Dict[str, object]:
    return {
        "source_primary": "sec_companyconcept",
        "source_primary_value": chosen.value,
        "source_primary_concept": chosen.concept,
        "source_primary_form": chosen.form,
        "source_primary_accn": chosen.accn,
        "source_primary_filed": chosen.filed,
        "source_secondary": chosen.source_secondary,
        "source_secondary_value": chosen.secondary_value,
        "agreement": chosen.agreement,
        "diff_pct": chosen.diff_pct,
        "generation_seed": seed,
        "generation_template": generation_template,
    }


def _empty_provenance(seed: int, generation_template: str) -> Dict[str, object]:
    return {
        "source_primary": None,
        "source_primary_value": None,
        "source_primary_concept": None,
        "source_primary_form": None,
        "source_primary_accn": None,
        "source_primary_filed": None,
        "source_secondary": None,
        "source_secondary_value": None,
        "agreement": None,
        "diff_pct": None,
        "generation_seed": seed,
        "generation_template": generation_template,
    }


def _append_question(questions: List[Dict[str, object]], q: Dict[str, object]) -> None:
    if not q.get("expected_answer"):
        return
    questions.append(q)


class ValueResolver:
    """Shared SEC + yfinance resolution cache."""

    def __init__(self, sec: SecClient, yf_client: YFinanceClient) -> None:
        self.sec = sec
        self.yf_client = yf_client
        self._annual_cache: Dict[Tuple[str, str, int], Optional[ChosenValue]] = {}
        self._quarter_cache: Dict[Tuple[str, str, int, int], Optional[ChosenValue]] = {}

    def annual(self, ticker_meta: TickerRecord, metric: str, fy: int) -> Optional[ChosenValue]:
        key = (ticker_meta.ticker, metric, fy)
        if key in self._annual_cache:
            return self._annual_cache[key]
        sec_fact = select_sec_fact(self.sec, ticker_meta, metric=metric, fy=fy, form="10-K")
        yv = self.yf_client.get_value(ticker_meta, metric=metric, fy=fy, quarter=None)
        chosen = cross_validated_value(sec_fact=sec_fact, yfinance_value=yv, metric=metric)
        self._annual_cache[key] = chosen
        return chosen

    def quarterly(self, ticker_meta: TickerRecord, metric: str, fy: int, quarter: int) -> Optional[ChosenValue]:
        key = (ticker_meta.ticker, metric, fy, quarter)
        if key in self._quarter_cache:
            return self._quarter_cache[key]
        sec_fact = select_sec_fact(
            self.sec,
            ticker_meta,
            metric=metric,
            fy=fy,
            form="10-Q",
            quarter=quarter,
        )
        yv = self.yf_client.get_value(ticker_meta, metric=metric, fy=fy, quarter=quarter)
        chosen = cross_validated_value(sec_fact=sec_fact, yfinance_value=yv, metric=metric)
        self._quarter_cache[key] = chosen
        return chosen


def build_category_a(
    resolver: ValueResolver,
    seed: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    questions: List[Dict[str, object]] = []
    candidates: List[Tuple[str, int, str]] = []
    for ticker in ORDERED_TICKERS:
        meta = TICKER_REGISTRY[ticker]
        for fy in YEARS:
            for metric in ANNUAL_METRICS:
                if metric_is_eligible(metric, meta.sector):
                    candidates.append((ticker, fy, metric))
    candidates = sorted(candidates)
    rng.shuffle(candidates)

    for ticker, fy, metric in candidates:
        if len(questions) >= QUESTION_TARGETS["A"]:
            break
        meta = TICKER_REGISTRY[ticker]
        chosen = resolver.annual(meta, metric=metric, fy=fy)
        if chosen is None:
            continue
        metric_label = get_metric_label(metric)
        qid = f"A{len(questions) + 1:03d}"
        expected = build_annual_expected_answer(meta.company_name, metric_label, chosen.value, chosen.concept, chosen.filed)
        evidence = (
            build_evidence_passage(metric_label, chosen.value, chosen.concept, chosen.filed)
            + " "
            + build_cross_validation_suffix(chosen.agreement, chosen.diff_pct)
        )
        _append_question(
            questions,
            {
                "question_id": qid,
                "category": "numerical_lookup",
                "ticker": ticker,
                "fiscal_year": fy,
                "difficulty": _difficulty_for_metric(metric),
                "question": annual_question(meta.company_name, metric_label, fy),
                "expected_answer": expected,
                "evidence_passage": evidence,
                "source_doc": f"{ticker}-{fy}-10K",
                "source_section": "metrics",
                "provenance": _provenance_from_chosen(chosen, seed, "annual_lookup"),
            },
        )
    return questions


def build_category_b(
    resolver: ValueResolver,
    seed: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    questions: List[Dict[str, object]] = []
    candidates: List[Tuple[str, int, int, str]] = []
    for ticker in ORDERED_TICKERS:
        meta = TICKER_REGISTRY[ticker]
        for fy in YEARS:
            for quarter in (1, 2, 3):
                for metric in QUARTERLY_METRICS:
                    if metric_is_eligible(metric, meta.sector):
                        candidates.append((ticker, fy, quarter, metric))
    candidates = sorted(candidates)
    rng.shuffle(candidates)

    for ticker, fy, quarter, metric in candidates:
        if len(questions) >= QUESTION_TARGETS["B"]:
            break
        meta = TICKER_REGISTRY[ticker]
        chosen = resolver.quarterly(meta, metric=metric, fy=fy, quarter=quarter)
        if chosen is None:
            continue
        metric_label = get_metric_label(metric)
        qid = f"B{len(questions) + 1:03d}"
        expected = build_quarterly_expected_answer(
            meta.company_name,
            metric_label,
            chosen.value,
            quarter,
            fy,
            chosen.concept,
            chosen.filed,
        )
        evidence = (
            build_evidence_passage(metric_label, chosen.value, chosen.concept, chosen.filed)
            + " "
            + build_cross_validation_suffix(chosen.agreement, chosen.diff_pct)
        )
        _append_question(
            questions,
            {
                "question_id": qid,
                "category": "numerical_lookup",
                "ticker": ticker,
                "fiscal_year": fy,
                "fiscal_quarter": quarter,
                "difficulty": _difficulty_for_metric(metric, quarterly=True),
                "question": quarterly_question(meta.company_name, metric_label, fy, quarter),
                "expected_answer": expected,
                "evidence_passage": evidence,
                "source_doc": f"{ticker}-{fy}Q{quarter}-10Q",
                "source_section": "metrics",
                "provenance": _provenance_from_chosen(chosen, seed, "quarterly_lookup"),
            },
        )
    return questions


def build_category_c(
    resolver: ValueResolver,
    seed: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    questions: List[Dict[str, object]] = []
    items: List[Tuple[str, str, int, int]] = []
    for ticker in ORDERED_TICKERS:
        for fy in YEARS:
            if fy - 1 in YEARS:
                items.append(("growth", ticker, fy - 1, fy))
            items.append(("margin", ticker, fy, fy))
            items.append(("de_ratio", ticker, fy, fy))
    items = sorted(items)
    rng.shuffle(items)

    for item_type, ticker, y1, y2 in items:
        if len(questions) >= QUESTION_TARGETS["C"]:
            break
        meta = TICKER_REGISTRY[ticker]
        qid = f"C{len(questions) + 1:03d}"
        if item_type == "growth":
            rev1 = resolver.annual(meta, "revenue", y1)
            rev2 = resolver.annual(meta, "revenue", y2)
            if rev1 is None or rev2 is None or rev1.value == 0:
                continue
            growth_pct = (rev2.value - rev1.value) / abs(rev1.value) * 100.0
            _append_question(
                questions,
                {
                    "question_id": qid,
                    "category": "yoy_comparison",
                    "ticker": ticker,
                    "fiscal_year": y2,
                    "difficulty": "medium",
                    "question": growth_question(meta.company_name, y1, y2),
                    "expected_answer": build_growth_expected_answer(
                        meta.company_name, y1, y2, rev1.value, rev2.value, growth_pct
                    ),
                    "evidence_passage": (
                        f"Computed from SEC annual revenue values: FY{y1}={rev1.value}, FY{y2}={rev2.value}."
                    ),
                    "source_doc": f"{ticker}-{y2}-10K",
                    "source_section": "metrics",
                    "provenance": {
                        "source_primary": "sec_companyconcept",
                        "source_primary_value": growth_pct,
                        "source_primary_concept": "derived:revenue_growth_pct",
                        "source_primary_form": "10-K",
                        "source_primary_accn": rev2.accn,
                        "source_primary_filed": rev2.filed,
                        "source_secondary": "yfinance"
                        if rev1.secondary_value is not None and rev2.secondary_value is not None
                        else None,
                        "source_secondary_value": (
                            ((rev2.secondary_value - rev1.secondary_value) / abs(rev1.secondary_value) * 100.0)
                            if rev1.secondary_value is not None
                            and rev2.secondary_value is not None
                            and rev1.secondary_value != 0
                            else None
                        ),
                        "agreement": None,
                        "diff_pct": None,
                        "generation_seed": seed,
                        "generation_template": "derived_growth",
                    },
                },
            )
        elif item_type == "margin":
            if not metric_is_eligible("gross_profit", meta.sector):
                continue
            gp = resolver.annual(meta, "gross_profit", y1)
            rev = resolver.annual(meta, "revenue", y1)
            if gp is None or rev is None or rev.value == 0:
                continue
            margin = gp.value / rev.value * 100.0
            _append_question(
                questions,
                {
                    "question_id": qid,
                    "category": "numerical_lookup",
                    "ticker": ticker,
                    "fiscal_year": y1,
                    "difficulty": "hard",
                    "question": gross_margin_question(meta.company_name, y1),
                    "expected_answer": build_margin_expected_answer(
                        meta.company_name, y1, gp.value, rev.value, margin
                    ),
                    "evidence_passage": f"Computed from SEC gross_profit={gp.value} and revenue={rev.value}.",
                    "source_doc": f"{ticker}-{y1}-10K",
                    "source_section": "metrics",
                    "provenance": {
                        "source_primary": "sec_companyconcept",
                        "source_primary_value": margin,
                        "source_primary_concept": "derived:gross_margin_pct",
                        "source_primary_form": "10-K",
                        "source_primary_accn": gp.accn,
                        "source_primary_filed": gp.filed,
                        "source_secondary": None,
                        "source_secondary_value": None,
                        "agreement": None,
                        "diff_pct": None,
                        "generation_seed": seed,
                        "generation_template": "derived_gross_margin",
                    },
                },
            )
        else:
            debt = resolver.annual(meta, "total_debt", y1)
            equity = resolver.annual(meta, "total_equity", y1)
            if debt is None or equity is None or equity.value == 0:
                continue
            ratio = debt.value / equity.value
            _append_question(
                questions,
                {
                    "question_id": qid,
                    "category": "numerical_lookup",
                    "ticker": ticker,
                    "fiscal_year": y1,
                    "difficulty": "hard",
                    "question": debt_to_equity_question(meta.company_name, y1),
                    "expected_answer": build_de_ratio_expected_answer(
                        meta.company_name, y1, debt.value, equity.value, ratio
                    ),
                    "evidence_passage": f"Computed from SEC total_debt={debt.value} and total_equity={equity.value}.",
                    "source_doc": f"{ticker}-{y1}-10K",
                    "source_section": "metrics",
                    "provenance": {
                        "source_primary": "sec_companyconcept",
                        "source_primary_value": ratio,
                        "source_primary_concept": "derived:debt_to_equity",
                        "source_primary_form": "10-K",
                        "source_primary_accn": debt.accn,
                        "source_primary_filed": debt.filed,
                        "source_secondary": None,
                        "source_secondary_value": None,
                        "agreement": None,
                        "diff_pct": None,
                        "generation_seed": seed,
                        "generation_template": "derived_debt_to_equity",
                    },
                },
            )
    return questions


def build_category_d(
    resolver: ValueResolver,
    seed: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    questions: List[Dict[str, object]] = []
    metric_candidates = sorted(CANONICAL_METRICS | {"operating_cash_flow"})
    candidates: List[Tuple[str, str, int, str]] = []
    for ticker_a, ticker_b in sorted(PAIR_CANDIDATES):
        for fy in YEARS:
            for metric in metric_candidates:
                candidates.append((ticker_a, ticker_b, fy, metric))
    rng.shuffle(candidates)

    for ticker_a, ticker_b, fy, metric in candidates:
        if len(questions) >= QUESTION_TARGETS["D"]:
            break
        meta_a = TICKER_REGISTRY[ticker_a]
        meta_b = TICKER_REGISTRY[ticker_b]
        if not metric_is_eligible(metric, meta_a.sector) or not metric_is_eligible(metric, meta_b.sector):
            continue
        value_a = resolver.annual(meta_a, metric, fy)
        value_b = resolver.annual(meta_b, metric, fy)
        if value_a is None or value_b is None:
            continue
        winner_ticker, winner_value, loser_ticker, loser_value = choose_comparison_winner(
            ticker_a, value_a.value, ticker_b, value_b.value
        )
        metric_label = get_metric_label(metric)
        qid = f"D{len(questions) + 1:03d}"
        _append_question(
            questions,
            {
                "question_id": qid,
                "category": "multi_company",
                "ticker": f"{ticker_a}/{ticker_b}",
                "tickers": [ticker_a, ticker_b],
                "fiscal_year": fy,
                "difficulty": "medium",
                "question": comparison_question(metric_label, fy, ticker_a, ticker_b),
                "expected_answer": build_comparison_expected_answer(
                    winner_ticker, metric_label, fy, winner_value, loser_ticker, loser_value
                ),
                "evidence_passage": (
                    f"SEC annual values: {ticker_a}={value_a.value}, {ticker_b}={value_b.value}, metric={metric_label}."
                ),
                "source_doc": f"{ticker_a}-{fy}-10K, {ticker_b}-{fy}-10K",
                "source_section": "metrics",
                "provenance": {
                    "source_primary": "sec_companyconcept",
                    "source_primary_value": winner_value,
                    "source_primary_concept": value_a.concept if winner_ticker == ticker_a else value_b.concept,
                    "source_primary_form": "10-K",
                    "source_primary_accn": value_a.accn if winner_ticker == ticker_a else value_b.accn,
                    "source_primary_filed": value_a.filed if winner_ticker == ticker_a else value_b.filed,
                    "source_secondary": "yfinance",
                    "source_secondary_value": value_a.secondary_value if winner_ticker == ticker_a else value_b.secondary_value,
                    "agreement": None,
                    "diff_pct": None,
                    "generation_seed": seed,
                    "generation_template": "cross_ticker_comparison",
                },
            },
        )
    return questions


def build_category_e(
    resolver: ValueResolver,
    sec_client: SecClient,
    seed: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    questions: List[Dict[str, object]] = []
    candidates: List[Tuple[str, int, str]] = []
    for ticker in ORDERED_TICKERS:
        for fy in YEARS:
            for subtype in ("employees", "filed_date", "shares_outstanding"):
                candidates.append((ticker, fy, subtype))
    candidates = sorted(candidates)
    rng.shuffle(candidates)

    for ticker, fy, subtype in candidates:
        if len(questions) >= QUESTION_TARGETS["E"]:
            break
        meta = TICKER_REGISTRY[ticker]
        qid = f"E{len(questions) + 1:03d}"

        if subtype == "employees":
            chosen = resolver.annual(meta, metric="employees", fy=fy)
            if chosen is None:
                continue
            _append_question(
                questions,
                {
                    "question_id": qid,
                    "category": "numerical_lookup",
                    "ticker": ticker,
                    "fiscal_year": fy,
                    "difficulty": "medium",
                    "question": employees_question(meta.company_name, fy),
                    "expected_answer": build_employees_expected_answer(meta.company_name, fy, chosen.value),
                    "evidence_passage": build_evidence_passage("employees", chosen.value, chosen.concept, chosen.filed),
                    "source_doc": f"{ticker}-{fy}-10K",
                    "source_section": "10-K Cover Page",
                    "provenance": _provenance_from_chosen(chosen, seed, "metadata_employees"),
                },
            )
        elif subtype == "shares_outstanding":
            chosen = resolver.annual(meta, metric="shares_outstanding", fy=fy)
            if chosen is None:
                continue
            _append_question(
                questions,
                {
                    "question_id": qid,
                    "category": "numerical_lookup",
                    "ticker": ticker,
                    "fiscal_year": fy,
                    "difficulty": "medium",
                    "question": shares_question(meta.company_name, fy),
                    "expected_answer": build_shares_expected_answer(meta.company_name, fy, chosen.value),
                    "evidence_passage": build_evidence_passage(
                        "shares_outstanding", chosen.value, chosen.concept, chosen.filed
                    ),
                    "source_doc": f"{ticker}-{fy}-10K",
                    "source_section": "10-K Cover Page",
                    "provenance": _provenance_from_chosen(chosen, seed, "metadata_shares"),
                },
            )
        else:
            filing = get_10k_filing_date(sec_client, cik=meta.cik, fy=fy)
            if filing is None:
                continue
            filing_date, accession = filing
            _append_question(
                questions,
                {
                    "question_id": qid,
                    "category": "citation_validation",
                    "ticker": ticker,
                    "fiscal_year": fy,
                    "difficulty": "easy",
                    "question": filing_date_question(meta.company_name, fy),
                    "expected_answer": build_filing_date_expected_answer(meta.company_name, fy, filing_date),
                    "evidence_passage": (
                        f"SEC submissions recent filings lists FY{fy} 10-K filing date={filing_date}, accession={accession}."
                    ),
                    "source_doc": f"{ticker}-{fy}-10K",
                    "source_section": "10-K Cover Page",
                    "provenance": {
                        "source_primary": "sec_submissions",
                        "source_primary_value": None,
                        "source_primary_concept": "filingDate",
                        "source_primary_form": "10-K",
                        "source_primary_accn": accession,
                        "source_primary_filed": filing_date,
                        "source_secondary": None,
                        "source_secondary_value": None,
                        "agreement": None,
                        "diff_pct": None,
                        "generation_seed": seed,
                        "generation_template": "metadata_filing_date",
                    },
                },
            )
    return questions


def build_category_f(seed: int) -> List[Dict[str, object]]:
    refusals = [
        ("AAPL", 2030, "What was AAPL's revenue in fiscal year 2030?"),
        ("IBM", 2023, "What was IBM's revenue in FY2023?"),
        ("MSFT", 2024, "What was MSFT's Q4 FY2024 revenue?"),
        ("MCD", 2023, "What was MCD's narrative discussion of FY2023 from its 10-K?"),
        ("FB/GOOG", 2023, "Compare FB's revenue to GOOG in FY2023"),
    ]
    questions: List[Dict[str, object]] = []
    for idx, (ticker, fy, text) in enumerate(refusals, start=1):
        questions.append(
            {
                "question_id": f"F{idx:03d}",
                "category": "refusal_test",
                "ticker": ticker,
                "fiscal_year": fy,
                "difficulty": "hard",
                "question": text,
                "expected_answer": (
                    "Refuse: the request is out of scope for the available FinSage corpus, "
                    "is ambiguous, or asks for unsupported periods/tickers."
                ),
                "evidence_passage": "Out-of-corpus or unsupported query by evaluation design.",
                "source_doc": "N/A",
                "source_section": "metrics",
                "provenance": _empty_provenance(seed, "refusal_handwritten"),
            }
        )
    return questions


def validate_distribution(questions: Sequence[Dict[str, object]]) -> None:
    if len(questions) != 100:
        raise ValueError(f"Expected 100 questions, got {len(questions)}")
    counts: Dict[str, int] = {k: 0 for k in QUESTION_TARGETS}
    for q in questions:
        prefix = str(q["question_id"])[0]
        if prefix in counts:
            counts[prefix] += 1
    for k, target in QUESTION_TARGETS.items():
        if counts[k] != target:
            raise ValueError(f"Category {k} expected {target}, got {counts[k]}")
    ids = [str(q["question_id"]) for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question_id found")
    for q in questions:
        if q.get("expected_answer") is None:
            raise ValueError(f"Missing expected_answer for {q['question_id']}")


def build_questions(seed: int) -> List[Dict[str, object]]:
    random.seed(seed)
    rng = random.Random(seed)
    sec_client = SecClient()
    yf_client = YFinanceClient()
    resolver = ValueResolver(sec_client, yf_client)

    category_a = build_category_a(resolver, seed, rng)
    category_b = build_category_b(resolver, seed, rng)
    category_c = build_category_c(resolver, seed, rng)
    category_d = build_category_d(resolver, seed, rng)
    category_e = build_category_e(resolver, sec_client, seed, rng)
    category_f = build_category_f(seed)

    built = {
        "A": category_a,
        "B": category_b,
        "C": category_c,
        "D": category_d,
        "E": category_e,
        "F": category_f,
    }
    for key, target in QUESTION_TARGETS.items():
        if len(built[key]) < target:
            raise RuntimeError(f"Insufficient questions for category {key}: {len(built[key])}/{target}")
        built[key] = built[key][:target]

    all_questions: List[Dict[str, object]] = []
    for key in ("A", "B", "C", "D", "E", "F"):
        all_questions.extend(built[key])

    all_questions = sorted(all_questions, key=lambda item: str(item["question_id"]))
    validate_distribution(all_questions)
    return all_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic FinSage ground truth eval questions.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic generation seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    questions = build_questions(seed=args.seed)
    output_path.write_text(
        json.dumps(questions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(questions)} questions to {output_path}")


if __name__ == "__main__":
    main()

