"""Question templates and expected-answer formatting helpers."""

from __future__ import annotations

from typing import Tuple


def annual_question(company_name: str, metric_label: str, fy: int) -> str:
    return f"What was {company_name}'s {metric_label} in fiscal year {fy}?"


def quarterly_question(company_name: str, metric_label: str, fy: int, quarter: int) -> str:
    return f"What was {company_name}'s {metric_label} in Q{quarter} of fiscal year {fy}?"


def growth_question(company_name: str, y1: int, y2: int) -> str:
    return f"What was {company_name}'s revenue growth from FY{y1} to FY{y2}?"


def gross_margin_question(company_name: str, fy: int) -> str:
    return f"What was {company_name}'s gross margin in fiscal year {fy}?"


def debt_to_equity_question(company_name: str, fy: int) -> str:
    return f"What was {company_name}'s debt-to-equity ratio at the end of fiscal year {fy}?"


def comparison_question(metric_label: str, fy: int, ticker_a: str, ticker_b: str) -> str:
    return (
        f"Which had higher {metric_label} in fiscal year {fy}: "
        f"{ticker_a} or {ticker_b}?"
    )


def employees_question(company_name: str, fy: int) -> str:
    return f"How many employees did {company_name} report in their FY{fy} 10-K?"


def filing_date_question(company_name: str, fy: int) -> str:
    return f"On what date did {company_name} file its FY{fy} 10-K with the SEC?"


def shares_question(company_name: str, fy: int) -> str:
    return (
        f"How many shares of common stock did {company_name} have outstanding "
        f"at the end of FY{fy}?"
    )


def format_money(value: float) -> str:
    return f"${value:,.2f}"


def format_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def build_annual_expected_answer(
    company_name: str,
    metric_label: str,
    value: float,
    concept: str,
    filed: str,
) -> str:
    return (
        f"{format_money(value)}. {company_name} reported {metric_label} of "
        f"{format_money(value)} (SEC concept {concept}, filed {filed})."
    )


def build_quarterly_expected_answer(
    company_name: str,
    metric_label: str,
    value: float,
    quarter: int,
    fy: int,
    concept: str,
    filed: str,
) -> str:
    return (
        f"{format_money(value)}. In FY{fy} Q{quarter}, {company_name} reported "
        f"{metric_label} of {format_money(value)} (SEC concept {concept}, filed {filed})."
    )


def build_growth_expected_answer(
    company_name: str,
    y1: int,
    y2: int,
    rev_y1: float,
    rev_y2: float,
    growth_pct: float,
) -> str:
    return (
        f"{format_percent(growth_pct)}. {company_name}'s revenue increased from "
        f"{format_money(rev_y1)} in FY{y1} to {format_money(rev_y2)} in FY{y2}."
    )


def build_margin_expected_answer(
    company_name: str,
    fy: int,
    gross_profit: float,
    revenue: float,
    margin_pct: float,
) -> str:
    return (
        f"{format_percent(margin_pct)}. {company_name}'s FY{fy} gross margin was "
        f"{format_money(gross_profit)} / {format_money(revenue)}."
    )


def build_de_ratio_expected_answer(
    company_name: str,
    fy: int,
    debt: float,
    equity: float,
    ratio: float,
) -> str:
    return (
        f"{ratio:.4f}. {company_name}'s FY{fy} debt-to-equity ratio was "
        f"{format_money(debt)} / {format_money(equity)}."
    )


def build_comparison_expected_answer(
    winner_ticker: str,
    metric_label: str,
    fy: int,
    winner_value: float,
    loser_ticker: str,
    loser_value: float,
) -> str:
    return (
        f"{format_money(winner_value)}. {winner_ticker} had higher {metric_label} in FY{fy} "
        f"than {loser_ticker} ({format_money(loser_value)})."
    )


def build_employees_expected_answer(company_name: str, fy: int, employees: float) -> str:
    return (
        f"{format_number(employees)}. {company_name} reported {format_number(employees)} "
        f"employees in its FY{fy} 10-K."
    )


def build_shares_expected_answer(company_name: str, fy: int, shares: float) -> str:
    return (
        f"{format_number(shares)}. {company_name} reported {format_number(shares)} shares "
        f"outstanding at FY{fy} year-end."
    )


def build_filing_date_expected_answer(company_name: str, fy: int, filing_date: str) -> str:
    return (
        f"{filing_date}. {company_name} filed its FY{fy} 10-K on {filing_date}."
    )


def build_evidence_passage(metric_label: str, value: float, concept: str, filed: str) -> str:
    return (
        f"SEC CompanyConcept fact: concept={concept}, value={value}, filed={filed}, "
        f"metric={metric_label}."
    )


def build_cross_validation_suffix(agreement: bool | None, diff_pct: float | None) -> str:
    if agreement is None:
        return "Secondary source unavailable."
    if agreement:
        return f"Cross-validated with yfinance (diff={diff_pct:.4f}%)."
    return f"SEC chosen over yfinance disagreement (diff={diff_pct:.4f}%)."


def choose_comparison_winner(
    ticker_a: str, value_a: float, ticker_b: str, value_b: float
) -> Tuple[str, float, str, float]:
    if value_a >= value_b:
        return ticker_a, value_a, ticker_b, value_b
    return ticker_b, value_b, ticker_a, value_a

