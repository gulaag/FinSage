"""Sector-aware metric concept mappings for SEC CompanyConcept lookup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class MetricSpec:
    """Metric definition across SEC and yfinance sources."""

    metric: str
    label: str
    taxonomy: str
    concept_candidates_default: List[str]
    concept_candidates_bank: List[str] | None = None
    is_flow: bool = True


METRIC_SPECS: Dict[str, MetricSpec] = {
    "revenue": MetricSpec(
        metric="revenue",
        label="revenue",
        taxonomy="us-gaap",
        concept_candidates_default=[
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
        ],
        concept_candidates_bank=["InterestAndDividendIncomeOperating", "Revenues"],
    ),
    "net_income": MetricSpec(
        metric="net_income",
        label="net income",
        taxonomy="us-gaap",
        concept_candidates_default=["NetIncomeLoss"],
    ),
    "operating_income": MetricSpec(
        metric="operating_income",
        label="operating income",
        taxonomy="us-gaap",
        concept_candidates_default=["OperatingIncomeLoss"],
    ),
    "gross_profit": MetricSpec(
        metric="gross_profit",
        label="gross profit",
        taxonomy="us-gaap",
        concept_candidates_default=["GrossProfit"],
    ),
    "total_assets": MetricSpec(
        metric="total_assets",
        label="total assets",
        taxonomy="us-gaap",
        concept_candidates_default=["Assets"],
        is_flow=False,
    ),
    "total_liabilities": MetricSpec(
        metric="total_liabilities",
        label="total liabilities",
        taxonomy="us-gaap",
        concept_candidates_default=["Liabilities"],
        is_flow=False,
    ),
    "total_equity": MetricSpec(
        metric="total_equity",
        label="total equity",
        taxonomy="us-gaap",
        concept_candidates_default=[
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        is_flow=False,
    ),
    "operating_cash_flow": MetricSpec(
        metric="operating_cash_flow",
        label="operating cash flow",
        taxonomy="us-gaap",
        concept_candidates_default=["NetCashProvidedByUsedInOperatingActivities"],
    ),
    "rd_expense": MetricSpec(
        metric="rd_expense",
        label="research and development expense",
        taxonomy="us-gaap",
        concept_candidates_default=["ResearchAndDevelopmentExpense"],
    ),
    "total_debt": MetricSpec(
        metric="total_debt",
        label="total debt",
        taxonomy="us-gaap",
        concept_candidates_default=["LongTermDebt", "LongTermDebtNoncurrent", "DebtCurrent"],
        is_flow=False,
    ),
    "employees": MetricSpec(
        metric="employees",
        label="employees",
        taxonomy="dei",
        concept_candidates_default=["EntityNumberOfEmployees"],
        is_flow=False,
    ),
    "shares_outstanding": MetricSpec(
        metric="shares_outstanding",
        label="shares outstanding",
        taxonomy="dei",
        concept_candidates_default=["EntityCommonStockSharesOutstanding"],
        is_flow=False,
    ),
}

ANNUAL_METRICS: List[str] = [
    "revenue",
    "net_income",
    "operating_income",
    "gross_profit",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "operating_cash_flow",
    "rd_expense",
]

QUARTERLY_METRICS: List[str] = [
    "revenue",
    "net_income",
    "operating_income",
    "gross_profit",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "operating_cash_flow",
]

GROSS_PROFIT_ALLOWED_SECTORS = {"tech_megacap", "saas", "pharma", "retail", "payments"}
RD_ALLOWED_SECTORS = {"tech_megacap", "saas", "pharma"}
BANK_REVENUE_TICKERS = {"BAC", "JPM", "GS"}

CANONICAL_METRICS = {"revenue", "net_income", "total_assets", "total_liabilities", "total_equity"}


def get_metric_label(metric: str) -> str:
    """Return human-readable metric label."""
    return METRIC_SPECS[metric].label


def metric_is_eligible(metric: str, sector: str) -> bool:
    """Apply fixed metric-sector compatibility rules from the brief."""
    if metric == "gross_profit":
        return sector in GROSS_PROFIT_ALLOWED_SECTORS
    if metric == "rd_expense":
        return sector in RD_ALLOWED_SECTORS
    return True


def concept_candidates_for_metric(metric: str, ticker: str) -> List[str]:
    """Return ordered SEC concept candidates with bank override for revenue."""
    spec = METRIC_SPECS[metric]
    if metric == "revenue" and ticker in BANK_REVENUE_TICKERS and spec.concept_candidates_bank:
        return list(spec.concept_candidates_bank)
    return list(spec.concept_candidates_default)

