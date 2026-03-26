"""
Unit tests for the TARGET_CONCEPT_MAP normalization logic defined in
databricks/notebooks/03_silver_decoder.py.

These tests run entirely in plain Python — no Spark session required —
making them fast enough to execute on every pull request in CI.
"""

import pytest
from typing import Optional
# ---------------------------------------------------------------------------
# Inline the canonical map so the test has no Spark / Databricks dependency.
# This mirrors the dict in 03_silver_decoder.py exactly; if you update that
# dict you must update this one too (or factor it into a shared constants.py).
# ---------------------------------------------------------------------------
TARGET_CONCEPT_MAP = {
    "Revenues":                                                    "revenue",
    "SalesRevenueNet":                                             "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax":         "revenue",
    "RevenuesNetOfInterestExpense":                                "revenue",
    "TotalRevenuesAndOtherIncome":                                 "revenue",
    "NetIncomeLoss":                                               "net_income",
    "ProfitLoss":                                                  "net_income",
    "NetIncomeLossAvailableToCommonStockholdersBasic":             "net_income",
    "GrossProfit":                                                 "gross_profit",
    "OperatingIncomeLoss":                                         "operating_income",
    "NetCashProvidedByUsedInOperatingActivities":                  "operating_cash_flow",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations": "operating_cash_flow",
    "Assets":                                                      "total_assets",
    "Liabilities":                                                 "total_liabilities",
    "StockholdersEquity":                                          "equity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "equity",
    "LongTermDebtCurrent":                                         "short_term_debt",
    "ShortTermBorrowings":                                         "short_term_debt",
    "ShortTermDebt":                                               "short_term_debt",
    "CommercialPaper":                                             "short_term_debt",
    "LongTermDebt":                                                "long_term_debt",
    "LongTermDebtNoncurrent":                                      "long_term_debt",
    "LongTermDebtAndCapitalLeaseObligations":                      "long_term_debt",
    "LongTermBorrowings":                                          "long_term_debt",
    "DebtInstrumentCarryingAmount":                                "long_term_debt",
    "ResearchAndDevelopmentExpense":                               "rd_expense",
    "ResearchAndDevelopmentExcludingAcquiredInProcessCost":        "rd_expense",
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def normalize(concept: str) -> Optional[str]:
    """Return the normalized metric name for a raw XBRL concept, or None."""
    return TARGET_CONCEPT_MAP.get(concept)


# ---------------------------------------------------------------------------
# Revenue mapping tests
# ---------------------------------------------------------------------------
class TestRevenueConcepts:
    def test_sales_revenue_net_maps_to_revenue(self):
        assert normalize("SalesRevenueNet") == "revenue"

    def test_revenues_maps_to_revenue(self):
        assert normalize("Revenues") == "revenue"

    def test_asc_606_revenue_concept_maps_to_revenue(self):
        assert normalize("RevenueFromContractWithCustomerExcludingAssessedTax") == "revenue"

    def test_bank_revenue_concept_maps_to_revenue(self):
        assert normalize("RevenuesNetOfInterestExpense") == "revenue"

    def test_total_revenues_maps_to_revenue(self):
        assert normalize("TotalRevenuesAndOtherIncome") == "revenue"

    def test_all_revenue_concepts_normalize_consistently(self):
        revenue_concepts = [
            "Revenues",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenuesNetOfInterestExpense",
            "TotalRevenuesAndOtherIncome",
        ]
        for concept in revenue_concepts:
            assert normalize(concept) == "revenue", (
                f"Expected 'revenue' for concept '{concept}'"
            )


# ---------------------------------------------------------------------------
# Income / profitability mapping tests
# ---------------------------------------------------------------------------
class TestIncomeConcepts:
    def test_net_income_loss_maps_to_net_income(self):
        assert normalize("NetIncomeLoss") == "net_income"

    def test_profit_loss_maps_to_net_income(self):
        assert normalize("ProfitLoss") == "net_income"

    def test_gross_profit_maps_to_gross_profit(self):
        assert normalize("GrossProfit") == "gross_profit"

    def test_operating_income_loss_maps_to_operating_income(self):
        assert normalize("OperatingIncomeLoss") == "operating_income"


# ---------------------------------------------------------------------------
# Balance sheet mapping tests
# ---------------------------------------------------------------------------
class TestBalanceSheetConcepts:
    def test_assets_maps_to_total_assets(self):
        assert normalize("Assets") == "total_assets"

    def test_liabilities_maps_to_total_liabilities(self):
        assert normalize("Liabilities") == "total_liabilities"

    def test_stockholders_equity_maps_to_equity(self):
        assert normalize("StockholdersEquity") == "equity"

    def test_long_term_debt_maps_correctly(self):
        assert normalize("LongTermDebt") == "long_term_debt"

    def test_short_term_debt_variants_map_consistently(self):
        for concept in ("LongTermDebtCurrent", "ShortTermBorrowings", "ShortTermDebt", "CommercialPaper"):
            assert normalize(concept) == "short_term_debt", (
                f"Expected 'short_term_debt' for concept '{concept}'"
            )


# ---------------------------------------------------------------------------
# Cash flow mapping tests
# ---------------------------------------------------------------------------
class TestCashFlowConcepts:
    def test_operating_cash_flow_maps_correctly(self):
        assert normalize("NetCashProvidedByUsedInOperatingActivities") == "operating_cash_flow"

    def test_operating_cash_flow_continuing_ops_maps_correctly(self):
        assert normalize("NetCashProvidedByUsedInOperatingActivitiesContinuingOperations") == "operating_cash_flow"


# ---------------------------------------------------------------------------
# R&D mapping tests
# ---------------------------------------------------------------------------
class TestRDConcepts:
    def test_rd_expense_maps_correctly(self):
        assert normalize("ResearchAndDevelopmentExpense") == "rd_expense"

    def test_rd_expense_ex_acquired_maps_correctly(self):
        assert normalize("ResearchAndDevelopmentExcludingAcquiredInProcessCost") == "rd_expense"


# ---------------------------------------------------------------------------
# Negative / boundary tests
# ---------------------------------------------------------------------------
class TestUnmappedConcepts:
    def test_unknown_concept_returns_none(self):
        assert normalize("SomeRandomXBRLConcept") is None

    def test_empty_string_returns_none(self):
        assert normalize("") is None

    def test_case_sensitive_mismatch_returns_none(self):
        # The map is case-sensitive; lowercase should not match.
        assert normalize("salesrevenuenet") is None

    def test_partial_match_returns_none(self):
        assert normalize("SalesRevenue") is None


# ---------------------------------------------------------------------------
# Map completeness / structural tests
# ---------------------------------------------------------------------------
class TestMapCompleteness:
    EXPECTED_NORMALIZED_VALUES = {
        "revenue", "net_income", "gross_profit", "operating_income",
        "operating_cash_flow", "total_assets", "total_liabilities",
        "equity", "short_term_debt", "long_term_debt", "rd_expense",
    }

    def test_all_values_are_known_normalized_metrics(self):
        for concept, normalized in TARGET_CONCEPT_MAP.items():
            assert normalized in self.EXPECTED_NORMALIZED_VALUES, (
                f"Concept '{concept}' maps to unknown value '{normalized}'"
            )

    def test_map_is_not_empty(self):
        assert len(TARGET_CONCEPT_MAP) > 0

    def test_no_none_values_in_map(self):
        for concept, normalized in TARGET_CONCEPT_MAP.items():
            assert normalized is not None, f"Concept '{concept}' has a None value"
