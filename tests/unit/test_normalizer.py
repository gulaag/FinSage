"""
FinSage — Unit Tests for XBRL Concept Normalisation
=====================================================
These tests verify the TARGET_CONCEPT_MAP — the dictionary that translates
raw XBRL concept names (e.g. "RevenueFromContractWithCustomerExcludingAssessedTax")
into canonical metric names (e.g. "revenue") used throughout the Silver and Gold layers.

Why this map needs tests:
    The SEC uses hundreds of different XBRL concept names for the same economic
    metric across different companies and filing periods. The map is the single
    source of truth that normalises this inconsistency. If a mapping is wrong
    or missing, the Gold layer silently produces incorrect or incomplete metrics.
    Tests catch regressions when the map is modified.

Source of truth:
    The map lives in src/finsage/constants.py — imported here directly.
    No duplication. No drift risk between the map and its tests.

Speed:
    Pure Python — no Spark session, no network calls, no fixtures.
    The entire suite runs in < 1 second, making it the first gate in CI.

How to run:
    pytest tests/unit/ -v
    pytest tests/unit/test_normalizer.py -v --tb=short   # this file only
"""

import pytest
from typing import Optional
from finsage.constants import TARGET_CONCEPT_MAP, VALID_NORMALIZED_METRICS


# ---------------------------------------------------------------------------
# Helper — wraps the dict lookup so tests read like plain English assertions.
# ---------------------------------------------------------------------------
def normalize(concept: str) -> Optional[str]:
    """Return the normalised metric name for a raw XBRL concept, or None."""
    return TARGET_CONCEPT_MAP.get(concept)


# ==============================================================================
# TestRevenueConcepts
# Revenue is reported under different XBRL tags depending on the company's
# industry and the accounting standard version they file under:
#   - "Revenues"                                       (general)
#   - "SalesRevenueNet"                                (older filers)
#   - "RevenueFromContractWithCustomerExcludingAssessedTax"  (ASC 606, post-2018)
#   - "RevenuesNetOfInterestExpense"                   (banks / financial services)
#   - "TotalRevenuesAndOtherIncome"                    (diversified companies)
# All must map to the single canonical value "revenue".
# ==============================================================================
class TestRevenueConcepts:
    def test_sales_revenue_net_maps_to_revenue(self):
        assert normalize("SalesRevenueNet") == "revenue"

    def test_revenues_maps_to_revenue(self):
        assert normalize("Revenues") == "revenue"

    def test_asc_606_revenue_concept_maps_to_revenue(self):
        # ASC 606 is the post-2018 revenue recognition standard.
        # Most large-cap companies switched to this concept after 2018.
        assert normalize("RevenueFromContractWithCustomerExcludingAssessedTax") == "revenue"

    def test_bank_revenue_concept_maps_to_revenue(self):
        # Banks (JPM, GS, BAC) report net interest income under this concept.
        assert normalize("RevenuesNetOfInterestExpense") == "revenue"

    def test_total_revenues_maps_to_revenue(self):
        assert normalize("TotalRevenuesAndOtherIncome") == "revenue"

    def test_all_revenue_concepts_normalize_consistently(self):
        # Bulk assertion: every known revenue concept must resolve to "revenue".
        # If a new concept is added to the map with a typo, this catches it.
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


# ==============================================================================
# TestIncomeConcepts
# Net income and profitability metrics — also reported under multiple XBRL tags
# depending on whether the company has minority interest, discontinued operations, etc.
# ==============================================================================
class TestIncomeConcepts:
    def test_net_income_loss_maps_to_net_income(self):
        assert normalize("NetIncomeLoss") == "net_income"

    def test_profit_loss_maps_to_net_income(self):
        # Used by companies with significant minority/non-controlling interests.
        assert normalize("ProfitLoss") == "net_income"

    def test_gross_profit_maps_to_gross_profit(self):
        assert normalize("GrossProfit") == "gross_profit"

    def test_operating_income_loss_maps_to_operating_income(self):
        assert normalize("OperatingIncomeLoss") == "operating_income"


# ==============================================================================
# TestBalanceSheetConcepts
# Balance sheet items — assets, liabilities, equity, and debt classifications.
# Short-term debt has the most variants because companies structure it differently
# (commercial paper, revolving credit, current portion of long-term debt, etc.).
# ==============================================================================
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
        # All four concepts represent the same economic item: near-term debt obligations.
        # A single canonical "short_term_debt" column in Gold covers all variants.
        for concept in ("LongTermDebtCurrent", "ShortTermBorrowings", "ShortTermDebt", "CommercialPaper"):
            assert normalize(concept) == "short_term_debt", (
                f"Expected 'short_term_debt' for concept '{concept}'"
            )


# ==============================================================================
# TestCashFlowConcepts
# Operating cash flow — the primary and its "continuing operations" variant,
# which excludes cash flows from discontinued business segments.
# ==============================================================================
class TestCashFlowConcepts:
    def test_operating_cash_flow_maps_correctly(self):
        assert normalize("NetCashProvidedByUsedInOperatingActivities") == "operating_cash_flow"

    def test_operating_cash_flow_continuing_ops_maps_correctly(self):
        # "ContinuingOperations" variant used when a company has divested a segment.
        assert normalize("NetCashProvidedByUsedInOperatingActivitiesContinuingOperations") == "operating_cash_flow"


# ==============================================================================
# TestRDConcepts
# R&D expense — important for tech and pharma companies (MSFT, NVDA, PFE, etc.).
# The "ExcludingAcquiredInProcessCost" variant excludes one-time acquisition R&D.
# ==============================================================================
class TestRDConcepts:
    def test_rd_expense_maps_correctly(self):
        assert normalize("ResearchAndDevelopmentExpense") == "rd_expense"

    def test_rd_expense_ex_acquired_maps_correctly(self):
        assert normalize("ResearchAndDevelopmentExcludingAcquiredInProcessCost") == "rd_expense"


# ==============================================================================
# TestUnmappedConcepts
# Negative / boundary tests — the map should return None for anything not
# explicitly mapped. This ensures the Silver layer's `if not normalized_line_item:
# continue` guard works correctly and unknown concepts are silently skipped
# rather than passed through as garbage data.
# ==============================================================================
class TestUnmappedConcepts:
    def test_unknown_concept_returns_none(self):
        assert normalize("SomeRandomXBRLConcept") is None

    def test_empty_string_returns_none(self):
        assert normalize("") is None

    def test_case_sensitive_mismatch_returns_none(self):
        # The map is intentionally case-sensitive — XBRL concept names are PascalCase.
        # A lowercase variant should never match; this guards against accidental
        # case-folding in the ingestion logic.
        assert normalize("salesrevenuenet") is None

    def test_partial_match_returns_none(self):
        # Dict lookup is exact — partial concept names must not match.
        assert normalize("SalesRevenue") is None


# ==============================================================================
# TestMapCompleteness
# Structural integrity tests on the map itself — not on individual concepts,
# but on the map as a whole. These catch map corruption: None values, unmapped
# normalized metrics, or categories being accidentally removed.
# ==============================================================================
class TestMapCompleteness:
    def test_all_values_are_known_normalized_metrics(self):
        # Every value in TARGET_CONCEPT_MAP must be a member of VALID_NORMALIZED_METRICS.
        # This ensures no typos (e.g. "revenu" instead of "revenue") slip into the map.
        for concept, normalised in TARGET_CONCEPT_MAP.items():
            assert normalised in VALID_NORMALIZED_METRICS, (
                f"Concept '{concept}' maps to unknown value '{normalised}'"
            )

    def test_map_is_not_empty(self):
        # Guard against an accidental wipe of the constants file.
        assert len(TARGET_CONCEPT_MAP) > 0

    def test_no_none_values_in_map(self):
        # Every key must map to a non-None string. A None value would silently
        # pass the `if not normalized_line_item` guard and corrupt the Silver table.
        for concept, normalised in TARGET_CONCEPT_MAP.items():
            assert normalised is not None, f"Concept '{concept}' has a None value"

    def test_valid_normalized_metrics_covers_expected_categories(self):
        # VALID_NORMALIZED_METRICS is the closed set of canonical metric names.
        # This test ensures all 11 expected financial categories are present,
        # so that removing a category from the set breaks the build immediately.
        expected_categories = {
            "revenue", "net_income", "gross_profit", "operating_income",
            "operating_cash_flow", "total_assets", "total_liabilities",
            "equity", "short_term_debt", "long_term_debt", "rd_expense",
        }
        assert expected_categories.issubset(VALID_NORMALIZED_METRICS), (
            f"Missing categories: {expected_categories - VALID_NORMALIZED_METRICS}"
        )
