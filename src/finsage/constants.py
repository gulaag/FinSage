"""
FinSage shared constants.

This is the single source of truth for the canonical XBRL concept → normalised
metric name mapping.  It is imported by:

  - databricks/notebooks/03_silver_decoder.py  (Databricks cluster, via installed wheel
                                                or sys.path injection from the DAB deploy)
  - tests/unit/test_normalizer.py              (CI runner, plain Python — no Spark needed)

When the finsage package is distributed as a wheel and installed on the cluster,
the notebook imports it with:
    from finsage.constants import TARGET_CONCEPT_MAP, STATEMENT_TYPE_MAP
"""

# ---------------------------------------------------------------------------
# Canonical XBRL concept → normalised metric name.
# Keys are raw us-gaap concept names exactly as they appear in SEC CompanyFacts JSON.
# Values are the snake_case metric names used throughout Silver and Gold layers.
# ---------------------------------------------------------------------------
TARGET_CONCEPT_MAP: dict[str, str] = {
    # Revenue
    "Revenues":                                                        "revenue",
    "SalesRevenueNet":                                                 "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax":             "revenue",
    "RevenuesNetOfInterestExpense":                                    "revenue",
    "TotalRevenuesAndOtherIncome":                                     "revenue",
    # Net income
    "NetIncomeLoss":                                                   "net_income",
    "ProfitLoss":                                                      "net_income",
    "NetIncomeLossAvailableToCommonStockholdersBasic":                 "net_income",
    # Profitability
    "GrossProfit":                                                     "gross_profit",
    "OperatingIncomeLoss":                                             "operating_income",
    # Cash flow
    "NetCashProvidedByUsedInOperatingActivities":                      "operating_cash_flow",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations":  "operating_cash_flow",
    # Balance sheet — assets & liabilities
    "Assets":                                                          "total_assets",
    "Liabilities":                                                     "total_liabilities",
    # Equity
    "StockholdersEquity":                                              "equity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "equity",
    # Debt
    "LongTermDebtCurrent":                                             "short_term_debt",
    "ShortTermBorrowings":                                             "short_term_debt",
    "ShortTermDebt":                                                   "short_term_debt",
    "CommercialPaper":                                                 "short_term_debt",
    "LongTermDebt":                                                    "long_term_debt",
    "LongTermDebtNoncurrent":                                          "long_term_debt",
    "LongTermDebtAndCapitalLeaseObligations":                          "long_term_debt",
    "LongTermBorrowings":                                              "long_term_debt",
    "DebtInstrumentCarryingAmount":                                    "long_term_debt",
    # R&D
    "ResearchAndDevelopmentExpense":                                   "rd_expense",
    "ResearchAndDevelopmentExcludingAcquiredInProcessCost":            "rd_expense",
}

# Derived: every normalised metric is a financial_metric for statement_type tagging.
STATEMENT_TYPE_MAP: dict[str, str] = {
    v: "financial_metric" for v in TARGET_CONCEPT_MAP.values()
}

# Complete set of valid normalised metric names — used in structural tests.
VALID_NORMALIZED_METRICS: frozenset[str] = frozenset(TARGET_CONCEPT_MAP.values())
