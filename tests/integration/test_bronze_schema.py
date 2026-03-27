"""
FinSage — Integration Tests for the Medallion Pipeline
=======================================================
Integration tests verify the pipeline's output against a LIVE Databricks
workspace where the notebooks have been run at least once. They connect to
real Delta tables via a live Spark session — no mocking.

When to run:
    After a full pipeline run (all 5 notebooks executed successfully).
    Never in CI on a PR — they are skipped by default via the `integration` mark.

How to run locally:
    pytest tests/integration/ -m integration -v

How to skip in CI (default pytest.ini behaviour):
    pytest -m "not integration"

The `spark` fixture is provided by tests/conftest.py, which initialises a
SparkSession connected to the Databricks workspace via the remote execution API.

Test structure (four layers of assertion):
    TestTableExistence   — every expected table exists in Unity Catalog
    TestBronzeSchema     — bronze tables have all required columns
    TestSilverSchema     — silver tables have correct columns + data rules
    TestGoldSchema       — gold table has all analytics-facing columns
    TestDataQuality      — actual row-level data assertions across all layers
"""

import pytest
from pyspark.sql.functions import col, min as spark_min, max as spark_max

# Mark every test in this file as `integration` so pytest can filter them.
# CI runs `pytest -m "not integration"` to skip these entirely on PRs.
pytestmark = pytest.mark.integration

# ==============================================================================
# Expected Table Catalogue
# These sets define the contract: if a table or column is missing after a
# pipeline run, a test fails and the deploy is blocked.
# ==============================================================================

# All tables that must exist after a successful Bronze ingestion run.
BRONZE_TABLES = [
    "main.finsage_bronze.filings",
    "main.finsage_bronze.xbrl_companyfacts_raw",
    "main.finsage_bronze.ingestion_errors",
    "main.finsage_bronze.sec_filings_download_log",
]

# All tables that must exist after a successful Silver decoding run.
SILVER_TABLES = [
    "main.finsage_silver.financial_statements",
    "main.finsage_silver.filing_sections",
]

# All tables that must exist after a successful Gold metrics run.
GOLD_TABLES = [
    "main.finsage_gold.company_metrics",
    "main.finsage_gold.filing_section_chunks",
]

# Required column sets — used in schema tests below.
# Adding a column to a notebook is safe. REMOVING one must also remove it here
# or the test will catch the regression immediately.
BRONZE_FILINGS_REQUIRED_COLS = {
    "filing_id", "ticker", "filing_type", "accession_number",
    "fiscal_year", "file_path", "content", "ingestion_status", "ingested_at",
}

SILVER_STATEMENTS_REQUIRED_COLS = {
    "statement_id", "ticker", "fiscal_year", "filing_type",
    "raw_line_item", "normalized_line_item", "value", "unit",
    "xbrl_validated", "parsed_at",
}

GOLD_METRICS_REQUIRED_COLS = {
    "ticker", "fiscal_year", "revenue", "net_income", "total_assets",
    "data_quality_score", "updated_at",
}


# ==============================================================================
# TestTableExistence
# Simplest possible assertion: does the table exist at all?
# Parametrized so each table generates its own test case in the pytest report,
# making it obvious exactly which table is missing when a run fails.
# ==============================================================================
class TestTableExistence:
    @pytest.mark.parametrize("table", BRONZE_TABLES)
    def test_bronze_table_exists(self, spark, table):
        assert spark.catalog.tableExists(table), f"Bronze table missing: {table}"

    @pytest.mark.parametrize("table", SILVER_TABLES)
    def test_silver_table_exists(self, spark, table):
        assert spark.catalog.tableExists(table), f"Silver table missing: {table}"

    @pytest.mark.parametrize("table", GOLD_TABLES)
    def test_gold_table_exists(self, spark, table):
        assert spark.catalog.tableExists(table), f"Gold table missing: {table}"


# ==============================================================================
# TestBronzeSchema
# Verifies that the Bronze layer tables have the columns the Silver notebook
# depends on. A missing column here would cause a hard failure in 03_silver_decoder.
# ==============================================================================
class TestBronzeSchema:
    def test_filings_has_required_columns(self, spark):
        actual = set(spark.table("main.finsage_bronze.filings").columns)
        missing = BRONZE_FILINGS_REQUIRED_COLS - actual
        assert not missing, f"Missing columns in bronze.filings: {missing}"

    def test_xbrl_raw_has_ticker_and_json(self, spark):
        # Spot-check the four columns that the Silver XBRL flattening reads directly.
        df = spark.table("main.finsage_bronze.xbrl_companyfacts_raw")
        for col_name in ("ticker", "raw_json", "api_status", "fetched_at"):
            assert col_name in df.columns, f"Missing column: {col_name}"


# ==============================================================================
# TestSilverSchema
# Verifies column contracts and business rules in the Silver layer.
# filing_sections must only contain 10-K rows — the notebook explicitly filters
# for this; if that filter were removed, quarterly data would pollute the LLM chunks.
# ==============================================================================
class TestSilverSchema:
    def test_financial_statements_has_required_columns(self, spark):
        actual = set(spark.table("main.finsage_silver.financial_statements").columns)
        missing = SILVER_STATEMENTS_REQUIRED_COLS - actual
        assert not missing, f"Missing columns in silver.financial_statements: {missing}"

    def test_filing_sections_has_text_column(self, spark):
        df = spark.table("main.finsage_silver.filing_sections")
        assert "section_text" in df.columns

    def test_filing_sections_only_annual(self, spark):
        """Section extraction is 10-K only — verify no quarterly sections slipped through."""
        df = spark.table("main.finsage_silver.filing_sections")
        if "filing_type" in df.columns:
            quarterly_count = df.filter(col("filing_type") == "10-Q").count()
            assert quarterly_count == 0, (
                f"Found {quarterly_count} quarterly filings in filing_sections — only 10-K expected"
            )


# ==============================================================================
# TestGoldSchema
# Verifies the Gold layer contract. These columns are what downstream consumers
# (dashboards, the RAG system, analysts) depend on. A missing column here means
# a broken consumer in production.
# ==============================================================================
class TestGoldSchema:
    def test_company_metrics_has_required_columns(self, spark):
        actual = set(spark.table("main.finsage_gold.company_metrics").columns)
        missing = GOLD_METRICS_REQUIRED_COLS - actual
        assert not missing, f"Missing columns in gold.company_metrics: {missing}"


# ==============================================================================
# TestDataQuality
# Row-level data assertions across all three layers. These catch silent data
# correctness failures that schema tests cannot — empty tables, out-of-range
# scores, negative revenue, and duplicate primary keys.
# ==============================================================================
class TestDataQuality:
    def test_bronze_filings_not_empty(self, spark):
        # If this fails, Auto Loader did not ingest any files — check the Volume path.
        count = spark.table("main.finsage_bronze.filings").count()
        assert count > 0, "bronze.filings is empty — ingestion may not have run"

    def test_silver_has_revenue_facts(self, spark):
        # Revenue is the most fundamental financial metric. If it is missing,
        # the XBRL concept map or the Silver flattening logic has a bug.
        df = spark.table("main.finsage_silver.financial_statements")
        revenue_count = df.filter(col("normalized_line_item") == "revenue").count()
        assert revenue_count > 0, "No revenue facts in silver.financial_statements"

    def test_gold_data_quality_score_in_range(self, spark):
        # data_quality_score is computed as a fraction [0.0, 1.0] in 04_gold_metrics.
        # Values outside this range indicate a bug in the scoring formula.
        df = spark.table("main.finsage_gold.company_metrics")
        stats = df.agg(
            spark_min("data_quality_score").alias("min_score"),
            spark_max("data_quality_score").alias("max_score"),
        ).collect()[0]
        assert 0.0 <= stats["min_score"], "data_quality_score below 0.0"
        assert stats["max_score"] <= 1.0, "data_quality_score above 1.0"

    def test_gold_no_negative_revenue(self, spark):
        # Revenue should never be negative for the 30 blue-chip tickers in scope.
        # A negative value indicates a wrong XBRL concept was mapped to 'revenue'.
        df = spark.table("main.finsage_gold.company_metrics")
        negative_revenue = df.filter(
            col("revenue").isNotNull() & (col("revenue") < 0)
        ).count()
        assert negative_revenue == 0, (
            f"{negative_revenue} rows have negative revenue in gold.company_metrics"
        )

    def test_silver_no_duplicate_statement_ids(self, spark):
        # statement_id is a SHA-256 hash of (ticker, accession, concept, unit, period_end).
        # Duplicates mean the MERGE in 03_silver_decoder is not deduplicating correctly.
        from pyspark.sql.functions import count as spark_count
        df = spark.table("main.finsage_silver.financial_statements")
        dup_count = (
            df.groupBy("statement_id")
            .agg(spark_count("*").alias("cnt"))
            .filter(col("cnt") > 1)
            .count()
        )
        assert dup_count == 0, (
            f"{dup_count} duplicate statement_id values in silver.financial_statements"
        )
