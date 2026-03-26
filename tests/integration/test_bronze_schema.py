"""
Integration tests for the FinSage Medallion pipeline.

These tests require a live Spark session connected to a Databricks workspace
where the pipeline has been run at least once.

Run integration tests:
    pytest tests/integration/ -m integration -v

Skip integration tests in CI (default):
    pytest -m "not integration"

The `spark` fixture is provided by tests/conftest.py.
"""

import pytest
from pyspark.sql.functions import col, min as spark_min, max as spark_max

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Expected table catalogue — all layers
# ---------------------------------------------------------------------------
BRONZE_TABLES = [
    "main.finsage_bronze.filings",
    "main.finsage_bronze.xbrl_companyfacts_raw",
    "main.finsage_bronze.ingestion_errors",
    "main.finsage_bronze.sec_filings_download_log",
]

SILVER_TABLES = [
    "main.finsage_silver.financial_statements",
    "main.finsage_silver.filing_sections",
]

GOLD_TABLES = [
    "main.finsage_gold.company_metrics",
    "main.finsage_gold.filing_section_chunks",
]

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


# ---------------------------------------------------------------------------
# Existence tests
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------
class TestBronzeSchema:
    def test_filings_has_required_columns(self, spark):
        actual = set(spark.table("main.finsage_bronze.filings").columns)
        missing = BRONZE_FILINGS_REQUIRED_COLS - actual
        assert not missing, f"Missing columns in bronze.filings: {missing}"

    def test_xbrl_raw_has_ticker_and_json(self, spark):
        df = spark.table("main.finsage_bronze.xbrl_companyfacts_raw")
        for col_name in ("ticker", "raw_json", "api_status", "fetched_at"):
            assert col_name in df.columns, f"Missing column: {col_name}"


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


class TestGoldSchema:
    def test_company_metrics_has_required_columns(self, spark):
        actual = set(spark.table("main.finsage_gold.company_metrics").columns)
        missing = GOLD_METRICS_REQUIRED_COLS - actual
        assert not missing, f"Missing columns in gold.company_metrics: {missing}"


# ---------------------------------------------------------------------------
# Data quality tests
# ---------------------------------------------------------------------------
class TestDataQuality:
    def test_bronze_filings_not_empty(self, spark):
        count = spark.table("main.finsage_bronze.filings").count()
        assert count > 0, "bronze.filings is empty — ingestion may not have run"

    def test_silver_has_revenue_facts(self, spark):
        df = spark.table("main.finsage_silver.financial_statements")
        revenue_count = df.filter(col("normalized_line_item") == "revenue").count()
        assert revenue_count > 0, "No revenue facts in silver.financial_statements"

    def test_gold_data_quality_score_in_range(self, spark):
        df = spark.table("main.finsage_gold.company_metrics")
        stats = df.agg(
            spark_min("data_quality_score").alias("min_score"),
            spark_max("data_quality_score").alias("max_score"),
        ).collect()[0]
        assert 0.0 <= stats["min_score"], "data_quality_score below 0.0"
        assert stats["max_score"] <= 1.0, "data_quality_score above 1.0"

    def test_gold_no_negative_revenue(self, spark):
        df = spark.table("main.finsage_gold.company_metrics")
        negative_revenue = df.filter(
            col("revenue").isNotNull() & (col("revenue") < 0)
        ).count()
        assert negative_revenue == 0, (
            f"{negative_revenue} rows have negative revenue in gold.company_metrics"
        )

    def test_silver_no_duplicate_statement_ids(self, spark):
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
