# Databricks notebook source
# ==============================================================================
# FinSage | 04 — Gold Metrics
# Aggregates Silver financial_statements into a wide, analysis-ready
# company_metrics table.  Applies strict fiscal-period alignment, canonical
# accession selection, and YoY growth derivation.
# ==============================================================================

# COMMAND ----------

# ── Runtime Parameters ────────────────────────────────────────────────────────
dbutils.widgets.text("catalog",       "main",       "Unity Catalog catalog")
dbutils.widgets.text("env",           "dev",        "Environment (dev/prod)")
dbutils.widgets.text("start_date",    "2020-01-01", "Earliest filing date")
dbutils.widgets.text("ticker_filter", "",           "Comma-separated tickers (empty=all)")

CATALOG       = dbutils.widgets.get("catalog")
ENV           = dbutils.widgets.get("env")
START_DATE    = dbutils.widgets.get("start_date")
TICKER_FILTER = dbutils.widgets.get("ticker_filter")
TICKER_SUBSET = [t.strip() for t in TICKER_FILTER.split(",") if t.strip()] if TICKER_FILTER else []

print(f"[CONFIG] catalog={CATALOG} | env={ENV} | start_date={START_DATE} | tickers={TICKER_SUBSET or 'ALL'}")

# COMMAND ----------

# %pip install lxml

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql.functions import (
    col, when, lit, row_number, coalesce, current_timestamp, lag,
    datediff, to_date, year, month, lower, trim,
    sum as spark_sum, countDistinct, max as spark_max,
)
from pyspark.sql.window import Window
from delta.tables import DeltaTable

silver_table = f"{CATALOG}.finsage_silver.financial_statements"
gold_table   = f"{CATALOG}.finsage_gold.company_metrics"

# --- 1. Temporal filters: annual 10-K facts from the last 5 years only ---
df = (
    spark.table(silver_table)
    .filter(col("filing_type").rlike("^10-K"))
    .filter(col("fiscal_period") == "FY")
    .filter(col("fiscal_year") >= 2020)
    .withColumn("filing_date_dt",  to_date("filing_date"))
    .withColumn("period_start_dt", to_date("period_start"))
    .withColumn("period_end_dt",   to_date("period_end"))
    .withColumn("duration_days",   datediff(col("period_end_dt"), col("period_start_dt")))
    .withColumn("period_end_year", year("period_end_dt"))
    .withColumn("period_end_month", month("period_end_dt"))
    .withColumn("unit_norm",       lower(trim(col("unit"))))
    .withColumn("is_usd",          col("unit_norm").isin("usd", "usd/shares"))
)

# --- 2. Concept priority: metric-specific, deterministic deduplication ---
concept_priority = (
    when((col("normalized_line_item") == "revenue") & (col("raw_line_item") == "RevenueFromContractWithCustomerExcludingAssessedTax"), lit(1))
    .when((col("normalized_line_item") == "revenue") & (col("raw_line_item") == "SalesRevenueNet"),  lit(2))
    .when((col("normalized_line_item") == "revenue") & (col("raw_line_item") == "Revenues"),          lit(3))
    .when((col("normalized_line_item") == "net_income") & (col("raw_line_item") == "NetIncomeLoss"),  lit(1))
    .when((col("normalized_line_item") == "net_income") & (col("raw_line_item") == "ProfitLoss"),     lit(2))
    .when((col("normalized_line_item") == "gross_profit") & (col("raw_line_item") == "GrossProfit"),  lit(1))
    .when((col("normalized_line_item") == "operating_income") & (col("raw_line_item") == "OperatingIncomeLoss"), lit(1))
    .when((col("normalized_line_item") == "operating_cash_flow") & (col("raw_line_item") == "NetCashProvidedByUsedInOperatingActivities"), lit(1))
    .when((col("normalized_line_item") == "operating_cash_flow") & (col("raw_line_item") == "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"), lit(2))
    .when((col("normalized_line_item") == "total_assets") & (col("raw_line_item") == "Assets"),       lit(1))
    .when((col("normalized_line_item") == "total_liabilities") & (col("raw_line_item") == "Liabilities"), lit(1))
    .when((col("normalized_line_item") == "equity") & (col("raw_line_item") == "StockholdersEquity"), lit(1))
    .when((col("normalized_line_item") == "equity") & (col("raw_line_item") == "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"), lit(2))
    .when((col("normalized_line_item") == "rd_expense") & (col("raw_line_item") == "ResearchAndDevelopmentExpense"), lit(1))
    .when((col("normalized_line_item") == "cost_of_revenue") & (col("raw_line_item") == "CostOfRevenue"),   lit(1))
    .when((col("normalized_line_item") == "cost_of_revenue") & (col("raw_line_item") == "CostOfGoodsSold"), lit(2))
    .when((col("normalized_line_item") == "cost_of_revenue") & (col("raw_line_item") == "CostOfSales"),     lit(3))
    .otherwise(lit(99))
)

df = (
    df
    .withColumn("concept_priority", concept_priority)
    .withColumn("is_duration_metric", col("normalized_line_item").isin(
        "revenue", "net_income", "gross_profit", "operating_income",
        "operating_cash_flow", "rd_expense", "cost_of_revenue",
    ))
    .withColumn("is_instant_metric", col("normalized_line_item").isin(
        "total_assets", "total_liabilities", "short_term_debt",
        "long_term_debt", "equity",
    ))
    # Strict fit: period_end_year must equal declared fiscal_year.
    # (Retail companies like Walmart close Jan of the following calendar year, so
    #  fiscal_year 2023 can have period_end_year 2024 — allow +1 offset.)
    .withColumn("annual_fit_score", when(
        col("is_duration_metric") &
        col("duration_days").between(350, 380) &
        (col("period_end_year") == col("fiscal_year")),
        lit(1)
    ).otherwise(lit(0)))
    .withColumn("instant_fit_score", when(
        col("is_instant_metric") &
        (col("period_end_year") == col("fiscal_year")),
        lit(1)
    ).otherwise(lit(0)))
)

df = df.filter((col("annual_fit_score") == 1) | (col("instant_fit_score") == 1))

# --- 3. Pick canonical accession per ticker-year based on metric coverage ---
required_metric_flag = when(
    col("normalized_line_item").isin(
        "revenue", "net_income", "operating_income",
        "operating_cash_flow", "total_assets", "equity",
    ), lit(1)
).otherwise(lit(0))

df_accession_quality = (
    df
    .withColumn("usable_fact_flag", when(
        (
            (col("is_duration_metric") & col("annual_fit_score").eqNullSafe(1)) |
            (col("is_instant_metric")  & col("instant_fit_score").eqNullSafe(1))
        ) & col("is_usd"),
        lit(1)
    ).otherwise(lit(0)))
    .groupBy("ticker", "company_name", "fiscal_year", "accession")
    .agg(
        spark_sum(when(required_metric_flag == 1, col("usable_fact_flag")).otherwise(lit(0))).alias("required_metric_hits"),
        countDistinct(when(col("usable_fact_flag") == 1, col("normalized_line_item"))).alias("distinct_metric_coverage"),
        spark_max("filing_date_dt").alias("latest_filing_date"),
    )
)

accession_window = Window.partitionBy("ticker", "fiscal_year").orderBy(
    col("required_metric_hits").desc(),
    col("distinct_metric_coverage").desc(),
    col("latest_filing_date").desc(),
)

df_canonical_accession = (
    df_accession_quality
    .withColumn("rn", row_number().over(accession_window))
    .filter(col("rn") == 1)
    .select("ticker", "fiscal_year", "accession")
)

df_canonical = (
    df.alias("f")
    .join(
        df_canonical_accession.alias("a"),
        on=[
            col("f.ticker")      == col("a.ticker"),
            col("f.fiscal_year") == col("a.fiscal_year"),
            col("f.accession")   == col("a.accession"),
        ],
        how="inner",
    )
    .select("f.*")
)
df_canonical = df_canonical.filter((col("annual_fit_score") == 1) | (col("instant_fit_score") == 1))

# --- 4. One best fact per metric within the canonical accession ---
fact_window = Window.partitionBy("ticker", "fiscal_year", "normalized_line_item").orderBy(
    col("is_usd").desc(),
    col("annual_fit_score").desc(),
    col("instant_fit_score").desc(),
    col("concept_priority").asc(),
    col("filing_date_dt").desc_nulls_last(),
    col("source_fetched_at").desc_nulls_last(),
)

df_best_fact = (
    df_canonical
    .withColumn("rn", row_number().over(fact_window))
    .filter(col("rn") == 1)
    .drop("rn")
)

# --- 5. Aggregate base metrics ---
df_base = (
    df_best_fact
    .groupBy("ticker", "company_name", "fiscal_year")
    .agg(
        spark_max(when(col("normalized_line_item") == "revenue",           col("value"))).alias("revenue"),
        spark_max(when(col("normalized_line_item") == "net_income",        col("value"))).alias("net_income"),
        spark_max(when(col("normalized_line_item") == "gross_profit",      col("value"))).alias("gross_profit_raw"),
        spark_max(when(col("normalized_line_item") == "cost_of_revenue",   col("value"))).alias("cost_of_revenue"),
        spark_max(when(col("normalized_line_item") == "operating_income",  col("value"))).alias("operating_income"),
        spark_max(when(col("normalized_line_item") == "operating_cash_flow", col("value"))).alias("operating_cash_flow"),
        spark_max(when(col("normalized_line_item") == "total_assets",      col("value"))).alias("total_assets"),
        spark_max(when(col("normalized_line_item") == "total_liabilities", col("value"))).alias("total_liabilities_raw"),
        spark_max(when(col("normalized_line_item") == "short_term_debt",   col("value"))).alias("short_term_debt"),
        spark_max(when(col("normalized_line_item") == "long_term_debt",    col("value"))).alias("long_term_debt"),
        spark_max(when(col("normalized_line_item") == "rd_expense",        col("value"))).alias("rd_expense"),
        spark_max(when(col("normalized_line_item") == "equity",            col("value"))).alias("equity"),
    )
)

# --- 6. Derived metrics (always after canonical fact selection) ---
df_metrics = (
    df_base
    .withColumn("gross_profit",     coalesce(col("gross_profit_raw"), col("revenue") - col("cost_of_revenue")))
    .withColumn("total_liabilities", coalesce(col("total_liabilities_raw"), col("total_assets") - col("equity")))
    .withColumn("total_debt",
        when(col("short_term_debt").isNull() & col("long_term_debt").isNull(), lit(None).cast("double"))
        .otherwise(coalesce(col("short_term_debt"), lit(0.0)) + coalesce(col("long_term_debt"), lit(0.0)))
    )
    .withColumn("gross_margin_pct",
        when(col("revenue").isNotNull() & (col("revenue") != 0) & col("gross_profit").isNotNull(),
             col("gross_profit") / col("revenue"))
    )
)

yoy_window = Window.partitionBy("ticker").orderBy("fiscal_year")
df_metrics = (
    df_metrics
    .withColumn("prior_year_revenue", lag("revenue").over(yoy_window))
    .withColumn("revenue_yoy_growth_pct",
        when(
            col("prior_year_revenue").isNotNull() & (col("prior_year_revenue") != 0) & col("revenue").isNotNull(),
            (col("revenue") - col("prior_year_revenue")) / col("prior_year_revenue"),
        )
    )
    .withColumn("debt_to_equity",
        when(col("equity").isNotNull() & (col("equity") != 0) & col("total_debt").isNotNull(),
             col("total_debt") / col("equity"))
    )
)

validated_metric_count = (
    when(col("revenue").isNotNull(),            lit(1)).otherwise(lit(0)) +
    when(col("net_income").isNotNull(),         lit(1)).otherwise(lit(0)) +
    when(col("gross_profit").isNotNull(),       lit(1)).otherwise(lit(0)) +
    when(col("operating_income").isNotNull(),   lit(1)).otherwise(lit(0)) +
    when(col("operating_cash_flow").isNotNull(),lit(1)).otherwise(lit(0)) +
    when(col("total_assets").isNotNull(),       lit(1)).otherwise(lit(0)) +
    when(col("total_liabilities").isNotNull(),  lit(1)).otherwise(lit(0)) +
    when(col("total_debt").isNotNull(),         lit(1)).otherwise(lit(0)) +
    when(col("rd_expense").isNotNull(),         lit(1)).otherwise(lit(0))
)

df_gold = (
    df_metrics
    .withColumn("fiscal_quarter",         lit(None).cast("int"))
    .withColumn("data_quality_score",     validated_metric_count / lit(9.0))
    .withColumn("updated_at",             current_timestamp())
    .select(
        "ticker", "company_name", "fiscal_year", "fiscal_quarter",
        "revenue", "net_income", "gross_profit", "operating_income",
        "operating_cash_flow", "total_assets", "total_liabilities",
        "total_debt", "rd_expense", "gross_margin_pct",
        "revenue_yoy_growth_pct", "debt_to_equity",
        "data_quality_score", "updated_at",
    )
)

if spark.catalog.tableExists(gold_table):
    dt = DeltaTable.forName(spark, gold_table)
    dt.alias("t").merge(
        df_gold.alias("s"),
        "t.ticker = s.ticker AND t.fiscal_year = s.fiscal_year AND t.fiscal_quarter <=> s.fiscal_quarter",
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    print("Gold company_metrics merge complete.")
else:
    df_gold.write.format("delta").saveAsTable(gold_table)
    print("Gold company_metrics table created.")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM main.finsage_gold.company_metrics
# MAGIC ORDER BY ticker, fiscal_year;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     COUNT(*)                  AS gold_rows,
# MAGIC     COUNT(DISTINCT ticker)    AS company_count,
# MAGIC     MIN(fiscal_year)          AS min_year,
# MAGIC     MAX(fiscal_year)          AS max_year,
# MAGIC     AVG(data_quality_score)   AS avg_data_quality_score
# MAGIC FROM main.finsage_gold.company_metrics;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ticker, fiscal_year, value
# MAGIC FROM main.finsage_silver.financial_statements
# MAGIC WHERE normalized_line_item = 'revenue';

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM main.finsage_gold.company_metrics WHERE ticker = 'AAPL';
