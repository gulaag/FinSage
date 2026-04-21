# Databricks notebook source
# ==============================================================================
# FinSage | 04b — Gold Quarterly Metrics
#
# Parallel to notebook 04: builds main.finsage_gold.company_metrics_quarterly
# from 10-Q filings, with discrete-quarter flow metrics and point-in-time
# balance sheet values. Q4 is intentionally NOT derived in v1 — FY-minus-
# (Q1+Q2+Q3) can be added as a post-processing pass later.
#
# Shape is deliberately the same as company_metrics so the agent can use a
# parallel tool (get_quarterly_metrics) without reshaping results. The two
# tables are merged on disjoint keys (fiscal_quarter IS NULL vs 1/2/3), so
# there is no conflict with the annual table.
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

from pyspark.sql.functions import (
    col, when, lit, row_number, coalesce, current_timestamp, lag,
    datediff, to_date, year, lower, trim,
    sum as spark_sum, countDistinct, max as spark_max,
)
from pyspark.sql.window import Window
from delta.tables import DeltaTable

silver_table = f"{CATALOG}.finsage_silver.financial_statements"
gold_table   = f"{CATALOG}.finsage_gold.company_metrics_quarterly"

# --- 1. Temporal filters: 10-Q discrete-quarter facts from 2020+ ---
df = (
    spark.table(silver_table)
    .filter(col("filing_type") == "10-Q")
    .filter(col("fiscal_period").isin("Q1", "Q2", "Q3"))
    .filter(col("fiscal_year") >= 2020)
    .withColumn("filing_date_dt",  to_date("filing_date"))
    .withColumn("period_start_dt", to_date("period_start"))
    .withColumn("period_end_dt",   to_date("period_end"))
    .withColumn("duration_days",   datediff(col("period_end_dt"), col("period_start_dt")))
    .withColumn("period_end_year", year("period_end_dt"))
    .withColumn("unit_norm",       lower(trim(col("unit"))))
    .withColumn("is_usd",          col("unit_norm").isin("usd", "usd/shares"))
    .withColumn("fiscal_quarter",
        when(col("fiscal_period") == "Q1", lit(1))
        .when(col("fiscal_period") == "Q2", lit(2))
        .when(col("fiscal_period") == "Q3", lit(3))
    )
)

# --- 2. Concept priority: same taxonomy as annual table ---
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
    # Discrete-quarter flow: 80–100 day window excludes 6-month/YTD cumulative
    # disclosures that share the same fiscal_period label.
    .withColumn("quarterly_fit_score", when(
        col("is_duration_metric") &
        col("duration_days").between(80, 100) &
        (col("period_end_year") == col("fiscal_year")),
        lit(1)
    ).otherwise(lit(0)))
    .withColumn("instant_fit_score", when(
        col("is_instant_metric") &
        (col("period_end_year") == col("fiscal_year")),
        lit(1)
    ).otherwise(lit(0)))
)

df = df.filter((col("quarterly_fit_score") == 1) | (col("instant_fit_score") == 1))

# --- 3. Canonical accession per (ticker, fiscal_year, fiscal_quarter) ---
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
            (col("is_duration_metric") & col("quarterly_fit_score").eqNullSafe(1)) |
            (col("is_instant_metric")  & col("instant_fit_score").eqNullSafe(1))
        ) & col("is_usd"),
        lit(1)
    ).otherwise(lit(0)))
    .groupBy("ticker", "company_name", "fiscal_year", "fiscal_quarter", "accession")
    .agg(
        spark_sum(when(required_metric_flag == 1, col("usable_fact_flag")).otherwise(lit(0))).alias("required_metric_hits"),
        countDistinct(when(col("usable_fact_flag") == 1, col("normalized_line_item"))).alias("distinct_metric_coverage"),
        spark_max("filing_date_dt").alias("latest_filing_date"),
    )
)

accession_window = Window.partitionBy("ticker", "fiscal_year", "fiscal_quarter").orderBy(
    col("required_metric_hits").desc(),
    col("distinct_metric_coverage").desc(),
    col("latest_filing_date").desc(),
)

df_canonical_accession = (
    df_accession_quality
    .withColumn("rn", row_number().over(accession_window))
    .filter(col("rn") == 1)
    .select("ticker", "fiscal_year", "fiscal_quarter", "accession")
)

df_canonical = (
    df.alias("f")
    .join(
        df_canonical_accession.alias("a"),
        on=[
            col("f.ticker")         == col("a.ticker"),
            col("f.fiscal_year")    == col("a.fiscal_year"),
            col("f.fiscal_quarter") == col("a.fiscal_quarter"),
            col("f.accession")      == col("a.accession"),
        ],
        how="inner",
    )
    .select("f.*")
)
df_canonical = df_canonical.filter((col("quarterly_fit_score") == 1) | (col("instant_fit_score") == 1))

# --- 4. One best fact per metric within the canonical accession ---
fact_window = Window.partitionBy("ticker", "fiscal_year", "fiscal_quarter", "normalized_line_item").orderBy(
    col("is_usd").desc(),
    col("quarterly_fit_score").desc(),
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
    .groupBy("ticker", "company_name", "fiscal_year", "fiscal_quarter")
    .agg(
        spark_max(when(col("normalized_line_item") == "revenue",             col("value"))).alias("revenue"),
        spark_max(when(col("normalized_line_item") == "net_income",          col("value"))).alias("net_income"),
        spark_max(when(col("normalized_line_item") == "gross_profit",        col("value"))).alias("gross_profit_raw"),
        spark_max(when(col("normalized_line_item") == "cost_of_revenue",     col("value"))).alias("cost_of_revenue"),
        spark_max(when(col("normalized_line_item") == "operating_income",    col("value"))).alias("operating_income"),
        spark_max(when(col("normalized_line_item") == "operating_cash_flow", col("value"))).alias("operating_cash_flow"),
        spark_max(when(col("normalized_line_item") == "total_assets",        col("value"))).alias("total_assets"),
        spark_max(when(col("normalized_line_item") == "total_liabilities",   col("value"))).alias("total_liabilities_raw"),
        spark_max(when(col("normalized_line_item") == "short_term_debt",     col("value"))).alias("short_term_debt"),
        spark_max(when(col("normalized_line_item") == "long_term_debt",      col("value"))).alias("long_term_debt"),
        spark_max(when(col("normalized_line_item") == "rd_expense",          col("value"))).alias("rd_expense"),
        spark_max(when(col("normalized_line_item") == "equity",              col("value"))).alias("equity"),
        spark_max(col("period_end_dt")).alias("period_end_date"),
    )
)

# --- 6. Derived metrics ---
df_metrics = (
    df_base
    .withColumn("gross_profit",      coalesce(col("gross_profit_raw"), col("revenue") - col("cost_of_revenue")))
    .withColumn("total_liabilities", coalesce(col("total_liabilities_raw"), col("total_assets") - col("equity")))
    .withColumn("total_equity",      coalesce(col("equity"), col("total_assets") - col("total_liabilities")))
    .withColumn("total_debt",
        when(col("short_term_debt").isNull() & col("long_term_debt").isNull(), lit(None).cast("double"))
        .otherwise(coalesce(col("short_term_debt"), lit(0.0)) + coalesce(col("long_term_debt"), lit(0.0)))
    )
    .withColumn("gross_margin_pct",
        when(col("revenue").isNotNull() & (col("revenue") != 0) & col("gross_profit").isNotNull(),
             col("gross_profit") / col("revenue"))
    )
)

# Same-quarter YoY: compare Q2-FY2024 vs Q2-FY2023, partitioned by (ticker, quarter).
yoy_window = Window.partitionBy("ticker", "fiscal_quarter").orderBy("fiscal_year")
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
        when(col("total_equity").isNotNull() & (col("total_equity") != 0) & col("total_debt").isNotNull(),
             col("total_debt") / col("total_equity"))
    )
)

validated_metric_count = (
    when(col("revenue").isNotNull(),             lit(1)).otherwise(lit(0)) +
    when(col("net_income").isNotNull(),          lit(1)).otherwise(lit(0)) +
    when(col("gross_profit").isNotNull(),        lit(1)).otherwise(lit(0)) +
    when(col("operating_income").isNotNull(),    lit(1)).otherwise(lit(0)) +
    when(col("operating_cash_flow").isNotNull(), lit(1)).otherwise(lit(0)) +
    when(col("total_assets").isNotNull(),        lit(1)).otherwise(lit(0)) +
    when(col("total_liabilities").isNotNull(),   lit(1)).otherwise(lit(0)) +
    when(col("total_debt").isNotNull(),          lit(1)).otherwise(lit(0)) +
    when(col("rd_expense").isNotNull(),          lit(1)).otherwise(lit(0))
)

df_gold = (
    df_metrics
    .withColumn("data_quality_score", validated_metric_count / lit(9.0))
    .withColumn("source_filing_type", lit("10-Q"))
    .withColumn("updated_at",         current_timestamp())
    .select(
        "ticker", "company_name", "fiscal_year", "fiscal_quarter", "period_end_date",
        "revenue", "net_income", "gross_profit", "operating_income",
        "operating_cash_flow", "total_assets", "total_liabilities",
        "total_equity", "total_debt", "rd_expense", "gross_margin_pct",
        "revenue_yoy_growth_pct", "debt_to_equity",
        "data_quality_score", "source_filing_type", "updated_at",
    )
)

# --- 7. Idempotent merge ---
spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")

if spark.catalog.tableExists(gold_table):
    dt = DeltaTable.forName(spark, gold_table)
    dt.alias("t").merge(
        df_gold.alias("s"),
        "t.ticker = s.ticker AND t.fiscal_year = s.fiscal_year AND t.fiscal_quarter = s.fiscal_quarter",
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    print("Gold company_metrics_quarterly merge complete.")
else:
    df_gold.write.format("delta").saveAsTable(gold_table)
    print("Gold company_metrics_quarterly table created.")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     COUNT(*)                                         AS quarterly_rows,
# MAGIC     COUNT(DISTINCT ticker)                           AS tickers,
# MAGIC     COUNT(DISTINCT CONCAT(ticker,'-',fiscal_year,'-',fiscal_quarter)) AS distinct_quarters,
# MAGIC     MIN(fiscal_year)                                 AS min_fy,
# MAGIC     MAX(fiscal_year)                                 AS max_fy,
# MAGIC     ROUND(AVG(data_quality_score), 3)                AS avg_dq
# MAGIC FROM main.finsage_gold.company_metrics_quarterly;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ticker, fiscal_year, fiscal_quarter,
# MAGIC        ROUND(revenue / 1e9, 2)            AS revenue_b,
# MAGIC        ROUND(net_income / 1e9, 2)         AS net_income_b,
# MAGIC        ROUND(operating_income / 1e9, 2)   AS op_income_b,
# MAGIC        ROUND(revenue_yoy_growth_pct, 4)   AS yoy_pct,
# MAGIC        ROUND(data_quality_score, 2)       AS dq
# MAGIC FROM main.finsage_gold.company_metrics_quarterly
# MAGIC WHERE ticker IN ('AAPL','MSFT','NVDA','TSLA')
# MAGIC ORDER BY ticker, fiscal_year DESC, fiscal_quarter DESC
# MAGIC LIMIT 40;
