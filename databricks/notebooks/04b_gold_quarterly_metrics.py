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
from functools import reduce as _reduce
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

FLOW_METRICS = [
    "revenue", "net_income", "gross_profit", "operating_income",
    "operating_cash_flow", "rd_expense", "cost_of_revenue",
]
INSTANT_METRICS = [
    "total_assets", "total_liabilities", "short_term_debt",
    "long_term_debt", "equity",
]

# --- 3. Classify period_type per fact ---
# Most filers XBRL-tag ONLY the cumulative flow concepts in Q2/Q3 10-Qs
# (SixMonthsEnded, NineMonthsEnded) and NOT the discrete three-months-ended
# variant. Silver confirms this: zero 80–100 day revenue facts exist at
# fiscal_period='Q2' or 'Q3' across all 30 tickers. We therefore keep both
# discrete and cumulative facts, then derive the discrete Q2/Q3 flows by
# subtraction downstream:
#   Q2_discrete = coalesce(Q2_90d_tagged, Q2_YTD_6mo − Q1_90d)
#   Q3_discrete = coalesce(Q3_90d_tagged, Q3_YTD_9mo − Q2_YTD_6mo)
#
# Do NOT gate on period_end_year == fiscal_year. Fiscal-year-offset companies
# (AAPL/MSFT/NKE/V/WMT with Sep/Jun/May/Jan/Jan year-ends) legitimately have
# quarter-ends falling in fiscal_year ± 1 on the calendar. Prior-period
# comparatives embedded in the same 10-Q (stamped upstream with the current
# fiscal_year) are instead rejected downstream by `period_end_dt DESC` in the
# fact_window tiebreaker.
df = (
    df
    .withColumn("concept_priority", concept_priority)
    .withColumn("is_duration_metric", col("normalized_line_item").isin(FLOW_METRICS))
    .withColumn("is_instant_metric",  col("normalized_line_item").isin(INSTANT_METRICS))
    .withColumn("period_type",
        when(col("is_duration_metric") & col("duration_days").between(80, 100), lit("discrete"))
        .when(col("is_duration_metric") & col("duration_days").between(170, 195) & (col("fiscal_period") == "Q2"), lit("ytd_6mo"))
        .when(col("is_duration_metric") & col("duration_days").between(260, 285) & (col("fiscal_period") == "Q3"), lit("ytd_9mo"))
        .when(col("is_instant_metric"), lit("instant"))
    )
    .filter(col("period_type").isNotNull() & col("is_usd"))
)

# --- 4. One best fact per (ticker, fy, fiscal_period, period_type, metric) ---
# period_end_dt DESC discards prior-year comparatives from the same 10-Q.
# concept_priority ASC prefers canonical raw_line_item per normalized metric.
fact_window = Window.partitionBy(
    "ticker", "fiscal_year", "fiscal_period", "period_type", "normalized_line_item"
).orderBy(
    col("period_end_dt").desc_nulls_last(),
    col("concept_priority").asc(),
    col("filing_date_dt").desc_nulls_last(),
    col("source_fetched_at").desc_nulls_last(),
)

df_best = (
    df
    .withColumn("rn", row_number().over(fact_window))
    .filter(col("rn") == 1)
    .drop("rn")
)

# --- 5. Wide aggregation per (ticker, fy): 53 slot columns ---
def _pick_value(metric: str, fp: str, pt: str):
    return spark_max(when(
        (col("normalized_line_item") == metric) &
        (col("fiscal_period") == fp) &
        (col("period_type") == pt),
        col("value"),
    ))

def _pick_period_end(fp: str):
    # Prefer instant (which is the actual quarter-end) then discrete, then cumulative.
    return spark_max(when(
        (col("fiscal_period") == fp) &
        (col("period_type").isin("instant", "discrete", "ytd_6mo", "ytd_9mo")),
        col("period_end_dt"),
    ))

_agg_exprs = []
# Q1: discrete flows only (Q1 10-Qs never emit ytd concepts)
for m in FLOW_METRICS:
    _agg_exprs.append(_pick_value(m, "Q1", "discrete").alias(f"q1_{m}_disc"))
# Q2: discrete (rare — F/GM/RIVN only) + ytd_6mo
for m in FLOW_METRICS:
    _agg_exprs.append(_pick_value(m, "Q2", "discrete").alias(f"q2_{m}_disc"))
    _agg_exprs.append(_pick_value(m, "Q2", "ytd_6mo").alias(f"q2_{m}_ytd6"))
# Q3: discrete + ytd_9mo
for m in FLOW_METRICS:
    _agg_exprs.append(_pick_value(m, "Q3", "discrete").alias(f"q3_{m}_disc"))
    _agg_exprs.append(_pick_value(m, "Q3", "ytd_9mo").alias(f"q3_{m}_ytd9"))
# Instants per quarter (point-in-time balance sheet values)
for q in ["Q1", "Q2", "Q3"]:
    qn = q.lower()
    for m in INSTANT_METRICS:
        _agg_exprs.append(_pick_value(m, q, "instant").alias(f"{qn}_{m}"))
# Per-quarter period_end_date (quarter-end, from instant preferentially)
for q in ["Q1", "Q2", "Q3"]:
    _agg_exprs.append(_pick_period_end(q).alias(f"{q.lower()}_period_end"))

df_wide = df_best.groupBy("ticker", "company_name", "fiscal_year").agg(*_agg_exprs)

# --- 6. Unpivot into 3 rows per ticker/fy with discrete-first + YTD fallback ---
def _derive_q2(m):
    # If a filer tagged the 90-day Q2 concept directly, use it; otherwise
    # back it out from YTD_6mo − Q1_90d.
    return coalesce(col(f"q2_{m}_disc"), col(f"q2_{m}_ytd6") - col(f"q1_{m}_disc"))

def _derive_q3(m):
    # Q3 standalone = YTD_9mo − YTD_6mo (or direct discrete if tagged).
    return coalesce(col(f"q3_{m}_disc"), col(f"q3_{m}_ytd9") - col(f"q2_{m}_ytd6"))

df_q1 = df_wide.select(
    col("ticker"), col("company_name"), col("fiscal_year"),
    lit(1).alias("fiscal_quarter"),
    col("q1_period_end").alias("period_end_date"),
    col("q1_revenue_disc").alias("revenue"),
    col("q1_net_income_disc").alias("net_income"),
    col("q1_gross_profit_disc").alias("gross_profit_raw"),
    col("q1_cost_of_revenue_disc").alias("cost_of_revenue"),
    col("q1_operating_income_disc").alias("operating_income"),
    col("q1_operating_cash_flow_disc").alias("operating_cash_flow"),
    col("q1_rd_expense_disc").alias("rd_expense"),
    col("q1_total_assets").alias("total_assets"),
    col("q1_total_liabilities").alias("total_liabilities_raw"),
    col("q1_equity").alias("equity"),
    col("q1_short_term_debt").alias("short_term_debt"),
    col("q1_long_term_debt").alias("long_term_debt"),
)

df_q2 = df_wide.select(
    col("ticker"), col("company_name"), col("fiscal_year"),
    lit(2).alias("fiscal_quarter"),
    col("q2_period_end").alias("period_end_date"),
    _derive_q2("revenue").alias("revenue"),
    _derive_q2("net_income").alias("net_income"),
    _derive_q2("gross_profit").alias("gross_profit_raw"),
    _derive_q2("cost_of_revenue").alias("cost_of_revenue"),
    _derive_q2("operating_income").alias("operating_income"),
    _derive_q2("operating_cash_flow").alias("operating_cash_flow"),
    _derive_q2("rd_expense").alias("rd_expense"),
    col("q2_total_assets").alias("total_assets"),
    col("q2_total_liabilities").alias("total_liabilities_raw"),
    col("q2_equity").alias("equity"),
    col("q2_short_term_debt").alias("short_term_debt"),
    col("q2_long_term_debt").alias("long_term_debt"),
)

df_q3 = df_wide.select(
    col("ticker"), col("company_name"), col("fiscal_year"),
    lit(3).alias("fiscal_quarter"),
    col("q3_period_end").alias("period_end_date"),
    _derive_q3("revenue").alias("revenue"),
    _derive_q3("net_income").alias("net_income"),
    _derive_q3("gross_profit").alias("gross_profit_raw"),
    _derive_q3("cost_of_revenue").alias("cost_of_revenue"),
    _derive_q3("operating_income").alias("operating_income"),
    _derive_q3("operating_cash_flow").alias("operating_cash_flow"),
    _derive_q3("rd_expense").alias("rd_expense"),
    col("q3_total_assets").alias("total_assets"),
    col("q3_total_liabilities").alias("total_liabilities_raw"),
    col("q3_equity").alias("equity"),
    col("q3_short_term_debt").alias("short_term_debt"),
    col("q3_long_term_debt").alias("long_term_debt"),
)

df_base = _reduce(lambda a, b: a.unionByName(b), [df_q1, df_q2, df_q3])
# Drop (ticker, fy, fq) rows that have no period_end_date — i.e. the quarter
# was never filed. Keeps the output tight and avoids ghost rows with all-NULL
# balance sheet values.
df_base = df_base.filter(col("period_end_date").isNotNull())

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
