# Databricks notebook source
# ==============================================================================
# FinSage | 03 — Silver Decoder
# Two transformations run in this notebook:
#   A) Flatten XBRL CompanyFacts JSON  → main.finsage_silver.financial_statements
#   B) Extract 10-K text sections      → main.finsage_silver.filing_sections
#
# TARGET_CONCEPT_MAP is the canonical mapping tested in tests/unit/test_normalizer.py
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

# ── A) XBRL CompanyFacts → financial_statements ─────────────────────────────

from pyspark.sql import Row
from pyspark.sql.functions import (
    col, row_number, current_timestamp, lit, when,
    sha2, concat_ws, coalesce,
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, TimestampType,
)
from pyspark.sql.window import Window
from delta.tables import DeltaTable
import json

# ---------------------------------------------------------------------------
# Import TARGET_CONCEPT_MAP from the shared constants module.
# When the finsage wheel is installed on the cluster:
#   from finsage.constants import TARGET_CONCEPT_MAP, STATEMENT_TYPE_MAP
# For interactive Databricks sessions without the wheel, fall back to inline.
# ---------------------------------------------------------------------------
try:
    from finsage.constants import TARGET_CONCEPT_MAP, STATEMENT_TYPE_MAP
except ImportError:
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
    STATEMENT_TYPE_MAP = {k: "financial_metric" for k in TARGET_CONCEPT_MAP.values()}

bronze_api = (
    spark.table(f"{CATALOG}.finsage_bronze.xbrl_companyfacts_raw")
    .filter(col("api_status") == "success")
)

def flatten_companyfacts(row):
    out = []
    try:
        payload = json.loads(row.raw_json)
        us_gaap = payload.get("facts", {}).get("us-gaap", {})
        for concept, concept_body in us_gaap.items():
            normalized_line_item = TARGET_CONCEPT_MAP.get(concept)
            if not normalized_line_item:
                continue
            units_map = concept_body.get("units", {})
            for unit, entries in units_map.items():
                for e in entries:
                    filing_type  = e.get("form")
                    fiscal_year  = e.get("fy")
                    accession    = e.get("accn")
                    raw_value    = e.get("val")
                    if filing_type not in ("10-K", "10-Q") or fiscal_year is None or accession is None:
                        continue
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError):
                        continue
                    out.append(Row(
                        ticker=row.ticker, cik=row.cik, company_name=row.entity_name,
                        raw_line_item=concept, normalized_line_item=normalized_line_item,
                        statement_type=STATEMENT_TYPE_MAP.get(normalized_line_item, "unknown"),
                        unit=unit, value=value, fiscal_year=int(fiscal_year),
                        fiscal_period=e.get("fp"), filing_type=filing_type,
                        filing_date=e.get("filed"), period_start=e.get("start"),
                        period_end=e.get("end"), accession=accession,
                        frame=e.get("frame"), source_url=row.source_url,
                        source_fetched_at=row.fetched_at,
                    ))
    except Exception as e:
        out.append(Row(
            ticker=row.ticker, cik=row.cik, company_name=row.entity_name,
            raw_line_item="ERROR", normalized_line_item="ERROR",
            statement_type="ERROR", unit=None, value=None, fiscal_year=None,
            fiscal_period=None, filing_type=None, filing_date=None,
            period_start=None, period_end=None, accession=None, frame=None,
            source_url=row.source_url, source_fetched_at=row.fetched_at,
        ))
    return out

facts_schema = StructType([
    StructField("ticker",             StringType(),    True),
    StructField("cik",                StringType(),    True),
    StructField("company_name",       StringType(),    True),
    StructField("raw_line_item",      StringType(),    True),
    StructField("normalized_line_item", StringType(),  True),
    StructField("statement_type",     StringType(),    True),
    StructField("unit",               StringType(),    True),
    StructField("value",              DoubleType(),    True),
    StructField("fiscal_year",        IntegerType(),   True),
    StructField("fiscal_period",      StringType(),    True),
    StructField("filing_type",        StringType(),    True),
    StructField("filing_date",        StringType(),    True),
    StructField("period_start",       StringType(),    True),
    StructField("period_end",         StringType(),    True),
    StructField("accession",          StringType(),    True),
    StructField("frame",              StringType(),    True),
    StructField("source_url",         StringType(),    True),
    StructField("source_fetched_at",  TimestampType(), True),
])

df_facts = spark.createDataFrame(
    bronze_api.rdd.flatMap(flatten_companyfacts), schema=facts_schema
)

df_financials = (
    df_facts
    .withColumn(
        "fiscal_quarter",
        when(col("fiscal_period") == "Q1", 1)
        .when(col("fiscal_period") == "Q2", 2)
        .when(col("fiscal_period") == "Q3", 3)
        .when(col("fiscal_period") == "Q4", 4)
        .otherwise(None)
    )
    .withColumn(
        "filing_id",
        concat_ws("-", col("ticker"), col("filing_type"),
                  col("fiscal_year").cast("string"), col("accession"))
    )
    .withColumn(
        "statement_id",
        sha2(concat_ws(
            "||",
            coalesce(col("ticker"),      lit("")),
            coalesce(col("accession"),   lit("")),
            coalesce(col("raw_line_item"), lit("")),
            coalesce(col("unit"),        lit("")),
            coalesce(col("period_end"),  lit("")),
        ), 256)
    )
    .withColumn("extraction_confidence", lit(1.0))
    .withColumn("xbrl_validated",        lit(True))
    .withColumn("xbrl_match",            lit(True))
    .withColumn("parsed_at",             current_timestamp())
)

window_spec = Window.partitionBy("statement_id").orderBy(
    col("source_fetched_at").desc(),
    col("filing_date").desc_nulls_last(),
)
df_financials_latest = (
    df_financials
    .withColumn("rn", row_number().over(window_spec))
    .filter(col("rn") == 1)
    .drop("rn")
)

# Log JSON parse errors to the Bronze error table before filtering them out.
df_json_errors = df_financials_latest.filter(col("raw_line_item") == "ERROR")
if df_json_errors.count() > 0:
    print(f"Warning: {df_json_errors.count()} JSON records failed to parse. Logging to ingestion_errors.")
    (
        df_json_errors.select(
            sha2(concat_ws("||", col("ticker"), col("source_url")), 256).alias("error_id"),
            lit("silver_json_flattening").alias("source_system"),
            col("source_url"),
            lit(None).cast("string").alias("file_path"),
            lit("json_parse_failure").alias("error_type"),
            lit("Exception thrown during JSON parsing").alias("error_message"),
            lit(0).alias("retry_count"),
            current_timestamp().alias("failed_at"),
        )
        .write.format("delta").mode("append")
        .saveAsTable("main.finsage_bronze.ingestion_errors")
    )

df_financials_latest = df_financials_latest.filter(col("raw_line_item") != "ERROR")

silver_table = f"{CATALOG}.finsage_silver.financial_statements"
if spark.catalog.tableExists(silver_table):
    DeltaTable.forName(spark, silver_table).alias("t").merge(
        df_financials_latest.alias("s"), "t.statement_id = s.statement_id"
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
else:
    df_financials_latest.write.format("delta").saveAsTable(silver_table)

print("Silver financial_statements processing complete.")

# COMMAND ----------

# ── B) 10-K text sections → filing_sections ─────────────────────────────────

from pyspark.sql.functions import (
    udf, col, decode, expr, regexp_replace, explode,
    current_timestamp, lit, row_number, sha2, concat_ws,
)
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType
import re

SECTION_RULES_10K = {
    "Business": {
        "start_patterns": [r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+1\b(?!\s*[ab]\b)"],
        "end_patterns":   [r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+1a\b",
                           r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+2\b"],
        "min_words": 250, "fallback_chars": 250000,
    },
    "Risk Factors": {
        "start_patterns": [r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+1a\b"],
        "end_patterns":   [r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+1b\b",
                           r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+2\b"],
        "min_words": 400, "fallback_chars": 350000,
    },
    "MD&A": {
        "start_patterns": [r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+7\b(?!\s*a\b)"],
        "end_patterns":   [r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+7a\b",
                           r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+8\b"],
        "min_words": 400, "fallback_chars": 350000,
    },
}

# 10-Q structure differs: MD&A is Part I Item 2 (not Item 7), Risk Factor updates
# are Part II Item 1A (only if material changes since last 10-K). Item 1 in a 10-Q
# is the condensed financial statements — not narrative Business discussion —
# so we intentionally do NOT extract "Business" from 10-Q filings.
SECTION_RULES_10Q = {
    "MD&A": {
        "start_patterns": [r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+2\b(?!\s*a\b)"],
        "end_patterns":   [r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+3\b",
                           r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+4\b",
                           r"(?im)^[\s>\-\.\(\)\d]{0,12}part\s+ii\b"],
        "min_words": 200, "fallback_chars": 300000,
    },
    "Risk Factors Updates": {
        "start_patterns": [r"(?im)^[\s>\-\.\(\)\d]{0,12}part\s+ii.{0,80}item\s+1a\b",
                           r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+1a\.\s*risk\s+factors\b"],
        "end_patterns":   [r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+2\.\s*unregistered",
                           r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+6\b",
                           r"(?im)^[\s>\-\.\(\)\d]{0,12}signatures?\b"],
        "min_words": 80, "fallback_chars": 100000,
    },
}

SECTION_RULES_BY_FORM = {
    "10-K": SECTION_RULES_10K,
    "10-Q": SECTION_RULES_10Q,
}

def _collect_positions(patterns, text):
    return sorted(set([
        match.start()
        for pattern in patterns
        for match in re.finditer(pattern, text)
    ]))

def _normalize_text(text):
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def _choose_best_block(text, rule):
    starts     = _collect_positions(rule["start_patterns"], text)
    ends       = _collect_positions(rule["end_patterns"],   text)
    if not starts:
        return None
    doc_len, best, best_score = max(len(text), 1), None, -1
    for s in starts:
        end_candidates = [e for e in ends if e > s + 25]
        e = end_candidates[0] if end_candidates else min(len(text), s + rule["fallback_chars"])
        candidate  = text[s:e].strip()
        word_count = len(candidate.split())
        if word_count < rule["min_words"]:
            continue
        score = word_count + ((s / doc_len) * 250)
        if score > best_score:
            best_score = score
            best = {"section_text": candidate, "word_count": word_count, "start_pos": s, "end_pos": e}
    return best

def extract_sections_hardened(clean_text, filing_type):
    if not clean_text:
        return {"sections": [], "error": "Empty text after cleaning"}
    rules = SECTION_RULES_BY_FORM.get(filing_type)
    if rules is None:
        return {"sections": [], "error": f"Unsupported filing_type: {filing_type}"}
    try:
        text     = _normalize_text(clean_text)
        sections = []
        for section_name, rule in rules.items():
            best_block = _choose_best_block(text, rule)
            if best_block:
                sections.append({
                    "section_name": section_name,
                    "section_text": best_block["section_text"],
                    "word_count":   best_block["word_count"],
                })
        # For 10-K all three sections are required; for 10-Q only MD&A is required
        # ("Risk Factors Updates" is present only when management flags changes).
        if filing_type == "10-K":
            required = {"Business", "Risk Factors", "MD&A"}
        else:
            required = {"MD&A"}
        missing = sorted(required - {s["section_name"] for s in sections})
        if missing:
            return {"sections": sections, "error": f"Missing sections: {', '.join(missing)}"}
        return {"sections": sections, "error": None}
    except Exception as e:
        return {"sections": [], "error": f"Section extraction error: {str(e)}"}

split_udf = udf(
    extract_sections_hardened,
    StructType([
        StructField("sections", ArrayType(StructType([
            StructField("section_name", StringType()),
            StructField("section_text", StringType()),
            StructField("word_count",   IntegerType()),
        ]))),
        StructField("error", StringType()),
    ])
)

df_bronze_clean = (
    spark.table(f"{CATALOG}.finsage_bronze.filings")
    .filter(col("filing_type").isin("10-K", "10-Q"))
    .withColumn("rn", row_number().over(
        Window.partitionBy("filing_id").orderBy(col("ingested_at").desc())
    ))
    .filter(col("rn") == 1)
    .drop("rn")
)

df_processed = (
    df_bronze_clean
    .withColumn("raw_text",       decode(col("content"), "UTF-8"))
    .withColumn("main_doc",       expr("substring_index(raw_text, '</DOCUMENT>', 1)"))
    # Strip base64-encoded images, scripts, and styles before text extraction
    .withColumn("no_images",      regexp_replace(col("main_doc"),  r"(?is)<img[^>]*src=[\"']data:image/[^>]*>", " "))
    .withColumn("no_script",      regexp_replace(col("no_images"), r"(?is)<script[^>]*>.*?</script>",          " "))
    .withColumn("no_style",       regexp_replace(col("no_script"), r"(?is)<style[^>]*>.*?</style>",            " "))
    .withColumn("text_with_breaks", regexp_replace(col("no_style"),
        r"(?i)</?(div|p|br|tr|li|table|tbody|thead|tfoot|td|th|h1|h2|h3|h4|h5|h6)[^>]*>", "\n"))
    .withColumn("no_html",        regexp_replace(col("text_with_breaks"), "<[^>]+>", " "))
    .withColumn("clean_text",     regexp_replace(col("no_html"),   "\u00a0", " "))
    .withColumn("clean_text",     regexp_replace(col("clean_text"), r"[\t\x0B\f\r ]+", " "))
    .withColumn("clean_text",     regexp_replace(col("clean_text"), r"\n{3,}", "\n\n"))
    .withColumn("udf_result",     split_udf(col("clean_text"), col("filing_type")))
    .select("filing_id", "ticker", "fiscal_year", "filing_type", "file_path",
            "udf_result.sections", "udf_result.error")
)
df_processed.cache()

df_errors = df_processed.filter(col("error").isNotNull())
if df_errors.count() > 0:
    print(f"Warning: {df_errors.count()} filings had missing sections. Logged to ingestion_errors.")
    df_errors.select(
        sha2(concat_ws("||", col("file_path"), col("error")), 256).alias("error_id"),
        lit("silver_section_extraction").alias("source_system"),
        lit(None).cast("string").alias("source_url"),
        col("file_path"),
        lit("parse_failure").alias("error_type"),
        col("error").alias("error_message"),
        lit(0).alias("retry_count"),
        current_timestamp().alias("failed_at"),
    ).write.format("delta").mode("append").saveAsTable("main.finsage_bronze.ingestion_errors")

df_final_sections = (
    df_processed.filter(col("error").isNull())
    .withColumn("sec", explode("sections"))
    .select(
        # filing_id is already unique per SEC accession, so (filing_id, section_name)
        # yields a unique section_id even across 10-K/10-Q without an explicit form tag.
        sha2(concat_ws("||", col("filing_id"), col("sec.section_name")), 256).alias("section_id"),
        "filing_id", "ticker", "fiscal_year", "filing_type",
        col("sec.section_name").alias("section_name"),
        col("sec.section_text").alias("section_text"),
        col("sec.word_count").alias("word_count"),
        current_timestamp().alias("parsed_at"),
    )
)

# Enable schema autoMerge so adding filing_type on an existing table is a no-op rebuild
spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")

filing_sections_table = f"{CATALOG}.finsage_silver.filing_sections"
if spark.catalog.tableExists(filing_sections_table):
    DeltaTable.forName(spark, filing_sections_table).alias("t").merge(
        df_final_sections.alias("s"), "t.section_id = s.section_id"
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    print("Silver filing_sections merge complete.")
else:
    df_final_sections.write.format("delta").saveAsTable(filing_sections_table)
    spark.sql(f"ALTER TABLE {filing_sections_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    print("Silver filing_sections table created with CDF enabled.")
df_processed.unpersist()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT error_id, source_system, source_url, file_path, error_type, error_message
# MAGIC FROM main.finsage_bronze.ingestion_errors;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     ticker,
# MAGIC     section_name,
# MAGIC     COUNT(*)         AS total_sections,
# MAGIC     AVG(word_count)  AS avg_words
# MAGIC FROM main.finsage_silver.filing_sections
# MAGIC GROUP BY ticker, section_name
# MAGIC ORDER BY ticker, section_name;

# COMMAND ----------

from pyspark.sql.functions import col

display(
    spark.table("main.finsage_silver.filing_sections")
    .select(
        "filing_id", "section_name", "word_count",
        col("section_text").substr(1, 10000).alias("text_snippet")
    )
    .limit(10)
)

# COMMAND ----------

from pyspark.sql.functions import col

df_all = spark.table("main.finsage_silver.filing_sections")
df_safe_view = df_all.select(
    "filing_id", "ticker", "fiscal_year", "section_name", "word_count",
    col("section_text").substr(1, 1500).alias("text_preview..."),
    "parsed_at",
)
display(df_safe_view)

# COMMAND ----------

from pyspark.sql.functions import col

df_financials = spark.table("main.finsage_silver.financial_statements")
display(df_financials.select(
    "statement_id", "filing_id", "ticker", "fiscal_year", "filing_type",
    "statement_type", "raw_line_item", "normalized_line_item",
    "value", "unit", "xbrl_validated", "parsed_at",
))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     ticker, fiscal_year, section_name, word_count,
# MAGIC     LEFT(section_text,  80) AS boundary_start,
# MAGIC     RIGHT(section_text, 80) AS boundary_end
# MAGIC FROM main.finsage_silver.filing_sections
# MAGIC ORDER BY ticker, fiscal_year, section_name
# MAGIC LIMIT 20;
