# Databricks notebook source
# ==============================================================================
# FinSage | 02 — Bronze Auto Loader
# Streams raw SEC filing bytes into Delta Lake using Databricks Auto Loader
# (cloudFiles format).  Also ingests XBRL CompanyFacts JSON from the SEC API.
# Bronze is append-only and auditable — zero business logic applied here.
# ==============================================================================

# COMMAND ----------

# Set RESET_PIPELINE = True ONLY when you need a full re-ingestion from scratch.
RESET_PIPELINE = False
if RESET_PIPELINE:
    print("Nuking tables and checkpoints...")
    spark.sql("DROP TABLE IF EXISTS main.finsage_bronze.filings")
    spark.sql("DROP TABLE IF EXISTS main.finsage_bronze.ingestion_errors")
    spark.sql("DROP TABLE IF EXISTS main.finsage_bronze.xbrl_companyfacts_raw")
    dbutils.fs.rm("/Volumes/main/finsage_bronze/checkpoints", recurse=True)
    print("Reset complete. Ready for fresh ingestion.")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Bronze is append-only and auditable: raw filing bytes, raw SEC API payloads,
# MAGIC -- and every failure.
# MAGIC CREATE TABLE IF NOT EXISTS main.finsage_bronze.ingestion_errors (
# MAGIC     error_id       STRING,
# MAGIC     source_system  STRING,
# MAGIC     source_url     STRING,
# MAGIC     file_path      STRING,
# MAGIC     error_type     STRING,
# MAGIC     error_message  STRING,
# MAGIC     retry_count    INT,
# MAGIC     failed_at      TIMESTAMP
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS main.finsage_bronze.filings (
# MAGIC     filing_id         STRING,
# MAGIC     ticker            STRING,
# MAGIC     filing_type       STRING,
# MAGIC     accession_number  STRING,
# MAGIC     fiscal_year       INT,
# MAGIC     file_path         STRING,
# MAGIC     content           BINARY,
# MAGIC     file_size_bytes   LONG,
# MAGIC     ingestion_status  STRING,
# MAGIC     ingested_at       TIMESTAMP
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS main.finsage_bronze.xbrl_companyfacts_raw (
# MAGIC     snapshot_id      STRING,
# MAGIC     ticker           STRING,
# MAGIC     cik              STRING,
# MAGIC     entity_name      STRING,
# MAGIC     source_url       STRING,
# MAGIC     raw_json         STRING,
# MAGIC     api_status       STRING,
# MAGIC     http_status_code INT,
# MAGIC     error_message    STRING,
# MAGIC     fetched_at       TIMESTAMP
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS main.finsage_bronze.checkpoints;

# COMMAND ----------

from pyspark.sql.functions import col, current_timestamp, split, lit, concat, concat_ws

volume_path      = "/Volumes/main/finsage_bronze/raw_filings/sec-edgar-filings/"
checkpoint_path  = "/Volumes/main/finsage_bronze/checkpoints/bronze_final_v1"
schema_location  = "/Volumes/main/finsage_bronze/checkpoints/schema_v1"
bad_records_path = "/Volumes/main/finsage_bronze/checkpoints/bad_records"
target_table     = "main.finsage_bronze.filings"

df_bronze = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("cloudFiles.schemaLocation", schema_location)
    .option("recursiveFileLookup", "true")
    .load(volume_path)
    .withColumn("file_path",        col("_metadata.file_path"))
    .withColumn("ticker",           split(col("file_path"), "/").getItem(6))
    .withColumn("filing_type",      split(col("file_path"), "/").getItem(7))
    .withColumn("accession_number", split(col("file_path"), "/").getItem(8))
    .withColumn("year_short",       split(col("accession_number"), "-").getItem(1))
    .withColumn("fiscal_year",      concat(lit("20"), col("year_short")).cast("int"))
    .withColumn(
        "filing_id",
        concat_ws("-", col("ticker"), col("filing_type"), col("fiscal_year"), col("accession_number"))
    )
    .withColumn("file_size_bytes",  col("length"))
    .withColumn("ingestion_status", lit("success"))
    .withColumn("ingested_at",      current_timestamp())
    .select(
        "filing_id", "ticker", "filing_type", "accession_number",
        "fiscal_year", "file_path", "content", "file_size_bytes",
        "ingestion_status", "ingested_at"
    )
)

(
    df_bronze.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", checkpoint_path)
    .option("badRecordsPath",     bad_records_path)
    .option("mergeSchema",        "true")
    .trigger(availableNow=True)
    .toTable(target_table)
)

# COMMAND ----------

# Diagnostic: inspect path array indices if the split positions ever shift.
# from pyspark.sql.functions import split, col
# df_check = spark.table("main.finsage_bronze.filings").select(
#     "file_path",
#     split(col("file_path"), "/").alias("path_array")
# ).limit(1)
# display(df_check)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     filing_id,
# MAGIC     ticker,
# MAGIC     filing_type,
# MAGIC     accession_number,
# MAGIC     fiscal_year,
# MAGIC     ingestion_status,
# MAGIC     COUNT(*) AS record_count
# MAGIC FROM main.finsage_bronze.filings
# MAGIC GROUP BY 1, 2, 3, 4, 5, 6
# MAGIC ORDER BY fiscal_year DESC, ticker;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     ticker,
# MAGIC     filing_type,
# MAGIC     accession_number,
# MAGIC     file_path,
# MAGIC     file_size_bytes,
# MAGIC     ingested_at
# MAGIC FROM main.finsage_bronze.filings
# MAGIC ORDER BY ingested_at DESC
# MAGIC LIMIT 10;

# COMMAND ----------

display(spark.sql("DESCRIBE DETAIL main.finsage_bronze.filings").select("sizeInBytes"))
count   = spark.table("main.finsage_bronze.filings").count()
tickers = spark.table("main.finsage_bronze.filings").select("ticker").distinct().count()
print(f"Total rows in Delta  : {count}")
print(f"Total unique companies: {tickers}")

# COMMAND ----------

import requests
import time
from uuid import uuid4
from pyspark.sql import Row
from pyspark.sql.functions import current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

USER_AGENT = "Digvijay Singh (digvijay@arsaga.jp)"
HEADERS    = {"User-Agent": USER_AGENT}

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "JPM",  "GS",   "BAC",   "V",    "MA",
    "JNJ",  "PFE",  "UNH",   "ABBV", "MRK",
    "WMT",  "KO",   "NKE",   "MCD",  "SBUX",
    "TSLA", "F",    "GM",    "RIVN", "LCID",
    "CRM",  "SNOW", "PLTR",  "NET",  "DDOG",
]

session    = requests.Session()
api_rows   = []
error_rows = []
company_map_url = "https://www.sec.gov/files/company_tickers.json"

# --- STEP 0: Skip tickers already fetched today ---
try:
    df_existing = spark.sql("""
        SELECT ticker
        FROM main.finsage_bronze.xbrl_companyfacts_raw
        WHERE to_date(fetched_at) = current_date() AND api_status = 'success'
    """)
    already_fetched_today = [row["ticker"] for row in df_existing.collect()]
except Exception:
    already_fetched_today = []

# --- STEP 1: Build ticker -> CIK map ---
try:
    company_map_resp = session.get(company_map_url, headers=HEADERS, timeout=30)
    company_map_resp.raise_for_status()
    company_map = company_map_resp.json()
except Exception as e:
    print(f"Failed to fetch ticker map: {e}")
    raise

ticker_to_cik = {}
for item in company_map.values():
    ticker = item.get("ticker", "").upper()
    if ticker in TICKERS:
        ticker_to_cik[ticker] = str(item.get("cik_str", "")).zfill(10)

# --- STEP 2: Fetch CompanyFacts JSON ---
print("Starting SEC API extraction...")
for ticker in TICKERS:
    if ticker in already_fetched_today:
        print(f"[{ticker}] Already fetched today. Skipping.")
        continue

    cik = ticker_to_cik.get(ticker)
    if not cik:
        error_rows.append(Row(
            error_id=str(uuid4()), source_system="sec_companyfacts_api",
            source_url=company_map_url, file_path=None,
            error_type="missing_cik",
            error_message="CIK not found in SEC company_tickers.json",
            retry_count=0
        ))
        continue

    source_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        resp = session.get(source_url, headers=HEADERS, timeout=45)
        if resp.status_code == 200:
            payload = resp.json()
            api_rows.append(Row(
                snapshot_id=str(uuid4()), ticker=ticker, cik=cik,
                entity_name=payload.get("entityName"), source_url=source_url,
                raw_json=resp.text, api_status="success",
                http_status_code=resp.status_code, error_message=None
            ))
            print(f"[OK]  {ticker}")
        else:
            api_rows.append(Row(
                snapshot_id=str(uuid4()), ticker=ticker, cik=cik,
                entity_name=None, source_url=source_url, raw_json=None,
                api_status="failed", http_status_code=resp.status_code,
                error_message=f"HTTP {resp.status_code}"
            ))
            error_rows.append(Row(
                error_id=str(uuid4()), source_system="sec_companyfacts_api",
                source_url=source_url, file_path=None, error_type="http_failure",
                error_message=f"HTTP {resp.status_code}", retry_count=0
            ))
            print(f"[ERR] {ticker}: HTTP {resp.status_code}")
    except Exception as e:
        api_rows.append(Row(
            snapshot_id=str(uuid4()), ticker=ticker, cik=cik,
            entity_name=None, source_url=source_url, raw_json=None,
            api_status="failed", http_status_code=None, error_message=str(e)
        ))
        error_rows.append(Row(
            error_id=str(uuid4()), source_system="sec_companyfacts_api",
            source_url=source_url, file_path=None, error_type="request_exception",
            error_message=str(e), retry_count=0
        ))
        print(f"[EXC] {ticker}: {e}")

    time.sleep(0.2)  # respect SEC rate limits

# --- STEP 3: Define schemas ---
api_schema = StructType([
    StructField("snapshot_id",      StringType(),  True),
    StructField("ticker",           StringType(),  True),
    StructField("cik",              StringType(),  True),
    StructField("entity_name",      StringType(),  True),
    StructField("source_url",       StringType(),  True),
    StructField("raw_json",         StringType(),  True),
    StructField("api_status",       StringType(),  True),
    StructField("http_status_code", IntegerType(), True),
    StructField("error_message",    StringType(),  True),
])
error_schema = StructType([
    StructField("error_id",      StringType(),  True),
    StructField("source_system", StringType(),  True),
    StructField("source_url",    StringType(),  True),
    StructField("file_path",     StringType(),  True),
    StructField("error_type",    StringType(),  True),
    StructField("error_message", StringType(),  True),
    StructField("retry_count",   IntegerType(), True),
])

# --- STEP 4: Write to Delta ---
if api_rows:
    (
        spark.createDataFrame(api_rows, schema=api_schema)
        .withColumn("fetched_at", current_timestamp())
        .write.format("delta").mode("append")
        .saveAsTable("main.finsage_bronze.xbrl_companyfacts_raw")
    )
if error_rows:
    (
        spark.createDataFrame(error_rows, schema=error_schema)
        .withColumn("failed_at", current_timestamp())
        .write.format("delta").mode("append")
        .saveAsTable("main.finsage_bronze.ingestion_errors")
    )

print(f"\nBronze API snapshots appended : {len(api_rows)}")
print(f"Bronze API failures logged    : {len(error_rows)}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     snapshot_id,
# MAGIC     ticker,
# MAGIC     cik,
# MAGIC     api_status,
# MAGIC     http_status_code,
# MAGIC     fetched_at,
# MAGIC     SUBSTRING(raw_json, 1, 2500) AS raw_json_preview,
# MAGIC     LENGTH(raw_json)             AS payload_size
# MAGIC FROM main.finsage_bronze.xbrl_companyfacts_raw
# MAGIC ORDER BY fetched_at DESC, ticker
# MAGIC LIMIT 5;

# COMMAND ----------

from pyspark.sql.functions import col

df_peek   = spark.table("main.finsage_bronze.xbrl_companyfacts_raw").filter(col("ticker") == "AAPL")
apple_json = df_peek.select("raw_json").first()[0]
search_word = '"Revenues"'
index = apple_json.find(search_word)

if index != -1:
    print("Found it! Slice of raw JSON containing the data:\n")
    print(apple_json[index: index + 500])
else:
    print("Could not find the exact word.")
