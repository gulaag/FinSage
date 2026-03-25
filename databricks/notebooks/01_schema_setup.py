# Databricks notebook source
# ==============================================================================
# FinSage | 01 — Schema & Volume Setup
# Idempotent DDL: creates the 3-layer Medallion schemas, the download-log table,
# and the raw-filings volume.  Safe to re-run at any time.
# ==============================================================================

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create the 3-layer Medallion structure inside 'main'
# MAGIC CREATE SCHEMA IF NOT EXISTS main.finsage_bronze; -- raw data & ingestion logs
# MAGIC CREATE SCHEMA IF NOT EXISTS main.finsage_silver; -- cleaned, parsed sections
# MAGIC CREATE SCHEMA IF NOT EXISTS main.finsage_gold;   -- final metrics & indicators

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS main.finsage_bronze.sec_filings_download_log (
# MAGIC     ticker            STRING,
# MAGIC     form_type         STRING,
# MAGIC     last_successful_run DATE,
# MAGIC     status            STRING,
# MAGIC     retry_count       INT,
# MAGIC     error_message     STRING,
# MAGIC     updated_at        TIMESTAMP
# MAGIC )
# MAGIC USING DELTA;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS main.finsage_bronze.raw_filings;

# COMMAND ----------

# MAGIC %pip install sec-edgar-downloader

# COMMAND ----------

# Grants — uncomment and adjust the principal before running in a shared workspace.
# %sql
# GRANT USE CATALOG ON CATALOG main TO `digvijay@arsaga.jp`;
# GRANT USE SCHEMA ON SCHEMA main.finsage_bronze TO `digvijay@arsaga.jp`;
# GRANT READ VOLUME, WRITE VOLUME ON VOLUME main.finsage_bronze.raw_filings TO `digvijay@arsaga.jp`;

# COMMAND ----------

import os
import time
import sys
from io import StringIO
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

from sec_edgar_downloader import Downloader
from pyspark.sql.functions import col, current_timestamp, expr
from delta.tables import DeltaTable

# --- CONFIGURATION ---
VOLUME_PATH = "/Volumes/main/finsage_bronze/raw_filings"
USER_AGENT = "Arsaga Partners digvijay@arsaga.jp"
LOG_TABLE = "main.finsage_bronze.sec_filings_download_log"
MAX_RETRIES = 3
MAX_CONCURRENT_WORKERS = 3  # Safe limit for SEC's 10 req/sec rate limit

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "GS", "BAC", "V", "MA",
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "WMT", "KO", "NKE", "MCD", "SBUX",
    "TSLA", "F", "GM", "RIVN", "LCID", "CRM", "SNOW", "PLTR", "NET", "DDOG"
]
FORM_TYPES = ["10-K", "10-Q"]

# --- 1. PRE-FLIGHT IDEMPOTENCY CHECK ---
# Fetch completed (ticker, form_type) pairs for today outside the threads to
# avoid Spark-session thread-safety issues.
today = date.today()
try:
    df_completed = spark.table(LOG_TABLE).filter(
        (col("status") == "SUCCESS") & (col("last_successful_run") == today)
    )
    completed_tasks = set([(row.ticker, row.form_type) for row in df_completed.collect()])
except Exception:
    completed_tasks = set()  # Table may be empty on first run

print(f"Found {len(completed_tasks)} tasks already completed today. Skipping them.")

# --- 2. THREAD-SAFE WORKER ---
def download_filing(ticker, form_type):
    if (ticker, form_type) in completed_tasks:
        return (ticker, form_type, "SKIPPED", 0, "")

    dl = Downloader("FinSage", USER_AGENT, VOLUME_PATH)
    success = False
    retries = 0
    error_msg = ""

    while not success and retries < MAX_RETRIES:
        old_stdout = sys.stdout
        sys.stdout = my_stdout = StringIO()
        try:
            dl.get(form_type, ticker, after="2020-01-01")
            output = my_stdout.getvalue()
            if "Error occurred while downloading" in output or "503" in output:
                raise Exception("SEC API Error/503 Detected.")
            success = True
        except Exception as e:
            retries += 1
            error_msg = str(e)
            time.sleep(10 * retries)  # exponential backoff: 10s, 20s, 30s
        finally:
            sys.stdout = old_stdout

    status = "SUCCESS" if success else "FAILED"
    print(f"[{status}] {ticker} {form_type} (Retries: {retries})")
    return (ticker, form_type, status, retries, error_msg)

# --- 3. PARALLEL EXECUTION ---
os.makedirs(VOLUME_PATH, exist_ok=True)
results = []

print("Starting parallel ingestion pipeline...")
with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
    futures = [executor.submit(download_filing, t, f) for t in TICKERS for f in FORM_TYPES]
    for future in as_completed(futures):
        results.append(future.result())

# --- 4. ATOMIC STATE UPDATE ---
processed_results = [r for r in results if r[2] != "SKIPPED"]

if processed_results:
    print("Syncing ingestion state to Delta Lake...")
    schema = "ticker STRING, form_type STRING, status STRING, retry_count INT, error_message STRING"
    df_updates = spark.createDataFrame(processed_results, schema=schema)

    df_updates = df_updates.withColumn(
        "last_successful_run",
        expr("IF(status = 'SUCCESS', current_date(), cast(null as date))")
    ).withColumn("updated_at", current_timestamp())

    dt_log = DeltaTable.forName(spark, LOG_TABLE)
    dt_log.alias("t").merge(
        df_updates.alias("s"),
        "t.ticker = s.ticker AND t.form_type = s.form_type"
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

print("\nINGESTION PIPELINE COMPLETE.")
