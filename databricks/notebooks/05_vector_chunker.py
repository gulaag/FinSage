# Databricks notebook source
# ==============================================================================
# FinSage | 05 — Vector Chunker
# Token-based chunking of Silver filing sections for RAG ingestion, followed
# by Databricks Vector Search index provisioning.
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

# MAGIC %pip install langchain-text-splitters tiktoken
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ── Token-based chunking with deterministic IDs ──────────────────────────────

import hashlib
import json
import re
import uuid
import pandas as pd
import tiktoken

from pyspark.sql import functions as F
from pyspark.sql import types as T
from delta.tables import DeltaTable

# --- Configuration ---
SOURCE_TABLE          = f"main.finsage_silver.filing_sections"
TARGET_TABLE          = f"main.finsage_gold.filing_section_chunks"
EMBEDDING_MODEL       = "text-embedding-3-large"
CHUNK_TOKENS          = 512
CHUNK_OVERLAP_TOKENS  = 64
CHUNK_VERSION         = f"tok_{CHUNK_TOKENS}_{CHUNK_OVERLAP_TOKENS}_v1"

if CHUNK_TOKENS <= 0:
    raise ValueError("CHUNK_TOKENS must be > 0")
if CHUNK_OVERLAP_TOKENS < 0:
    raise ValueError("CHUNK_OVERLAP_TOKENS must be >= 0")
if CHUNK_OVERLAP_TOKENS >= CHUNK_TOKENS:
    raise ValueError("CHUNK_OVERLAP_TOKENS must be < CHUNK_TOKENS")

# --- Tokenizer helpers ---
_ENCODING = None

def get_encoding():
    global _ENCODING
    if _ENCODING is None:
        try:
            _ENCODING = tiktoken.encoding_for_model(EMBEDDING_MODEL)
        except KeyError:
            _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING

def normalize_text(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[^\S\n\t]+", " ", t)
    t = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", t)
    return t.strip()

def deterministic_chunk_id(
    filing_id: str,
    section_name: str,
    chunk_index: int,
    chunk_text: str,
    chunk_version: str,
) -> str:
    payload = {
        "filing_id":    str(filing_id),
        "section_name": str(section_name),
        "chunk_index":  int(chunk_index),
        "chunk_text":   chunk_text,
        "chunk_version": chunk_version,
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

# --- UDF schema ---
chunk_struct = T.StructType([
    T.StructField("chunk_id",      T.StringType(),  nullable=False),
    T.StructField("chunk_index",   T.IntegerType(), nullable=False),
    T.StructField("chunk_text",    T.StringType(),  nullable=False),
    T.StructField("token_count",   T.IntegerType(), nullable=False),
    T.StructField("word_count",    T.IntegerType(), nullable=False),
    T.StructField("char_count",    T.IntegerType(), nullable=False),
    T.StructField("start_token",   T.IntegerType(), nullable=False),
    T.StructField("end_token",     T.IntegerType(), nullable=False),
    T.StructField("chunk_version", T.StringType(),  nullable=False),
])
chunk_array_schema = T.ArrayType(chunk_struct, containsNull=False)

@F.pandas_udf(chunk_array_schema)
def chunk_sections_udf(
    section_text_col: pd.Series,
    filing_id_col: pd.Series,
    section_name_col: pd.Series,
) -> pd.Series:
    enc  = get_encoding()
    step = CHUNK_TOKENS - CHUNK_OVERLAP_TOKENS
    out  = []

    for text, filing_id, section_name in zip(section_text_col, filing_id_col, section_name_col):
        if text is None:
            out.append([])
            continue
        normalized = normalize_text(str(text))
        if not normalized:
            out.append([])
            continue
        token_ids = enc.encode(normalized)
        if not token_ids:
            out.append([])
            continue

        row_chunks  = []
        chunk_index = 0
        for start in range(0, len(token_ids), step):
            end        = min(start + CHUNK_TOKENS, len(token_ids))
            if start >= end:
                break
            chunk_ids  = token_ids[start:end]
            chunk_text = enc.decode(chunk_ids).strip()
            if not chunk_text:
                if end == len(token_ids):
                    break
                continue
            cid = deterministic_chunk_id(
                filing_id=str(filing_id), section_name=str(section_name),
                chunk_index=chunk_index, chunk_text=chunk_text,
                chunk_version=CHUNK_VERSION,
            )
            row_chunks.append({
                "chunk_id":     cid,
                "chunk_index":  chunk_index,
                "chunk_text":   chunk_text,
                "token_count":  len(chunk_ids),
                "word_count":   len(chunk_text.split()),
                "char_count":   len(chunk_text),
                "start_token":  start,
                "end_token":    end,
                "chunk_version": CHUNK_VERSION,
            })
            chunk_index += 1
            if end == len(token_ids):
                break
        out.append(row_chunks)
    return pd.Series(out)

# --- Build chunk dataframe ---
run_id    = str(uuid.uuid4())
# filing_type is new on filing_sections (added when 10-Q support landed in notebook 03);
# coalesce to '10-K' so backfilled rows that predate the column retain the legacy semantics.
df_source = (
    spark.table(SOURCE_TABLE)
    .withColumn(
        "filing_type",
        F.coalesce(F.col("filing_type"), F.lit("10-K")) if "filing_type" in spark.table(SOURCE_TABLE).columns
        else F.lit("10-K"),
    )
    .select("filing_id", "ticker", "fiscal_year", "filing_type", "section_name", "section_text")
)

df_chunks = (
    df_source
    .withColumn("chunk_array", chunk_sections_udf(
        F.col("section_text"), F.col("filing_id"), F.col("section_name")
    ))
    .withColumn("chunk", F.explode("chunk_array"))
    .select(
        F.col("chunk.chunk_id").alias("chunk_id"),
        F.col("filing_id"),
        F.col("ticker"),
        F.col("fiscal_year"),
        F.col("filing_type"),
        F.col("section_name"),
        F.col("chunk.chunk_index").alias("chunk_index"),
        F.col("chunk.chunk_text").alias("chunk_text"),
        F.col("chunk.token_count").alias("token_count"),
        F.col("chunk.word_count").alias("word_count"),
        F.col("chunk.char_count").alias("char_count"),
        F.col("chunk.start_token").alias("start_token"),
        F.col("chunk.end_token").alias("end_token"),
        F.col("chunk.chunk_version").alias("chunk_version"),
        F.lit(CHUNK_TOKENS).alias("config_chunk_tokens"),
        F.lit(CHUNK_OVERLAP_TOKENS).alias("config_overlap_tokens"),
        F.current_timestamp().alias("chunked_at"),
        F.lit(run_id).alias("run_id"),
    )
)
df_chunks = df_chunks.withColumn("content_hash", F.sha2(F.col("chunk_text"), 256))
# --- Data quality guards ---
dup_count = (
    df_chunks.groupBy("chunk_id").count()
    .filter(F.col("count") > 1).limit(1).count()
)
if dup_count > 0:
    raise RuntimeError("Duplicate chunk_id detected. Aborting write.")

bad_rows = (
    df_chunks.filter(
        F.col("chunk_text").isNull() |
        (F.col("token_count") <= 0) |
        (F.col("chunk_index") < 0)
    ).limit(1).count()
)
if bad_rows > 0:
    raise RuntimeError("Invalid chunk rows detected. Aborting write.")

# --- Idempotent write ---
# autoMerge ensures the new filing_type column lands on the existing table
spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")

if spark.catalog.tableExists(TARGET_TABLE):
    target = DeltaTable.forName(spark, TARGET_TABLE)
    (
        target.alias("t")
        .merge(df_chunks.alias("s"), "t.chunk_id = s.chunk_id")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
else:
    df_chunks.write.format("delta").mode("overwrite").saveAsTable(TARGET_TABLE)

display(
    df_chunks.select(
        "ticker", "section_name", "chunk_index",
        "token_count", "word_count", "chunk_text",
    ).limit(15)
)

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE main.finsage_gold.filing_section_chunks
# MAGIC SET TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import logging
import random
import time
from typing import Any, Dict, Optional, Tuple

from databricks.vector_search.client import VectorSearchClient

VECTOR_SEARCH_ENDPOINT_NAME  = "finsage_vs_endpoint"
SOURCE_TABLE                 = "main.finsage_gold.filing_section_chunks"
INDEX_NAME                   = "main.finsage_gold.filing_chunks_index"
PRIMARY_KEY                  = "chunk_id"
EMBEDDING_COLUMN             = "chunk_text"
EMBEDDING_MODEL_ENDPOINT     = "databricks-bge-large-en"
PIPELINE_TYPE                = "TRIGGERED"

ENDPOINT_READY_TIMEOUT_SEC = 30 * 60
INDEX_READY_TIMEOUT_SEC    = 90 * 60
POLL_SEC                   = 15

log = logging.getLogger("vector-index-setup")
if not log.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s - %(message)s")

def _retryable_call(fn, retries: int = 8, base_sleep: float = 1.5, max_sleep: float = 20.0):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt == retries:
                break
            backoff = min(max_sleep, base_sleep * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
            log.warning("Call failed (attempt %s/%s): %s. Retrying in %.1fs...", attempt, retries, e, backoff)
            time.sleep(backoff)
    raise last_exc

def _nested_get(d: Dict[str, Any], *paths: Tuple[str, ...]) -> Optional[Any]:
    for path in paths:
        cur, ok = d, True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            return cur
    return None

def _normalize_state(x: Any) -> str:
    return "UNKNOWN" if x is None else str(x).strip().upper()

def _assert_source_table_quality(table: str, pk: str, text_col: str):
    cols    = {f.name for f in spark.table(table).schema.fields}
    missing = [c for c in [pk, text_col] if c not in cols]
    if missing:
        raise ValueError(f"Source table {table} missing required columns: {missing}")

    counts = spark.sql(f"""
        SELECT
            COUNT(*) AS total_rows,
            SUM(CASE WHEN {pk} IS NULL THEN 1 ELSE 0 END) AS null_pk,
            SUM(CASE WHEN {text_col} IS NULL OR TRIM({text_col}) = '' THEN 1 ELSE 0 END) AS bad_text
        FROM {table}
    """).collect()[0].asDict()

    dup = spark.sql(f"""
        SELECT COUNT(*) AS dup_pk_rows
        FROM (SELECT {pk} FROM {table} GROUP BY {pk} HAVING COUNT(*) > 1)
    """).collect()[0]["dup_pk_rows"]

    log.info("Preflight table stats: %s", counts)
    log.info("Duplicate PK groups  : %s", dup)
    if counts["null_pk"] > 0:
        raise ValueError(f"{table}.{pk} contains NULLs ({counts['null_pk']}).")
    if dup > 0:
        raise ValueError(f"{table}.{pk} contains duplicates ({dup} duplicate keys).")
    if counts["bad_text"] > 0:
        log.warning("%s.%s has %s null/empty rows; these rows may fail embedding.",
                    table, text_col, counts["bad_text"])
    try:
        cdf = spark.sql(f"SHOW TBLPROPERTIES {table} ('delta.enableChangeDataFeed')").collect()[0]["value"]
        log.info("delta.enableChangeDataFeed=%s", cdf)
    except Exception as e:
        log.warning("Could not read CDF property: %s", e)

def ensure_endpoint(vsc: VectorSearchClient, endpoint_name: str):
    eps   = _retryable_call(lambda: vsc.list_endpoints()).get("endpoints", [])
    names = {e.get("name") for e in eps}
    if endpoint_name not in names:
        log.info("Creating endpoint: %s", endpoint_name)
        _retryable_call(lambda: vsc.create_endpoint(name=endpoint_name, endpoint_type="STANDARD"))
    else:
        log.info("Endpoint exists: %s", endpoint_name)

def wait_for_endpoint_online(vsc: VectorSearchClient, endpoint_name: str, timeout_sec: int):
    start = time.time()
    while True:
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"Timeout waiting for endpoint {endpoint_name} to become ONLINE")
        ep    = _retryable_call(lambda: vsc.get_endpoint(name=endpoint_name))
        state = _normalize_state(_nested_get(ep, ("endpoint_status", "state"), ("status", "state")))
        msg   = _nested_get(ep, ("endpoint_status", "message"), ("status", "message"))
        log.info("Endpoint state=%s message=%s", state, msg)
        if state == "ONLINE":
            return
        if state in {"FAILED", "ERROR"}:
            raise RuntimeError(f"Endpoint {endpoint_name} entered terminal state={state}, message={msg}")
        time.sleep(POLL_SEC)

def ensure_index(vsc, endpoint_name, index_name, source_table,
                 primary_key, embedding_source_column,
                 embedding_model_endpoint_name, pipeline_type):
    idxs  = _retryable_call(lambda: vsc.list_indexes(endpoint_name)).get("vector_indexes", [])
    names = {i.get("name") for i in idxs}
    if index_name not in names:
        log.info("Creating index: %s", index_name)
        _retryable_call(lambda: vsc.create_delta_sync_index(
            endpoint_name=endpoint_name,
            source_table_name=source_table,
            index_name=index_name,
            pipeline_type=pipeline_type,
            primary_key=primary_key,
            embedding_source_column=embedding_source_column,
            embedding_model_endpoint_name=embedding_model_endpoint_name,
        ))
    else:
        log.info("Index exists: %s", index_name)

def wait_for_index_ready(vsc, endpoint_name, index_name, timeout_sec):
    start = time.time()
    while True:
        if time.time() - start > timeout_sec:
            idx  = _retryable_call(lambda: vsc.get_index(endpoint_name=endpoint_name, index_name=index_name))
            desc = _retryable_call(lambda: idx.describe())
            raise TimeoutError(
                f"Timeout waiting for index {index_name}.\n"
                f"Latest describe:\n{json.dumps(desc, indent=2, default=str)}"
            )
        idx   = _retryable_call(lambda: vsc.get_index(endpoint_name=endpoint_name, index_name=index_name))
        desc  = _retryable_call(lambda: idx.describe())
        state = _normalize_state(_nested_get(
            desc,
            ("status", "state"), ("index_status", "state"),
            ("index_status", "detailed_state"), ("state",),
        ))
        msg          = _nested_get(desc, ("status", "message"), ("index_status", "message"), ("message",))
        indexed_rows = _nested_get(desc, ("status", "indexed_row_count"), ("indexed_row_count",))
        total_rows   = _nested_get(desc, ("status", "total_row_count"),   ("total_row_count",))
        log.info("Index state=%s indexed=%s total=%s message=%s", state, indexed_rows, total_rows, msg)
        if state in {"ONLINE", "READY", "ONLINE_NO_PENDING_UPDATE"} or state.startswith("ONLINE") or (msg and "succeeded" in msg.lower()):
            log.info("Success detected via state or message. Proceeding...")
            return desc
        if state in {"FAILED", "ERROR", "UNHEALTHY"}:
            raise RuntimeError(
                f"Index {index_name} entered terminal state={state}, message={msg}\n"
                f"Describe:\n{json.dumps(desc, indent=2, default=str)}"
            )
        time.sleep(POLL_SEC)

def trigger_sync_if_needed(vsc, endpoint_name, index_name, pipeline_type,
                           sync_timeout_sec: int = 30):
    if pipeline_type.upper() != "TRIGGERED":
        log.info("Pipeline type is %s; explicit sync not required.", pipeline_type)
        return
    idx = _retryable_call(lambda: vsc.get_index(endpoint_name=endpoint_name, index_name=index_name))
    if not hasattr(idx, "sync"):
        log.warning("Index object has no sync() method in this SDK version; skip explicit trigger.")
        return

    import threading
    sync_exc: list = []

    def _do_sync():
        try:
            idx.sync()
        except Exception as e:
            sync_exc.append(e)

    log.info("Triggering index sync for TRIGGERED pipeline (non-blocking, timeout=%ss).", sync_timeout_sec)
    t = threading.Thread(target=_do_sync, daemon=True)
    t.start()
    t.join(timeout=sync_timeout_sec)

    if t.is_alive():
        # idx.sync() is still blocking — SDK did not receive a completion signal.
        # Verify directly via the index state; if ONLINE the sync already finished.
        log.warning("idx.sync() did not return within %ss. Verifying index state directly.", sync_timeout_sec)
        desc = _retryable_call(lambda: vsc.get_index(endpoint_name=endpoint_name, index_name=index_name))
        state = _normalize_state(_nested_get(desc, ("status", "state"), ("state",)))
        if state.startswith("ONLINE") or state == "READY":
            log.info("Index confirmed %s via direct poll — sync complete. Ignoring SDK hang.", state)
        else:
            log.warning("Index state=%s after sync timeout. Proceeding anyway — monitor the index.", state)
    elif sync_exc:
        log.warning("idx.sync() raised: %s. Index may still sync in background.", sync_exc[0])
    else:
        log.info("idx.sync() returned cleanly.")

def run_vector_index_setup():
    vsc = VectorSearchClient(disable_notice=True)
    _assert_source_table_quality(SOURCE_TABLE, PRIMARY_KEY, EMBEDDING_COLUMN)
    ensure_endpoint(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
    wait_for_endpoint_online(vsc, VECTOR_SEARCH_ENDPOINT_NAME, ENDPOINT_READY_TIMEOUT_SEC)
    ensure_index(
        vsc=vsc, endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=INDEX_NAME,
        source_table=SOURCE_TABLE, primary_key=PRIMARY_KEY,
        embedding_source_column=EMBEDDING_COLUMN,
        embedding_model_endpoint_name=EMBEDDING_MODEL_ENDPOINT,
        pipeline_type=PIPELINE_TYPE,
    )
    wait_for_index_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, INDEX_NAME, INDEX_READY_TIMEOUT_SEC)
    trigger_sync_if_needed(vsc, VECTOR_SEARCH_ENDPOINT_NAME, INDEX_NAME, PIPELINE_TYPE)
    log.info("Vector indexing setup complete. Endpoint=%s | Index=%s",
             VECTOR_SEARCH_ENDPOINT_NAME, INDEX_NAME)

run_vector_index_setup()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc        = VectorSearchClient(disable_notice=True)
INDEX_NAME = "main.finsage_gold.filing_chunks_index"

def search_financial_filings(query: str, num_results: int = 3):
    print(f"Searching for: '{query}'")
    try:
        index   = vsc.get_index(endpoint_name="finsage_vs_endpoint", index_name=INDEX_NAME)
        results = index.similarity_search(
            query_text=query,
            columns_to_return=["ticker", "fiscal_year", "section_name", "chunk_text"],
            num_results=num_results,
        )
        docs = results.get("result", {}).get("data_array", [])
        if not docs:
            return "No results found."
        return "\n---\n".join(
            f"[{d[0]} | {d[1]} | {d[2]}]\n{d[3]}" for d in docs
        )
    except Exception as e:
        return f"Search failed: {str(e)}"

print(search_financial_filings("What did Apple say about supply chain or manufacturing risks?"))

# COMMAND ----------

# Commented: describe the index in full JSON for debugging
# import json
# idx  = vsc.get_index(endpoint_name="finsage_vs_endpoint", index_name="main.finsage_gold.filing_chunks_index")
# desc = idx.describe()
# print(json.dumps(desc, indent=2, default=str))

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc        = VectorSearchClient(disable_notice=True)
INDEX_NAME = "main.finsage_gold.filing_chunks_index"

def search_financial_filings(query: str, num_results: int = 3):
    print(f"Searching for: '{query}'")
    index   = vsc.get_index(endpoint_name="finsage_vs_endpoint", index_name=INDEX_NAME)
    results = index.similarity_search(
        query_text=query,
        columns=["ticker", "fiscal_year", "section_name", "chunk_text"],
        num_results=num_results,
    )
    docs = results.get("result", {}).get("data_array", [])
    if not docs:
        return "No results found."
    return "\n---\n".join(
        f"[{d[0]} | {d[1]} | {d[2]}]\n{d[3]}" for d in docs
    )

print(search_financial_filings("What did Apple say about supply chain or manufacturing risks?"))
