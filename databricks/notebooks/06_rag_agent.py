# Databricks notebook source
# ==============================================================================
# FinSage | 06 — RAG Agent
#
# Builds a function-calling financial Q&A agent with two tools:
#   1. search_filings      — semantic search over filing_chunks_index (Vector Search)
#   2. get_company_metrics — structured lookup over company_metrics (Gold table)
#
# The agent is:
#   - Traced end-to-end with MLflow (every tool call + LLM call gets a span)
#   - Registered as a Unity Catalog model (main.finsage_gold.finsage_rag_agent)
#   - Deployed to Databricks Model Serving (finsage_agent_endpoint)
#
# LLM: databricks-meta-llama-3-3-70b-instruct (function-calling capable, READY)
# Framework: mlflow.pyfunc — no LangChain dependency
# ==============================================================================

# COMMAND ----------

# ── 1. Runtime Parameters ─────────────────────────────────────────────────────
dbutils.widgets.text("catalog",              "main",                                    "UC catalog")
dbutils.widgets.text("env",                  "dev",                                     "Environment")
dbutils.widgets.text("llm_endpoint",         "databricks-meta-llama-3-3-70b-instruct",  "LLM serving endpoint")
dbutils.widgets.text("vs_endpoint",          "finsage_vs_endpoint",                     "Vector Search endpoint")
dbutils.widgets.text("num_results",          "5",                                       "Top-k retrieval results")
dbutils.widgets.text("similarity_threshold", "0.6",                                     "Min similarity score (0-1)")

CATALOG              = dbutils.widgets.get("catalog")
ENV                  = dbutils.widgets.get("env")
LLM_ENDPOINT         = dbutils.widgets.get("llm_endpoint")
VS_ENDPOINT          = dbutils.widgets.get("vs_endpoint")
NUM_RESULTS          = int(dbutils.widgets.get("num_results"))
SIMILARITY_THRESHOLD = float(dbutils.widgets.get("similarity_threshold"))

VS_INDEX_NAME           = f"{CATALOG}.finsage_gold.filing_chunks_index"
METRICS_TABLE           = f"{CATALOG}.finsage_gold.company_metrics"
METRICS_QUARTERLY_TABLE = f"{CATALOG}.finsage_gold.company_metrics_quarterly"
SILVER_FINANCIALS_TABLE = f"{CATALOG}.finsage_silver.financial_statements"
UC_MODEL_NAME           = f"{CATALOG}.finsage_gold.finsage_rag_agent"
AGENT_ENDPOINT          = "finsage_agent_endpoint"
MAX_ITERATIONS          = 5

print(f"[CONFIG] catalog={CATALOG} | env={ENV} | llm={LLM_ENDPOINT} | vs_index={VS_INDEX_NAME}")

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch mlflow databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ── 2. Imports + re-declare constants (required after restartPython wipes state) ──
import json
import logging
import mlflow
import mlflow.deployments
import re
import time
import requests
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
log = logging.getLogger("finsage-agent")

mlflow.set_registry_uri("databricks-uc")

# Re-read widgets — restartPython() clears all Python variables but widgets persist
CATALOG              = dbutils.widgets.get("catalog")
ENV                  = dbutils.widgets.get("env")
LLM_ENDPOINT         = dbutils.widgets.get("llm_endpoint")
VS_ENDPOINT          = dbutils.widgets.get("vs_endpoint")
NUM_RESULTS          = int(dbutils.widgets.get("num_results"))
SIMILARITY_THRESHOLD = float(dbutils.widgets.get("similarity_threshold"))

VS_INDEX_NAME           = f"{CATALOG}.finsage_gold.filing_chunks_index"
METRICS_TABLE           = f"{CATALOG}.finsage_gold.company_metrics"
METRICS_QUARTERLY_TABLE = f"{CATALOG}.finsage_gold.company_metrics_quarterly"
SILVER_FINANCIALS_TABLE = f"{CATALOG}.finsage_silver.financial_statements"
UC_MODEL_NAME           = f"{CATALOG}.finsage_gold.finsage_rag_agent"
AGENT_ENDPOINT          = "finsage_agent_endpoint"
MAX_ITERATIONS          = 5

print(f"[CONFIG restored] catalog={CATALOG} | llm={LLM_ENDPOINT} | metrics={METRICS_TABLE}")

# COMMAND ----------

# ── 3. Pre-load Gold metrics into memory ──────────────────────────────────────
# company_metrics has only 180 rows — load once as a nested dict for zero-latency
# lookup inside the pyfunc serving container (no SQL warehouse needed at runtime).
#
# Defensive re-init — idempotent if cell 2 already ran in this kernel, required
# if the user clicked "Run cell" on this cell directly after restartPython
# wiped state. Same pattern repeated in every later cell that uses module-level
# state.
import json, logging, re, time
import requests
log = logging.getLogger("finsage-agent")
CATALOG                 = dbutils.widgets.get("catalog")
METRICS_TABLE           = f"{CATALOG}.finsage_gold.company_metrics"
METRICS_QUARTERLY_TABLE = f"{CATALOG}.finsage_gold.company_metrics_quarterly"
SILVER_FINANCIALS_TABLE = f"{CATALOG}.finsage_silver.financial_statements"

def _load_metrics_cache(table: str) -> dict:
    df = spark.table(table).select(
        "ticker", "company_name", "fiscal_year",
        "revenue", "net_income", "gross_profit", "operating_income",
        "operating_cash_flow", "total_assets", "total_liabilities",
        "total_equity", "total_debt", "rd_expense", "gross_margin_pct",
        "revenue_yoy_growth_pct", "debt_to_equity", "data_quality_score",
    )
    cache = {}
    for row in df.collect():
        r = row.asDict()
        ticker = r.pop("ticker")
        fy = r.pop("fiscal_year")
        cache.setdefault(ticker.upper(), {})[fy] = r
    log.info("Metrics cache loaded: %d tickers", len(cache))
    return cache

def _load_quarterly_cache(table: str) -> dict:
    """Cache shape: {ticker: {(fiscal_year, fiscal_quarter): {...metrics}}}.
    Returns empty dict if the table doesn't exist yet — lets the agent degrade
    gracefully before notebook 04b has run for the first time.
    """
    if not spark.catalog.tableExists(table):
        log.warning("Quarterly metrics table %s does not exist — skipping cache load.", table)
        return {}
    df = spark.table(table).select(
        "ticker", "company_name", "fiscal_year", "fiscal_quarter", "period_end_date",
        "revenue", "net_income", "gross_profit", "operating_income",
        "operating_cash_flow", "total_assets", "total_liabilities",
        "total_equity", "total_debt", "rd_expense", "gross_margin_pct",
        "revenue_yoy_growth_pct", "debt_to_equity", "data_quality_score",
    )
    cache = {}
    for row in df.collect():
        r = row.asDict()
        ticker = r.pop("ticker")
        fy = r.pop("fiscal_year")
        fq = r.pop("fiscal_quarter")
        if r.get("period_end_date") is not None:
            r["period_end_date"] = str(r["period_end_date"])  # serialize for JSON artifact
        cache.setdefault(ticker.upper(), {})[f"{int(fy)}-Q{int(fq)}"] = r
    log.info("Quarterly cache loaded: %d tickers", len(cache))
    return cache


def _load_filing_metadata_cache(silver_table: str) -> dict:
    """
    Cache shape:
      {ticker: {fy: {"filing_date": "YYYY-MM-DD", "employees": float|None, "shares_outstanding": float|None}}}
    Sourced independently from silver financial_statements (10-K FY facts).
    """
    ticker_cik_map = {
        "AAPL": "0000320193", "ABBV": "0001551152", "AMZN": "0001018724", "BAC": "0000070858",
        "CRM": "0001108524", "DDOG": "0001561550", "F": "0000037996", "GM": "0001467858",
        "GOOGL": "0001652044", "GS": "0000886982", "JNJ": "0000200406", "JPM": "0000019617",
        "KO": "0000021344", "LCID": "0001811210", "MA": "0001141391", "MCD": "0000063908",
        "MRK": "0000310158", "MSFT": "0000789019", "NET": "0001477333", "NKE": "0000320187",
        "NVDA": "0001045810", "PFE": "0000078003", "PLTR": "0001321655", "RIVN": "0001874178",
        "SBUX": "0000829224", "SNOW": "0001640147", "TSLA": "0001318605", "UNH": "0000731766",
        "V": "0001403161", "WMT": "0000104169",
    }

    # Determine which fiscal years are relevant per ticker from annual metrics cache.
    ticker_years = {
        t: sorted(int(y) for y in years.keys())
        for t, years in METRICS_CACHE.items()
        if t in ticker_cik_map
    }
    if not ticker_years:
        return {}

    sec_session = requests.Session()
    sec_session.headers.update({
        "User-Agent": "FinSage Agent Metadata digvijay@arsaga.jp",
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    })
    min_interval_sec = 0.12
    last_call_ts = 0.0

    def _throttle():
        nonlocal last_call_ts
        elapsed = time.monotonic() - last_call_ts
        if elapsed < min_interval_sec:
            time.sleep(min_interval_sec - elapsed)

    def _get_json(url: str):
        nonlocal last_call_ts
        for attempt in range(3):
            _throttle()
            resp = sec_session.get(url, timeout=30)
            last_call_ts = time.monotonic()
            if resp.status_code == 404:
                return None
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in {429, 500, 502, 503, 504, 403} and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            resp.raise_for_status()
        return None

    def _best_fact_by_fy(payload: dict, preferred_unit: str, fy: int):
        units = payload.get("units", {}) if payload else {}
        candidate_units = [preferred_unit] if preferred_unit in units else list(units.keys())
        best = None
        for unit in candidate_units:
            for fact in units.get(unit, []):
                if fact.get("fy") != fy:
                    continue
                if fact.get("form") not in {"10-K", "10-K/A"}:
                    continue
                if best is None or (fact.get("filed") or "") > (best.get("filed") or ""):
                    best = fact
        return best

    cache = {}
    for ticker, years in ticker_years.items():
        cik = ticker_cik_map[ticker]
        emp_payload = _get_json(
            f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/dei/EntityNumberOfEmployees.json"
        )
        shares_payload = _get_json(
            f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/dei/EntityCommonStockSharesOutstanding.json"
        )
        submissions = _get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")

        filing_date_by_report_year = {}
        if submissions:
            recent = submissions.get("filings", {}).get("recent", {})
            for form, report_date, filing_date in zip(
                recent.get("form", []),
                recent.get("reportDate", []),
                recent.get("filingDate", []),
            ):
                if form not in {"10-K", "10-K/A"} or not report_date or not filing_date:
                    continue
                fy = int(str(report_date)[:4])
                prev = filing_date_by_report_year.get(fy)
                if prev is None or filing_date > prev:
                    filing_date_by_report_year[fy] = filing_date

        for fy in years:
            emp_fact = _best_fact_by_fy(emp_payload, preferred_unit="pure", fy=fy)
            shares_fact = _best_fact_by_fy(shares_payload, preferred_unit="shares", fy=fy)
            cache.setdefault(ticker, {})[fy] = {
                "filing_date": filing_date_by_report_year.get(fy),
                "employees": float(emp_fact.get("val")) if emp_fact and emp_fact.get("val") is not None else None,
                "shares_outstanding": float(shares_fact.get("val")) if shares_fact and shares_fact.get("val") is not None else None,
            }

    log.info("Metadata cache loaded from SEC: %d tickers", len(cache))
    return cache

METRICS_CACHE           = _load_metrics_cache(METRICS_TABLE)
QUARTERLY_METRICS_CACHE = _load_quarterly_cache(METRICS_QUARTERLY_TABLE)
FILING_METADATA_CACHE   = _load_filing_metadata_cache(SILVER_FINANCIALS_TABLE)
print(
    f"[CACHE] annual: {len(METRICS_CACHE)} tickers | quarterly: {len(QUARTERLY_METRICS_CACHE)} tickers | "
    f"metadata: {len(FILING_METADATA_CACHE)} tickers"
)

# Fail-fast assertions — if any cache is empty, the agent can't answer that
# entire question class. Better to crash here than to ship a half-broken model.
assert len(METRICS_CACHE) >= 30, (
    f"Annual metrics cache has only {len(METRICS_CACHE)} tickers (expected ≥30). "
    f"Re-run notebook 04 before logging the agent."
)
assert len(QUARTERLY_METRICS_CACHE) >= 30, (
    f"Quarterly metrics cache has only {len(QUARTERLY_METRICS_CACHE)} tickers (expected ≥30). "
    f"Re-run notebook 04b before logging the agent."
)
# Filing-metadata cache is allowed to be partial (SEC API can throttle), but
# warn loudly if substantially incomplete.
if len(FILING_METADATA_CACHE) < 25:
    log.warning(
        "Filing-metadata cache has only %d tickers — get_filing_metadata will "
        "return 'No metadata' for the missing tickers. SEC EDGAR may have "
        "throttled this run; consider re-running cell 3 if you want full coverage.",
        len(FILING_METADATA_CACHE),
    )

# COMMAND ----------

# ── 4. Tool: search_filings ───────────────────────────────────────────────────

# Defensive re-init — idempotent if cell 2 already ran. Required when this
# cell is run standalone after restartPython() (cell 2) wiped Python globals.
import json, logging, re
import mlflow
from databricks.vector_search.client import VectorSearchClient
log = logging.getLogger("finsage-agent")
CATALOG              = dbutils.widgets.get("catalog")
VS_ENDPOINT          = dbutils.widgets.get("vs_endpoint")
VS_INDEX_NAME        = f"{CATALOG}.finsage_gold.filing_chunks_index"
NUM_RESULTS          = int(dbutils.widgets.get("num_results"))
SIMILARITY_THRESHOLD = float(dbutils.widgets.get("similarity_threshold"))

# Section names valid across both filing types. Business and Risk Factors are
# 10-K-only; MD&A appears in both 10-K and 10-Q; Risk Factors Updates is 10-Q
# Part II Item 1A (only present when the filer actually updates risks mid-year).
VALID_SECTION_NAMES = ("Business", "Risk Factors", "MD&A", "Risk Factors Updates")
VALID_FILING_TYPES  = ("10-K", "10-Q")

# Probe the live VS index once at import and remember which metadata columns
# actually exist. The source chunks table was extended with `filing_type` when
# 10-Q support landed, but a DELTA_SYNC VS index only picks up new columns
# after a full re-provision. If the pipeline is still serving the pre-10-Q
# schema (or is stuck behind a stale CDF cursor), requesting `filing_type` in
# `columns=` or `filters=` will 400. Adapt at runtime so the tool stays
# functional on either schema.
_VS_CANDIDATE_COLS = ("ticker", "fiscal_year", "filing_type", "section_name", "chunk_text")

def _detect_index_columns():
    try:
        vsc = VectorSearchClient(disable_notice=True)
        idx = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX_NAME)
        desc = idx.describe() or {}
        schema_spec = (
            desc.get("delta_sync_index_spec", {}).get("schema_json")
            or desc.get("schema_json")
        )
        if schema_spec:
            import json as _json
            fields = _json.loads(schema_spec).get("fields", [])
            return {f["name"] for f in fields}
    except Exception as e:
        log.warning("Could not introspect VS index schema (%s); assuming legacy columns.", e)
    # Legacy fallback: only the original 4-column shape
    return {"chunk_id", "ticker", "fiscal_year", "section_name", "chunk_text"}

_VS_INDEX_COLS       = _detect_index_columns()
_FILING_TYPE_IN_INDEX = "filing_type" in _VS_INDEX_COLS
log.info("VS index columns present: %s (filing_type supported=%s)",
         sorted(_VS_INDEX_COLS), _FILING_TYPE_IN_INDEX)


@mlflow.trace(name="search_filings", span_type="RETRIEVER")
def search_filings(
    query: str,
    ticker: str = None,
    section_name: str = None,
    fiscal_year: int = None,
    filing_type: str = None,
    num_results: int = NUM_RESULTS,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> str:
    """
    Semantic search over SEC 10-K and 10-Q filing sections.
    Returns relevant passages with source metadata.

    10-K sections: Business, Risk Factors, MD&A.
    10-Q sections: MD&A, Risk Factors Updates (when the filer includes them).
    Pass filing_type='10-K' or '10-Q' to scope results to annual vs interim
    (only applied if the live VS index has indexed the `filing_type` column).
    """
    vsc = VectorSearchClient(disable_notice=True)
    index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX_NAME)

    filters = {}
    if ticker:
        filters["ticker"] = ticker.upper()
    if section_name and section_name in VALID_SECTION_NAMES:
        filters["section_name"] = section_name
    if fiscal_year:
        filters["fiscal_year"] = fiscal_year
    if filing_type and filing_type in VALID_FILING_TYPES:
        if _FILING_TYPE_IN_INDEX:
            filters["filing_type"] = filing_type
        else:
            log.warning(
                "filing_type='%s' requested but VS index has not indexed that column yet; "
                "ignoring filter.", filing_type
            )

    # Only request columns that are actually materialized in the index.
    columns_to_fetch = [c for c in _VS_CANDIDATE_COLS if c in _VS_INDEX_COLS]

    try:
        results = index.similarity_search(
            query_text=query,
            columns=columns_to_fetch,
            filters=filters if filters else None,
            num_results=num_results,
            query_type="ANN",
        )
    except Exception as e:
        log.warning("Vector search failed: %s", e)
        return f"Search failed: {str(e)}"

    data = results.get("result", {}).get("data_array", [])
    if not data:
        return "No relevant passages found for this query."

    # Map column name → positional index so we read each row robustly whether
    # `filing_type` is present or not.
    col_pos = {c: i for i, c in enumerate(columns_to_fetch)}
    passages = []
    for row in data:
        score = row[len(columns_to_fetch)] if len(row) > len(columns_to_fetch) else None
        if score is not None and score < similarity_threshold:
            continue
        ticker_val = row[col_pos["ticker"]]
        fy         = row[col_pos["fiscal_year"]]
        section    = row[col_pos["section_name"]]
        text       = row[col_pos["chunk_text"]]
        f_type     = row[col_pos["filing_type"]] if "filing_type" in col_pos else None
        src_parts  = [ticker_val, f"FY{int(fy)}"]
        if f_type:
            src_parts.append(f_type)
        src_parts.append(section)
        passages.append(f"[Source: {' | '.join(src_parts)}]\n{text[:1200]}")

    if not passages:
        return "No passages met the similarity threshold. Try a broader query."

    return "\n\n---\n\n".join(passages)


# Quick smoke test
_test = search_filings("supply chain risks manufacturing", ticker="AAPL", section_name="Risk Factors", num_results=2)
print("[search_filings test]", _test[:300])

# COMMAND ----------

# ── 5. Tool: get_company_metrics ──────────────────────────────────────────────

# Defensive re-init for the @mlflow.trace decorator below.
import mlflow

@mlflow.trace(name="get_company_metrics", span_type="TOOL")
def get_company_metrics(
    ticker: str,
    fiscal_year_start: int = None,
    fiscal_year_end: int = None,
    metrics_cache: dict = None,
) -> str:
    """
    Retrieves structured financial metrics for a company from the Gold table.
    Returns revenue, net income, margins, YoY growth, debt ratios.
    """
    cache = metrics_cache or METRICS_CACHE
    ticker_upper = ticker.upper()

    if ticker_upper not in cache:
        available = sorted(cache.keys())
        return f"No metrics found for ticker '{ticker_upper}'. Available tickers: {available}"

    ticker_data = cache[ticker_upper]
    years = sorted(ticker_data.keys())

    if fiscal_year_start:
        years = [y for y in years if y >= fiscal_year_start]
    if fiscal_year_end:
        years = [y for y in years if y <= fiscal_year_end]

    if not years:
        return f"No data for {ticker_upper} in the requested fiscal year range."

    def _fmt(v, pct=False):
        if v is None:
            return "N/A"
        if pct:
            return f"{v * 100:.1f}%"
        if abs(v) >= 1e9:
            return f"${v / 1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"${v / 1e6:.1f}M"
        return f"${v:,.0f}"

    lines = [f"Financial metrics for {ticker_upper} ({ticker_data[years[0]].get('company_name', ticker_upper)}):"]
    for fy in years:
        m = ticker_data[fy]
        lines.append(
            f"\nFY{int(fy)}:"
            f"\n  Revenue:              {_fmt(m.get('revenue'))}"
            f"\n  Net Income:           {_fmt(m.get('net_income'))}"
            f"\n  Gross Profit:         {_fmt(m.get('gross_profit'))}"
            f"\n  Operating Income:     {_fmt(m.get('operating_income'))}"
            f"\n  Operating Cash Flow:  {_fmt(m.get('operating_cash_flow'))}"
            f"\n  Total Assets:         {_fmt(m.get('total_assets'))}"
            f"\n  Total Liabilities:    {_fmt(m.get('total_liabilities'))}"
            f"\n  Total Equity:         {_fmt(m.get('total_equity'))}"
            f"\n  Total Debt:           {_fmt(m.get('total_debt'))}"
            f"\n  R&D Expense:          {_fmt(m.get('rd_expense'))}"
            f"\n  Gross Margin:         {_fmt(m.get('gross_margin_pct'), pct=True)}"
            f"\n  Revenue YoY Growth:   {_fmt(m.get('revenue_yoy_growth_pct'), pct=True)}"
            f"\n  Debt/Equity:          {str(round(m['debt_to_equity'], 2)) + 'x' if m.get('debt_to_equity') is not None else 'N/A'}"
            f"\n  Data Quality Score:   {m.get('data_quality_score', 0):.0%}"
        )
    return "\n".join(lines)


# Quick smoke test
_test2 = get_company_metrics("AAPL", fiscal_year_start=2022, fiscal_year_end=2023)
print("[get_company_metrics test]\n", _test2)

# COMMAND ----------

# ── 5b. Tool: get_quarterly_metrics ───────────────────────────────────────────
# Parallel to get_company_metrics but backed by the 10-Q-driven quarterly table.
# Use for interim-period questions; annual questions should still route to
# get_company_metrics since 10-K numbers are audited.

# Defensive re-init for the @mlflow.trace decorator below.
import mlflow

@mlflow.trace(name="get_quarterly_metrics", span_type="TOOL")
def get_quarterly_metrics(
    ticker: str,
    fiscal_year: int = None,
    fiscal_quarter: int = None,
    fiscal_year_start: int = None,
    fiscal_year_end: int = None,
    metrics_cache: dict = None,
) -> str:
    """
    Retrieves discrete-quarter financial metrics from the Gold quarterly table.
    Pass fiscal_year + fiscal_quarter for a single quarter, OR a fiscal_year_start/end
    range for a multi-quarter trend. Q4 is not currently derived from 10-K data.
    """
    cache = metrics_cache if metrics_cache is not None else QUARTERLY_METRICS_CACHE
    if not cache:
        return (
            "Quarterly metrics are not available yet. Run notebook 04b to populate "
            "main.finsage_gold.company_metrics_quarterly, or fall back to get_company_metrics "
            "for annual figures."
        )

    ticker_upper = ticker.upper()
    if ticker_upper not in cache:
        available = sorted(cache.keys())
        return f"No quarterly metrics for ticker '{ticker_upper}'. Available tickers: {available}"

    ticker_data = cache[ticker_upper]

    def _parse_key(k):
        # keys stored as "YYYY-QN"
        fy_str, q_str = k.split("-Q")
        return int(fy_str), int(q_str)

    keys = sorted(ticker_data.keys(), key=_parse_key)
    filtered = []
    for k in keys:
        fy, fq = _parse_key(k)
        if fiscal_year is not None and fy != fiscal_year:
            continue
        if fiscal_quarter is not None and fq != fiscal_quarter:
            continue
        if fiscal_year_start is not None and fy < fiscal_year_start:
            continue
        if fiscal_year_end is not None and fy > fiscal_year_end:
            continue
        filtered.append(k)

    if not filtered:
        return f"No quarterly data for {ticker_upper} matching the requested filters."

    def _fmt(v, pct=False):
        if v is None:
            return "N/A"
        if pct:
            return f"{v * 100:.1f}%"
        if abs(v) >= 1e9:
            return f"${v / 1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"${v / 1e6:.1f}M"
        return f"${v:,.0f}"

    company = ticker_data[filtered[0]].get("company_name", ticker_upper)
    lines = [f"Quarterly financial metrics for {ticker_upper} ({company}):"]
    for k in filtered:
        fy, fq = _parse_key(k)
        m = ticker_data[k]
        lines.append(
            f"\nFY{fy} Q{fq} (period ending {m.get('period_end_date', 'N/A')}):"
            f"\n  Revenue:              {_fmt(m.get('revenue'))}"
            f"\n  Net Income:           {_fmt(m.get('net_income'))}"
            f"\n  Gross Profit:         {_fmt(m.get('gross_profit'))}"
            f"\n  Operating Income:     {_fmt(m.get('operating_income'))}"
            f"\n  Operating Cash Flow:  {_fmt(m.get('operating_cash_flow'))}"
            f"\n  Total Assets:         {_fmt(m.get('total_assets'))}"
            f"\n  Total Liabilities:    {_fmt(m.get('total_liabilities'))}"
            f"\n  Total Equity:         {_fmt(m.get('total_equity'))}"
            f"\n  Total Debt:           {_fmt(m.get('total_debt'))}"
            f"\n  R&D Expense:          {_fmt(m.get('rd_expense'))}"
            f"\n  Gross Margin:         {_fmt(m.get('gross_margin_pct'), pct=True)}"
            f"\n  Revenue YoY (same Q): {_fmt(m.get('revenue_yoy_growth_pct'), pct=True)}"
            f"\n  Debt/Equity:          {str(round(m['debt_to_equity'], 2)) + 'x' if m.get('debt_to_equity') is not None else 'N/A'}"
            f"\n  Data Quality Score:   {m.get('data_quality_score', 0):.0%}"
        )
    return "\n".join(lines)


# Quick smoke test (gracefully no-ops if quarterly cache is empty)
_test3 = get_quarterly_metrics("AAPL", fiscal_year=2024)
print("[get_quarterly_metrics test]\n", _test3[:500])

# COMMAND ----------

# ── 5c. Tool: get_filing_metadata ─────────────────────────────────────────────

# Defensive re-init for the @mlflow.trace decorator below.
import mlflow

@mlflow.trace(name="get_filing_metadata", span_type="TOOL")
def get_filing_metadata(
    ticker: str,
    fiscal_year: int,
    metadata_cache: dict = None,
) -> str:
    """
    Deterministic lookup for 10-K cover-page style metadata.
    """
    cache = metadata_cache if metadata_cache is not None else FILING_METADATA_CACHE
    ticker_upper = ticker.upper()
    if ticker_upper not in cache:
        return f"No filing metadata for ticker '{ticker_upper}'."
    item = cache[ticker_upper].get(int(fiscal_year))
    if not item:
        return f"No filing metadata for {ticker_upper} in FY{int(fiscal_year)}."
    filing_date = item.get("filing_date")
    employees = item.get("employees")
    shares = item.get("shares_outstanding")
    return (
        f"10-K metadata for {ticker_upper} FY{int(fiscal_year)}:"
        f"\n  Filing Date: {filing_date or 'N/A'}"
        f"\n  Employees: {int(employees) if employees is not None else 'N/A'}"
        f"\n  Shares Outstanding: {int(shares) if shares is not None else 'N/A'}"
    )

# COMMAND ----------

# ── 6. Tool schemas (OpenAI function-calling format) ──────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_filings",
            "description": (
                "Semantically search SEC 10-K and 10-Q filing text. 10-K sections: Business, "
                "Risk Factors, MD&A. 10-Q sections: MD&A and (when present) Risk Factors Updates. "
                "Use for qualitative questions about strategy, risks, products, competition, regulation, "
                "supply chain, or anything requiring direct quotes from annual or interim filings. "
                "Scope to filing_type='10-K' for annual language and '10-Q' for interim commentary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query describing what to find in the filings."
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Optional stock ticker to restrict search (e.g. 'AAPL', 'MSFT'). Leave empty to search all companies."
                    },
                    "section_name": {
                        "type": "string",
                        "enum": ["Business", "Risk Factors", "MD&A", "Risk Factors Updates"],
                        "description": (
                            "Optional section filter. 'Business' (10-K only): strategy, products, "
                            "segments. 'Risk Factors' (10-K only): annual risk disclosure. "
                            "'MD&A' (both 10-K and 10-Q): management's discussion of results. "
                            "'Risk Factors Updates' (10-Q Part II Item 1A): mid-year amendments "
                            "to the annual risk factors."
                        )
                    },
                    "filing_type": {
                        "type": "string",
                        "enum": ["10-K", "10-Q"],
                        "description": (
                            "Optional filing-type filter. Use '10-K' for annual-report language, "
                            "'10-Q' for interim/quarterly commentary. Omit to search both."
                        )
                    },
                    "fiscal_year": {
                        "type": "integer",
                        "description": "Optional fiscal year to restrict search (e.g. 2024). Use when the question specifies 'most recent', 'latest', or a specific year. First call get_company_metrics (or get_quarterly_metrics for interim questions) to find the latest available year for that ticker, then pass it here."
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of passages to retrieve (default 5, max 10).",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_metrics",
            "description": (
                "Retrieve ANNUAL (10-K) structured financial metrics for a company. "
                "Use for annual numerical questions about revenue, profit, margins, debt, growth rates, "
                "or any fiscal-year financial comparison. Covers FY2020–FY2026 for 30 companies. "
                "For quarterly/interim-period numbers use get_quarterly_metrics instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. 'AAPL', 'NVDA', 'JPM')."
                    },
                    "fiscal_year_start": {
                        "type": "integer",
                        "description": "Optional start of fiscal year range (e.g. 2021)."
                    },
                    "fiscal_year_end": {
                        "type": "integer",
                        "description": "Optional end of fiscal year range (e.g. 2024)."
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_quarterly_metrics",
            "description": (
                "Retrieve QUARTERLY (10-Q) discrete-quarter financial metrics for a company. "
                "Use for interim-period questions: 'Q2 2024 revenue', 'how did net income trend across Q1-Q3', "
                "same-quarter YoY growth, intra-year balance sheet movement. Covers Q1/Q2/Q3 for 2020+. "
                "Q4 is NOT in this table (use get_company_metrics for annual totals). Pass a specific "
                "fiscal_year + fiscal_quarter for one quarter, or a fiscal_year_start/end range for a trend."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. 'AAPL', 'MSFT')."
                    },
                    "fiscal_year": {
                        "type": "integer",
                        "description": "Specific fiscal year (e.g. 2024). Combine with fiscal_quarter."
                    },
                    "fiscal_quarter": {
                        "type": "integer",
                        "enum": [1, 2, 3],
                        "description": "Fiscal quarter: 1, 2, or 3. Q4 is not stored in the quarterly table."
                    },
                    "fiscal_year_start": {
                        "type": "integer",
                        "description": "Optional start of fiscal year range for a trend view."
                    },
                    "fiscal_year_end": {
                        "type": "integer",
                        "description": "Optional end of fiscal year range for a trend view."
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_filing_metadata",
            "description": (
                "Retrieve deterministic 10-K cover-page metadata fields for one ticker and fiscal year: "
                "filing date, employee count, and shares outstanding."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. 'AAPL')."
                    },
                    "fiscal_year": {
                        "type": "integer",
                        "description": "Fiscal year (e.g. 2024)."
                    },
                },
                "required": ["ticker", "fiscal_year"]
            }
        }
    }
]

TOOL_DISPATCH = {
    "search_filings":        lambda args: search_filings(**args),
    "get_company_metrics":   lambda args: get_company_metrics(**args),
    "get_quarterly_metrics": lambda args: get_quarterly_metrics(**args),
    "get_filing_metadata":   lambda args: get_filing_metadata(**args),
}

print(f"[TOOLS] Registered: {list(TOOL_DISPATCH.keys())}")

# COMMAND ----------

# ── 7. FinSageAgent — pyfunc model class ──────────────────────────────────────

# Defensive re-init — class definition AND predict() at runtime both reference
# these names. Even when this cell runs after restartPython() wiped state,
# the class body and predict() resolve cleanly.
import mlflow, re, json, time
MAX_ITERATIONS = 5  # also exposed as class attribute below — see FinSageAgent

SYSTEM_PROMPT = """\
You are FinSage — an expert financial analyst AI deployed inside a chat
interface for analysts, portfolio managers, and finance researchers. Every
answer you produce is shown verbatim to a professional user, so the bar is
factual precision, conversational clarity, and explicit source attribution.

CORPUS SCOPE
You have data for exactly 30 publicly traded companies, fiscal years
2020–2026 (some tickers extend slightly beyond this range; always defer to
what the tools return):
  AAPL, ABBV, AMZN, BAC, CRM, DDOG, F, GM, GOOGL, GS,
  JNJ, JPM, KO, LCID, MA, MCD, MRK, MSFT, NET, NKE,
  NVDA, PFE, PLTR, RIVN, SBUX, SNOW, TSLA, UNH, V, WMT.

If a user asks about a ticker outside this list (IBM, FB/META, GOOG, NFLX,
ORCL, etc.), decline gracefully and tell them which tickers you DO cover.
Do not attempt to answer from training-data memory — that produces stale or
fabricated numbers.

YOUR FOUR TOOLS
1. search_filings(query, ticker, section_name, fiscal_year, filing_type)
   - Semantic search over SEC 10-K + 10-Q text.
   - 10-K sections: Business, Risk Factors, MD&A.
   - 10-Q sections: MD&A and (when present) Risk Factors Updates.
   - Use for: strategy, risks, products, competitive landscape, supply chain,
     regulation, management commentary, anything qualitative.

2. get_company_metrics(ticker, fiscal_year_start, fiscal_year_end)
   - ANNUAL structured metrics from 10-K: revenue, net income, operating
     income, gross profit, operating cash flow, total assets/liabilities/
     equity/debt, R&D expense, gross margin, YoY growth, debt-to-equity.
   - Use for any annual numerical question.

3. get_quarterly_metrics(ticker, fiscal_year, fiscal_quarter, ...)
   - DISCRETE quarterly (Q1/Q2/Q3 only) data from 10-Qs. Q4 is intentionally
     NOT stored — filers don't tag Q4 as a discrete fact, only as part of
     the annual 10-K.
   - Use for any within-year quarterly question.

4. get_filing_metadata(ticker, fiscal_year)
   - 10-K cover-page facts: filing date, employee count, shares outstanding.
   - Use for any "when filed", "how many employees", "shares outstanding"
     question.

ROUTING — pick BEFORE you answer
- Annual numerical (revenue, margins, growth, ratios, debt) → get_company_metrics
- Quarterly Q1/Q2/Q3 numerical → get_quarterly_metrics
- Filing metadata (employees, shares, filing date) → get_filing_metadata
- Qualitative narrative (risk, strategy, MD&A) → search_filings
- Multi-step (e.g. "growth from FY22 to FY23") → call the metrics tool with
  a year range, then compute the derived value yourself
- Cross-ticker comparison → call get_company_metrics once per ticker
- Q4 question: explain Q4 isn't stored discretely; offer the implied value
  computed as `annual − (Q1 + Q2 + Q3)` if the user wants it

OUT-OF-SCOPE / UNANSWERABLE — DECLINE WITH REASONING
You must decline (politely, with the specific reason) when:
- The fiscal year is in the future or hasn't been filed yet (FY2030, etc.)
- The ticker isn't in the 30-company corpus (IBM, FB, META, GOOG, etc.)
- The user uses an ambiguous/deprecated ticker (FB → say "FB was Meta's
  former ticker; Meta isn't currently in the FinSage corpus")
- A retrieval-only question targets a ticker whose filing text isn't indexed
  (currently MCD's 10-K narrative — the agent has annual metrics but no
   indexed 10-K text for McDonald's)
- A tool returns "No data found" — treat that as authoritative
- A tool returns a value of "N/A" or null for the SPECIFIC metric the user
  asked about (e.g. user asks "net income for Q1 FY24" and the tool returns
  net_income=N/A) — refuse cleanly for that specific metric

PARTIAL-DATA RULE (critical for chatbot UX)
When the user asks for ONE specific metric and the tool returns N/A for it,
DO NOT lead with — or fold into the same sentence as — a different metric's
value. A chatbot reader scans for the headline number and may treat any
prominent dollar figure as the answer to their question, which is harmful.
Instead:
  • State plainly that the requested metric is unavailable for that period.
  • If you want to offer a related metric as context, do so in a SEPARATE
    paragraph, clearly labeled, and never on the first line.
  • Never present an alternative metric as if it answered the original
    question.

REFUSALS must:
- Acknowledge the question briefly
- Name the specific reason (future period / not in corpus / Q4 not stored /
  metric N/A in the underlying filing)
- Offer a concrete alternative when one exists ("I can answer for FY2024" or
  "Try GOOGL instead of GOOG" or "I have the annual figure if that helps") —
  but format it as a follow-up, NOT as if it were the answer.
Never invent data. Never cite a source for data you didn't retrieve.

CHAT TONE
- Conversational and professional. The user is sophisticated; don't over-
  explain basics, but don't be terse either.
- Lead with the answer. First sentence carries the headline number/fact.
  Subsequent sentences add context, comparison, or formula.
- Use natural prose, not bullet-point dumps. Short paragraphs are fine.
- Numbers are formatted at display precision:
    Dollars: `$391.04 billion` or `$391.04B` (not `391035000000`)
    Percentages: `43.31%` (not `0.4331`)
    Ratios: `0.85x` or `0.85` (no more than 4 decimals)
- For comparisons/trends, name both endpoints and the delta:
  "Revenue grew from $383B in FY23 to $391B in FY24 — a 2.0% increase."
- For derived metrics, show the formula on first use:
  "Operating margin = Operating Income ÷ Revenue = $114.3B ÷ $391.0B = 29.2%"

CITATIONS — every factual claim
- Metrics-tool answers: end the response with one or more
    [Source: TICKER | FY#### | metrics]
  lines. Add Q# for quarterly: `[Source: NVDA | FY2024 Q3 | metrics]`.
- Filing-text answers (search_filings): tag every quoted span with
  `[VERBATIM]` for direct quotes or `[SUMMARY]` for paraphrases, then end
  with `[Source: TICKER | FY#### | 10-K/10-Q | Section]`.
- Multi-source answers: include one [Source:] line per distinct source.
- Filing-metadata answers: `[Source: TICKER | FY#### | 10-K Cover Page]`.
- Refusals: no source citation needed — there's nothing to cite.

PRECISION GUARDRAILS
- If a tool returns a value in raw dollars, you display the human-friendly
  scale ($B, $M).
- If the user asks "exactly", give the full number from the tool.
- If two tools return slightly different numbers (rare), prefer the most
  recent filing and note the discrepancy.
- Never round in a way that crosses a rounding boundary (e.g. don't report
  $391.04B as "approximately $400 billion" — that's misleading).

EXAMPLES — gold-standard answer shapes

User: "What was Apple's revenue in fiscal year 2023?"
You:  "Apple's total net sales in fiscal year 2023 were $383.29 billion, a
      ~3% decline from FY2022's $394.33 billion driven primarily by softer
      iPhone and Mac demand.

      [Source: AAPL | FY2023 | metrics]"

User: "What was AAPL's revenue in FY2030?"
You:  "Apple's fiscal year 2030 hasn't occurred yet — it's a future period
      with no SEC filing in existence. The most recent fiscal year I can
      answer for Apple is FY2024 (ended September 28, 2024). Want that
      figure instead?"

User: "Compare FB's revenue to GOOG in FY2023."
You:  "FB was Meta Platforms' former ticker (renamed META in 2022), and GOOG
      is Alphabet's Class C share. Neither is currently in the FinSage
      corpus — I track Alphabet only as Class A (GOOGL), and Meta isn't
      included at this time. I can compare GOOGL FY2023 revenue against any
      of my other 29 tickers if that helps."

User: "What was MSFT's Q4 FY2024 revenue?"
You:  "Q4 isn't stored as a standalone quarter in the FinSage data — SEC
      filers report Q4 only as part of the annual 10-K, not in a discrete
      10-Q. Microsoft's full fiscal year 2024 revenue was $245.12 billion.
      The implied Q4 value is FY2024 total minus Q1+Q2+Q3, which works out
      to roughly $64.7B. Want me to break that down by segment?

      [Source: MSFT | FY2024 | metrics]"

User: "What was Amazon's net income in Q1 of fiscal year 2024?"
      (the quarterly tool returns net_income=N/A for AMZN Q1 FY2024 but
       has revenue and operating_income populated)
You:  "Amazon's Q1 FY2024 net income isn't tagged in the underlying 10-Q
      with a standalone three-month value, so I can't return a verified
      figure for that specific metric.

      For context, Amazon did report Q1 FY2024 revenue of $143.31 billion
      and operating income of $15.31 billion (both from the same filing).
      If a net-income figure is essential, the FY2024 annual total may be
      a reasonable substitute — let me know.

      [Source: AMZN | FY2024 Q1 | metrics]"
      (NOTE the structure: refuse the specific metric in paragraph 1; only
       then introduce the related figures as context, never as the answer.)

User: "Summarize McDonald's risk factors from their FY2023 10-K."
      (search_filings returns no passages — MCD 10-K text didn't extract
       into the silver section table; only their 10-Q narrative is indexed)
You:  "I cannot answer this. McDonald's (MCD) annual 10-K narrative isn't
      in the FinSage retrieval index — there's a known section-extraction
      gap for MCD's 10-K filings (the silver layer captured 0 sections from
      their 7 10-K documents, while their 10-Q narrative did extract
      cleanly). I have MCD's annual structured metrics (revenue, margins,
      etc.) and 10-Q MD&A text, but not 10-K Risk Factors or Business
      narrative. Want me to pull the 10-Q risk-factor updates instead, or
      summarize the structured financials?"
      (NOTE no [Source:] line — there's nothing to cite. Refusals never
       fabricate a source.)
"""


class FinSageAgent(mlflow.pyfunc.PythonModel):

    # Class attribute so predict() never depends on module-global MAX_ITERATIONS
    # — survives any cell-execution order after restartPython.
    MAX_ITERATIONS = 5

    def load_context(self, context):
        import json
        # Annual metrics cache (required)
        with open(context.artifacts["metrics_cache"], "r") as f:
            raw = json.load(f)
        self._metrics_cache = {
            ticker: {int(fy): metrics for fy, metrics in years.items()}
            for ticker, years in raw.items()
        }

        # Quarterly metrics cache (optional — absent if notebook 04b hasn't run yet)
        self._quarterly_cache = {}
        q_path = context.artifacts.get("quarterly_cache") if hasattr(context.artifacts, "get") else None
        if q_path is None and isinstance(context.artifacts, dict):
            q_path = context.artifacts.get("quarterly_cache")
        if q_path:
            try:
                with open(q_path, "r") as f:
                    self._quarterly_cache = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                self._quarterly_cache = {}

        self._filing_metadata_cache = {}
        md_path = context.artifacts.get("filing_metadata_cache") if hasattr(context.artifacts, "get") else None
        if md_path is None and isinstance(context.artifacts, dict):
            md_path = context.artifacts.get("filing_metadata_cache")
        if md_path:
            try:
                with open(md_path, "r") as f:
                    raw_md = json.load(f)
                self._filing_metadata_cache = {
                    ticker: {int(fy): vals for fy, vals in years.items()}
                    for ticker, years in raw_md.items()
                }
            except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
                self._filing_metadata_cache = {}

        self._llm_endpoint    = context.model_config.get("llm_endpoint", "databricks-meta-llama-3-3-70b-instruct")
        self._vs_endpoint     = context.model_config.get("vs_endpoint",  "finsage_vs_endpoint")
        self._vs_index        = context.model_config.get("vs_index",     "main.finsage_gold.filing_chunks_index")
        self._num_results     = int(context.model_config.get("num_results",     5))
        self._sim_threshold   = float(context.model_config.get("similarity_threshold", 0.6))

    def predict(self, context, model_input, params=None):
        import mlflow.deployments, json

        source_pattern = re.compile(r"\[Source:\s*[^\]]+\]", re.IGNORECASE)

        def _extract_sources(text: str) -> list[str]:
            return source_pattern.findall(text or "")

        def _enforce_citation_format(content: str, tool_sources: list[str], used_retrieval: bool) -> str:
            """Final-stage safety net: if the LLM forgot to mark a retrieval-grounded
            answer with [SUMMARY] / [Source: ...], we add them. Metrics-only answers
            are NOT mutated — the system prompt already requires the LLM to emit
            `[Source: TICKER | FY#### | metrics]` itself.

            Critical: only fires when search_filings actually returned non-empty
            results (`tool_sources` is non-empty). If retrieval ran but came back
            empty (e.g. MCD 10-K text gap), the LLM correctly produces a refusal
            and we MUST NOT prepend [SUMMARY] — that turns a clean refusal into a
            falsely-cited answer that confuses both users and eval scorers."""
            if not content:
                return content
            if not used_retrieval or not tool_sources:
                return content
            final = content
            if not re.search(r"\[(VERBATIM|SUMMARY)\]", final, flags=re.IGNORECASE):
                final = "[SUMMARY] " + final
            if not source_pattern.search(final):
                deduped = []
                seen = set()
                for src in tool_sources:
                    if src not in seen:
                        deduped.append(src)
                        seen.add(src)
                final = final.rstrip() + "\n\n" + "\n".join(deduped[:2])
            return final

        # Accept both DataFrame input (Databricks serving) and dict input (notebook
        # testing or unwrap_python_model() in eval). When pyfunc serving is active,
        # the input arrives as a single-row DataFrame whose `messages` cell may have
        # been JSON-stringified by the signature transformer — handle that gracefully.
        if hasattr(model_input, "to_dict"):
            records = model_input.to_dict(orient="records")
            payload = records[0] if records else {}
        elif isinstance(model_input, list) and model_input:
            payload = model_input[0] if isinstance(model_input[0], dict) else {}
        elif isinstance(model_input, dict):
            payload = model_input
        else:
            return {
                "content": (
                    f"Unsupported input type {type(model_input).__name__}. "
                    "Send a dict like {'messages': [...]} or a list of such dicts."
                ),
                "messages": [],
            }

        messages = payload.get("messages", [])
        if isinstance(messages, str):
            # MLflow signature mangling: messages can arrive as a JSON string.
            try:
                messages = json.loads(messages)
            except (json.JSONDecodeError, TypeError):
                messages = []

        if not isinstance(messages, list) or not messages:
            return {"content": "No valid messages in request payload.", "messages": []}

        # Every question goes through the LLM tool loop. The system prompt is the
        # single source of truth for routing, refusal handling, citations, and
        # tone — no regex-based shortcuts. This costs more wall-time per
        # question (~5–15 s vs <100 ms) but produces conversational, properly-
        # cited, professionally-toned answers suitable for direct display in
        # the FinSage chat UI. See SYSTEM_PROMPT for the full contract.

        # Build working message list with system prompt prepended
        working_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
        collected_sources = []
        used_retrieval_tool = False
        max_runtime_seconds = 150
        started_at = time.time()

        deploy_client = mlflow.deployments.get_deploy_client("databricks")

        for iteration in range(self.MAX_ITERATIONS):
            if time.time() - started_at > max_runtime_seconds:
                break
            response = deploy_client.predict(
                endpoint=self._llm_endpoint,
                inputs={
                    "messages":    working_messages,
                    "tools":       TOOL_SCHEMAS,
                    "tool_choice": "auto",
                    "temperature": 0.1,
                    "max_tokens":  1024,
                },
            )

            choice      = response["choices"][0]
            finish      = choice.get("finish_reason", "")
            msg         = choice["message"]
            tool_calls  = msg.get("tool_calls") or []

            # No tool calls → final answer
            if not tool_calls or finish == "stop":
                final_content = msg.get("content", "")
                final_content = _enforce_citation_format(
                    final_content,
                    tool_sources=collected_sources,
                    used_retrieval=used_retrieval_tool,
                )
                working_messages.append({"role": "assistant", "content": final_content})
                return {"content": final_content, "messages": working_messages[1:]}  # strip system prompt

            # Append assistant message with tool_calls to history
            working_messages.append(msg)

            # Execute each tool call
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                raw_args  = tc["function"].get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}

                if tool_name == "search_filings":
                    used_retrieval_tool = True
                    result = search_filings(
                        query=args.get("query", ""),
                        ticker=args.get("ticker"),
                        section_name=args.get("section_name"),
                        fiscal_year=args.get("fiscal_year"),
                        filing_type=args.get("filing_type"),
                        num_results=args.get("num_results", self._num_results),
                        similarity_threshold=self._sim_threshold,
                    )
                elif tool_name == "get_company_metrics":
                    result = get_company_metrics(
                        ticker=args.get("ticker", ""),
                        fiscal_year_start=args.get("fiscal_year_start"),
                        fiscal_year_end=args.get("fiscal_year_end"),
                        metrics_cache=self._metrics_cache,
                    )
                elif tool_name == "get_quarterly_metrics":
                    result = get_quarterly_metrics(
                        ticker=args.get("ticker", ""),
                        fiscal_year=args.get("fiscal_year"),
                        fiscal_quarter=args.get("fiscal_quarter"),
                        fiscal_year_start=args.get("fiscal_year_start"),
                        fiscal_year_end=args.get("fiscal_year_end"),
                        metrics_cache=self._quarterly_cache,
                    )
                elif tool_name == "get_filing_metadata":
                    result = get_filing_metadata(
                        ticker=args.get("ticker", ""),
                        fiscal_year=args.get("fiscal_year"),
                        metadata_cache=self._filing_metadata_cache,
                    )
                else:
                    result = f"Unknown tool: {tool_name}"

                if tool_name == "search_filings":
                    collected_sources.extend(_extract_sources(result))

                working_messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.get("id", tool_name),
                    "content":      result,
                })

        # Max iterations reached — ask the LLM for a best-effort answer
        working_messages.append({
            "role":    "user",
            "content": "Based on the tool results above, provide your best answer now."
        })
        final_response = deploy_client.predict(
            endpoint=self._llm_endpoint,
            inputs={"messages": working_messages, "temperature": 0.1, "max_tokens": 768},
        )
        final_content = final_response["choices"][0]["message"].get("content", "")
        final_content = _enforce_citation_format(
            final_content,
            tool_sources=collected_sources,
            used_retrieval=used_retrieval_tool,
        )
        return {"content": final_content, "messages": working_messages[1:]}


print("[FinSageAgent] Class defined.")

# COMMAND ----------

# ── 8. Local smoke tests ──────────────────────────────────────────────────────

# Defensive re-init for everything this cell + agent.predict() touch at
# module level. Note that this cell still requires cells 4 (caches) and 7
# (FinSageAgent class) to have run in this kernel — those produce non-widget
# objects (METRICS_CACHE, QUARTERLY_METRICS_CACHE, FILING_METADATA_CACHE,
# FinSageAgent, TOOL_SCHEMAS, search_filings + the metric tools).
import json, time, re
CATALOG              = dbutils.widgets.get("catalog")
LLM_ENDPOINT         = dbutils.widgets.get("llm_endpoint")
VS_ENDPOINT          = dbutils.widgets.get("vs_endpoint")
VS_INDEX_NAME        = f"{CATALOG}.finsage_gold.filing_chunks_index"
NUM_RESULTS          = int(dbutils.widgets.get("num_results"))
SIMILARITY_THRESHOLD = float(dbutils.widgets.get("similarity_threshold"))
MAX_ITERATIONS       = 5

agent = FinSageAgent()

# Simulate load_context manually for notebook testing
class _FakeContext:
    class artifacts:
        pass
    model_config = {
        "llm_endpoint": LLM_ENDPOINT,
        "vs_endpoint":  VS_ENDPOINT,
        "vs_index":     VS_INDEX_NAME,
        "num_results":  NUM_RESULTS,
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }

fake_ctx = _FakeContext()
fake_ctx.artifacts = {
    "metrics_cache":    "/tmp/metrics_cache.json",
    "quarterly_cache":  "/tmp/quarterly_cache.json",
    "filing_metadata_cache": "/tmp/filing_metadata_cache.json",
}

# Save caches to temp files (mimics what MLflow will do)
import json, os
os.makedirs("/tmp", exist_ok=True)
with open("/tmp/metrics_cache.json", "w") as f:
    serialisable = {t: {str(fy): m for fy, m in yrs.items()} for t, yrs in METRICS_CACHE.items()}
    json.dump(serialisable, f)
with open("/tmp/quarterly_cache.json", "w") as f:
    # Quarterly keys are already strings ("YYYY-QN") so no transform needed
    json.dump(QUARTERLY_METRICS_CACHE, f)
with open("/tmp/filing_metadata_cache.json", "w") as f:
    serialisable_md = {t: {str(fy): vals for fy, vals in yrs.items()} for t, yrs in FILING_METADATA_CACHE.items()}
    json.dump(serialisable_md, f)

agent.load_context(fake_ctx)

TEST_QUESTIONS = [
    "What was Apple's revenue and net income in fiscal year 2023?",
    "What supply chain risks did NVIDIA disclose in their most recent 10-K?",
    "Compare Microsoft and Alphabet's operating margins in 2023.",
]

# Track smoke-test outcomes — fail the cell loudly if the agent never produced
# a numerical signal on a deterministic-answerable question. Better to catch
# regressions here than after a 10-minute MLflow log + UC register + serving
# rollout cycle.
_smoke_failures = []
for q in TEST_QUESTIONS:
    print(f"\n{'='*70}")
    print(f"Q: {q}")
    try:
        result = agent.predict(None, {"messages": [{"role": "user", "content": q}]})
        content = result.get("content", "") if isinstance(result, dict) else str(result)
    except Exception as e:
        _smoke_failures.append((q, f"{type(e).__name__}: {e}"))
        print(f"A: ERROR — {type(e).__name__}: {e}")
        continue
    print(f"A: {content[:800]}")
    # Apple revenue + Microsoft/Alphabet comparison should both produce a $ figure.
    if q == TEST_QUESTIONS[0] and "$" not in content:
        _smoke_failures.append((q, "no $ figure in revenue lookup"))
    if q == TEST_QUESTIONS[2] and "%" not in content:
        _smoke_failures.append((q, "no % figure in operating margin compare"))

if _smoke_failures:
    raise RuntimeError(
        "Smoke tests failed; not logging this agent. Failures:\n  " +
        "\n  ".join(f"{q!r} → {reason}" for q, reason in _smoke_failures)
    )
print("\n[SMOKE] all 3 questions produced expected signal — proceeding to log_model.")

# COMMAND ----------

# ── 9. MLflow logging & Unity Catalog registration ────────────────────────────

# Defensive re-init — re-read all widget-derived constants this cell uses so
# it survives "Run cell" in isolation after restartPython() wiped state.
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex

CATALOG              = dbutils.widgets.get("catalog")
ENV                  = dbutils.widgets.get("env")
LLM_ENDPOINT         = dbutils.widgets.get("llm_endpoint")
VS_ENDPOINT          = dbutils.widgets.get("vs_endpoint")
VS_INDEX_NAME        = f"{CATALOG}.finsage_gold.filing_chunks_index"
NUM_RESULTS          = int(dbutils.widgets.get("num_results"))
SIMILARITY_THRESHOLD = float(dbutils.widgets.get("similarity_threshold"))
UC_MODEL_NAME        = f"{CATALOG}.finsage_gold.finsage_rag_agent"
MAX_ITERATIONS       = 5

input_schema  = Schema([ColSpec("string", "messages")])
output_schema = Schema([ColSpec("string", "content")])
signature     = ModelSignature(inputs=input_schema, outputs=output_schema)

input_example = {
    "messages": [
        {"role": "user", "content": "What was Apple's revenue growth in fiscal year 2023?"}
    ]
}

# Resource dependencies — Databricks Agent Framework injects M2M OAuth credentials
# into the serving container for these resources at runtime. Without this list, the
# deployed model cannot authenticate to call the LLM endpoint or Vector Search index.
resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
    DatabricksVectorSearchIndex(index_name=VS_INDEX_NAME),
]

# When this notebook runs as a job task (vs. interactively in a Git Folder),
# MLflow has no implicit experiment context — start_run() then fails with
# "Could not find experiment with ID None". Bind to a stable workspace
# experiment path before opening the run; create it on first execution.
EXPERIMENT_PATH = f"/Users/digvijay@arsaga.jp/finsage_rag_agent_{ENV}"
try:
    mlflow.set_experiment(EXPERIMENT_PATH)
except Exception:
    from mlflow.tracking import MlflowClient
    _client = MlflowClient()
    _exp = _client.get_experiment_by_name(EXPERIMENT_PATH)
    _exp_id = _exp.experiment_id if _exp else _client.create_experiment(EXPERIMENT_PATH)
    mlflow.set_experiment(experiment_id=_exp_id)

with mlflow.start_run(run_name=f"finsage_rag_agent_{ENV}") as run:
    mlflow.log_params({
        "llm_endpoint":          LLM_ENDPOINT,
        "vs_index":              VS_INDEX_NAME,
        "num_results":           NUM_RESULTS,
        "similarity_threshold":  SIMILARITY_THRESHOLD,
        "max_iterations":        MAX_ITERATIONS,
    })

    # log_model(artifacts={...}) handles cache persistence — no separate
    # log_artifact() needed (would write the same JSON twice into the run).
    # `requests` is intentionally NOT in pip_requirements: it's only used at
    # notebook-time for SEC EDGAR fetches in cell 3, never inside the served
    # agent's predict path. databricks-vectorsearch transitively brings the
    # `requests` it needs at serving time.
    model_info = mlflow.pyfunc.log_model(
        artifact_path="finsage_rag_agent",
        python_model=agent,
        artifacts={
            "metrics_cache":    "/tmp/metrics_cache.json",
            "quarterly_cache":  "/tmp/quarterly_cache.json",
            "filing_metadata_cache": "/tmp/filing_metadata_cache.json",
        },
        model_config={
            "llm_endpoint":          LLM_ENDPOINT,
            "vs_endpoint":           VS_ENDPOINT,
            "vs_index":              VS_INDEX_NAME,
            "num_results":           NUM_RESULTS,
            "similarity_threshold":  SIMILARITY_THRESHOLD,
        },
        signature=signature,
        input_example=input_example,
        registered_model_name=UC_MODEL_NAME,
        resources=resources,
        pip_requirements=[
            "mlflow>=3.0,<4.0",
            "databricks-vectorsearch>=0.40,<1.0",
            "databricks-sdk>=0.30,<1.0",
        ],
    )

    print(f"[MLflow] Run ID: {run.info.run_id}")
    print(f"[MLflow] Model URI: {model_info.model_uri}")
    print(f"[UC] Registered as: {UC_MODEL_NAME}")

# COMMAND ----------

# ── 10. Deploy to Databricks Model Serving ────────────────────────────────────

# Defensive re-init — make this cell safe to "Run cell" in isolation.
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from databricks.sdk.errors import ResourceDoesNotExist

CATALOG        = dbutils.widgets.get("catalog")
UC_MODEL_NAME  = f"{CATALOG}.finsage_gold.finsage_rag_agent"
AGENT_ENDPOINT = "finsage_agent_endpoint"

w = WorkspaceClient()

# Get the latest registered model version
client        = mlflow.tracking.MlflowClient()
all_versions  = client.search_model_versions(f"name='{UC_MODEL_NAME}'")
model_version = max(all_versions, key=lambda v: int(v.version)).version
print(f"[DEPLOY] Deploying {UC_MODEL_NAME} version {model_version} → {AGENT_ENDPOINT}")

served_entity = ServedEntityInput(
    entity_name=UC_MODEL_NAME,
    entity_version=str(model_version),
    workload_size="Small",
    scale_to_zero_enabled=True,
)

endpoint_config = EndpointCoreConfigInput(name=AGENT_ENDPOINT, served_entities=[served_entity])

try:
    existing = w.serving_endpoints.get(AGENT_ENDPOINT)
    print(f"[DEPLOY] Endpoint exists (state={existing.state.ready}). Updating config...")
    w.serving_endpoints.update_config(name=AGENT_ENDPOINT, served_entities=[served_entity])
    print("[DEPLOY] Update submitted.")
except ResourceDoesNotExist:
    print("[DEPLOY] Creating new endpoint...")
    w.serving_endpoints.create(name=AGENT_ENDPOINT, config=endpoint_config)
    print("[DEPLOY] Creation submitted.")

print(f"[DEPLOY] Monitor at: https://dbc-f33010ed-00fc.cloud.databricks.com/ml/endpoints/{AGENT_ENDPOINT}")

# COMMAND ----------

# ── 11. Wait for endpoint + live test ─────────────────────────────────────────

# Defensive re-init — survives standalone "Run cell" after restartPython().
import time
from databricks.sdk import WorkspaceClient

w              = WorkspaceClient()
AGENT_ENDPOINT = "finsage_agent_endpoint"
# model_version is set by cell 10 when run sequentially. If this cell is run
# standalone, recover the latest UC version.
try:
    model_version  # noqa: F821
except NameError:
    import mlflow
    _client = mlflow.tracking.MlflowClient()
    CATALOG = dbutils.widgets.get("catalog")
    _all = _client.search_model_versions(f"name='{CATALOG}.finsage_gold.finsage_rag_agent'")
    model_version = max(_all, key=lambda v: int(v.version)).version

print(f"Waiting for endpoint '{AGENT_ENDPOINT}' to serve v{model_version}...")
timeout, poll = 20 * 60, 20
start = time.time()

# Endpoint.state.ready stays READY while a new version rolls out in the background
# (the previous version keeps serving traffic until the atomic swap). Exiting on
# ready alone runs live tests against the OLD version. Require all three: service
# ready, no in-flight config update, and the version we just deployed is the one
# being served.
deployment_ok = False
while True:
    if time.time() - start > timeout:
        print("Timeout waiting for endpoint. Check the Serving UI manually.")
        break
    try:
        ep              = w.serving_endpoints.get(AGENT_ENDPOINT)
        state           = str(ep.state.ready)         if ep.state and ep.state.ready         else "UNKNOWN"
        config_update   = str(ep.state.config_update) if ep.state and ep.state.config_update else ""
        served_versions = {se.entity_version for se in (ep.config.served_entities or [])} if ep.config else set()
        print(f"  state={state} config_update={config_update} served_versions={served_versions}")

        if "FAILED" in state.upper() or "FAILED" in config_update.upper():
            print(f"Endpoint failed: state={state} config_update={config_update}")
            break

        if (state == "EndpointStateReady.READY"
                and "NOT_UPDATING" in config_update.upper()
                and str(model_version) in served_versions):
            print(f"Endpoint is READY and serving v{model_version}.")
            deployment_ok = True
            break
    except Exception as e:
        print(f"  Polling error: {e}")
    time.sleep(poll)

# Live test via SDK — our pyfunc has a custom (non-chat) output schema,
# so we send via dataframe_records and read from resp.predictions.
# Only runs if the deployment actually succeeded — otherwise we'd be testing
# the OLD served version and getting misleading PASS results.
if not deployment_ok:
    print(f"\n[LIVE TEST] Skipped — endpoint never reached READY for v{model_version}. "
          f"Inspect the Serving UI before drawing conclusions.")
else:
    live_test_questions = [
        "What was NVIDIA's revenue and net income in fiscal year 2024?",
        "What risks did Tesla disclose about autonomous driving in their 10-K?",
    ]

    live_test_failures = []
    for q in live_test_questions:
        print(f"\n{'='*70}\nQ: {q}")
        try:
            resp = w.serving_endpoints.query(
                name=AGENT_ENDPOINT,
                dataframe_records=[{"messages": [{"role": "user", "content": q}]}],
            )
            preds = resp.predictions
            pred  = preds[0] if isinstance(preds, list) and preds else preds
            answer = pred.get("content", str(pred)) if isinstance(pred, dict) else str(pred)
            print(f"A: {answer[:800]}")
            if not answer or "error" in answer.lower()[:120]:
                live_test_failures.append((q, "empty/error response"))
        except Exception as e:
            print(f"Live test error: {type(e).__name__}: {e}")
            live_test_failures.append((q, f"{type(e).__name__}: {e}"))

    if live_test_failures:
        print(
            f"\n[LIVE TEST] {len(live_test_failures)}/{len(live_test_questions)} live "
            f"queries returned errors. The model is registered (v{model_version}) but "
            f"the endpoint may need attention."
        )
    else:
        print(f"\n[LIVE TEST] All {len(live_test_questions)} queries returned valid responses.")
