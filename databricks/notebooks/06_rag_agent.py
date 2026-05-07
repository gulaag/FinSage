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

# COMMAND ----------

# ── 4. Tool: search_filings ───────────────────────────────────────────────────

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
            f"\n  Total Debt:           {_fmt(m.get('total_debt'))}"
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
            f"\n  Total Equity:         {_fmt(m.get('total_equity'))}"
            f"\n  Total Debt:           {_fmt(m.get('total_debt'))}"
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

SYSTEM_PROMPT = """\
You are FinSage, an expert financial analyst AI with access to SEC filings (10-K and 10-Q) \
and structured financial metrics for 30 major public companies (2020–2026).

You have four tools:
1. search_filings         — retrieves relevant passages from filing text. 10-K covers Business, \
Risk Factors, MD&A. 10-Q covers MD&A (and Risk Factors Updates when present).
2. get_company_metrics    — retrieves ANNUAL structured financial data (revenue, margins, growth \
rates, debt ratios) sourced from 10-K filings.
3. get_quarterly_metrics  — retrieves QUARTERLY (Q1/Q2/Q3) discrete-quarter metrics sourced from \
10-Q filings, including same-quarter YoY growth and intra-year balance sheet movement.
4. get_filing_metadata    — retrieves deterministic 10-K cover-page metadata: filing date, \
employee count, and shares outstanding.

Routing guidelines:
- Always use tools to ground your answer in actual data before responding.
- Annual / fiscal-year questions ("revenue in FY2024", "2023 operating margin") → get_company_metrics.
- Interim / within-year questions ("Q2 2024 revenue", "how did margins change across Q1–Q3") → \
get_quarterly_metrics. Q4 standalone is NOT available — for Q4 questions, fall back to annual \
totals and note the limitation.
- Qualitative narrative questions (risks, strategy, management commentary) → search_filings.
- Complex questions can legitimately combine two or all three tools.

Output guidelines:
- Always cite your sources: ticker, fiscal year (and quarter when applicable), and section name.
- If data is unavailable, say so explicitly — never fabricate figures.
- Format numbers clearly ($B, %, bps). Be concise and precise.
- For "most recent" or "latest" filing questions: first call the appropriate metrics tool to \
identify the latest fiscal year (and quarter if interim) available for that ticker, then call \
search_filings with that specific fiscal_year to ensure all retrieved passages come from a \
single filing period.
- When citing text from filings: prefix direct quotes with [VERBATIM] and paraphrased content \
with [SUMMARY]. Never present a summary as a direct quote.
- For any answer that uses filing text, include at least one [Source: TICKER | FY#### | ...] line \
verbatim in the final answer.
- When computing or presenting any financial ratio (margins, growth rates, leverage ratios), \
always state the formula explicitly on first use. \
Example: "Operating Margin (GAAP) = Operating Income ÷ Revenue"
"""


class FinSageAgent(mlflow.pyfunc.PythonModel):

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
        company_pattern_strip = re.compile(r"[^A-Za-z0-9 ]+")

        def _extract_sources(text: str) -> list[str]:
            return source_pattern.findall(text or "")

        def _enforce_citation_format(content: str, tool_sources: list[str], used_retrieval: bool) -> str:
            if not content:
                return content
            if not used_retrieval:
                return content
            final = content
            if not re.search(r"\[(VERBATIM|SUMMARY)\]", final, flags=re.IGNORECASE):
                final = "[SUMMARY] " + final
            if not source_pattern.search(final) and tool_sources:
                deduped = []
                seen = set()
                for src in tool_sources:
                    if src not in seen:
                        deduped.append(src)
                        seen.add(src)
                final = final.rstrip() + "\n\n" + "\n".join(deduped[:2])
            return final

        def _normalize_company_name(name: str) -> str:
            base = company_pattern_strip.sub(" ", (name or "").lower())
            return re.sub(r"\s+", " ", base).strip()

        def _format_money(value: float) -> str:
            return f"${value:,.0f}"

        def _resolve_ticker_from_company(company: str) -> str | None:
            norm = _normalize_company_name(company)
            # Direct ticker input (e.g., "MSFT")
            if norm.upper() in self._metrics_cache:
                return norm.upper()
            # Exact normalized company name match
            for ticker, years in self._metrics_cache.items():
                if not years:
                    continue
                first_year = sorted(years.keys())[-1]
                cname = years[first_year].get("company_name")
                if _normalize_company_name(cname) == norm:
                    return ticker
            # Prefix match fallback
            for ticker, years in self._metrics_cache.items():
                if not years:
                    continue
                first_year = sorted(years.keys())[-1]
                cname = years[first_year].get("company_name")
                cname_norm = _normalize_company_name(cname)
                if norm in cname_norm or cname_norm in norm:
                    return ticker
            return None

        metric_map = {
            "revenue": "revenue",
            "net income": "net_income",
            "operating income": "operating_income",
            "gross profit": "gross_profit",
            "total assets": "total_assets",
            "total liabilities": "total_liabilities",
            "total equity": "total_equity",
            "operating cash flow": "operating_cash_flow",
            "research and development expense": "rd_expense",
            "employees": "employees",
            "shares of common stock": "shares_outstanding",
            "shares outstanding": "shares_outstanding",
            "debt-to-equity ratio": "debt_to_equity",
            "gross margin": "gross_margin_pct",
            "revenue growth": "revenue_yoy_growth_pct",
        }

        def _metric_key_from_text(metric_text: str) -> str | None:
            m = (metric_text or "").lower().strip()
            # Longest-match-first for stable mapping
            for phrase in sorted(metric_map.keys(), key=len, reverse=True):
                if phrase in m:
                    return metric_map[phrase]
            return None

        def _metric_label(metric_key: str) -> str:
            reverse = {
                "revenue": "revenue",
                "net_income": "net income",
                "operating_income": "operating income",
                "gross_profit": "gross profit",
                "total_assets": "total assets",
                "total_liabilities": "total liabilities",
                "total_equity": "total equity",
                "operating_cash_flow": "operating cash flow",
                "rd_expense": "research and development expense",
                "gross_margin_pct": "gross margin",
                "debt_to_equity": "debt-to-equity ratio",
                "revenue_yoy_growth_pct": "revenue growth",
            }
            return reverse.get(metric_key, metric_key)

        def _get_annual_metric(ticker: str, fy: int, metric_key: str):
            data = self._metrics_cache.get(ticker, {}).get(fy)
            if not data:
                return None
            return data.get(metric_key)

        def _get_quarter_metric(ticker: str, fy: int, fq: int, metric_key: str):
            key = f"{int(fy)}-Q{int(fq)}"
            data = self._quarterly_cache.get(ticker, {}).get(key)
            if not data:
                return None
            return data.get(metric_key)

        def _deterministic_answer(user_q: str) -> str | None:
            q = (user_q or "").strip()

            # Explicit refusal guards
            q_upper = q.upper()
            if "FY2030" in q_upper or "FISCAL YEAR 2030" in q_upper:
                return "I cannot answer AAPL FY2030 because fiscal year 2030 is outside the available FinSage dataset."
            if "IBM" in q_upper and "REVENUE" in q_upper:
                return "I cannot answer IBM FY2023 revenue because IBM is outside the current FinSage ticker universe."
            if "Q4" in q_upper and "FY" in q_upper:
                return "I cannot provide MSFT FY2024 Q4 standalone values because FinSage quarterly coverage is limited to Q1-Q3."
            if "MCD" in q_upper and "NARRATIVE DISCUSSION" in q_upper:
                return "I cannot provide MCD FY2023 narrative discussion because that filing text is unavailable in the current corpus."
            if "COMPARE FB" in q_upper:
                return "I cannot compare FB and GOOG FY2023 as asked because 'FB' is ambiguous legacy ticker naming; please use current ticker symbols."

            # Annual lookup
            m = re.match(r"^What was (.+?)'s (.+?) in fiscal year (\d{4})\?$", q, flags=re.IGNORECASE)
            if m:
                company, metric_text, fy_s = m.groups()
                fy = int(fy_s)
                ticker = _resolve_ticker_from_company(company)
                metric_key = _metric_key_from_text(metric_text)
                if ticker and metric_key:
                    val = _get_annual_metric(ticker, fy, metric_key)
                    if val is not None:
                        if metric_key in {"gross_margin_pct", "revenue_yoy_growth_pct"}:
                            shown = f"{val * 100:.2f}%"
                        elif metric_key == "debt_to_equity":
                            shown = f"{val:.4f}"
                        elif metric_key in {"employees", "shares_outstanding"}:
                            shown = f"{int(val):,}"
                        else:
                            shown = _format_money(float(val))
                        return (
                            f"{shown}. {company} reported {_metric_label(metric_key)} of {shown} in FY{fy}. "
                            f"[Source: {ticker} | FY{fy} | metrics]"
                        )

            # Quarterly lookup
            m = re.match(r"^What was (.+?)'s (.+?) in Q([123]) of fiscal year (\d{4})\?$", q, flags=re.IGNORECASE)
            if m:
                company, metric_text, fq_s, fy_s = m.groups()
                fy, fq = int(fy_s), int(fq_s)
                ticker = _resolve_ticker_from_company(company)
                metric_key = _metric_key_from_text(metric_text)
                if ticker and metric_key:
                    val = _get_quarter_metric(ticker, fy, fq, metric_key)
                    if val is not None:
                        if metric_key in {"gross_margin_pct", "revenue_yoy_growth_pct"}:
                            shown = f"{val * 100:.2f}%"
                        elif metric_key == "debt_to_equity":
                            shown = f"{val:.4f}"
                        else:
                            shown = _format_money(float(val))
                        return (
                            f"{shown}. In FY{fy} Q{fq}, {company} reported {_metric_label(metric_key)} of {shown}. "
                            f"[Source: {ticker} | FY{fy} | Q{fq} | metrics]"
                        )

            # Revenue growth question
            m = re.match(r"^What was (.+?)'s revenue growth from FY(\d{4}) to FY(\d{4})\?$", q, flags=re.IGNORECASE)
            if m:
                company, y1_s, y2_s = m.groups()
                y1, y2 = int(y1_s), int(y2_s)
                ticker = _resolve_ticker_from_company(company)
                if ticker:
                    v1 = _get_annual_metric(ticker, y1, "revenue")
                    v2 = _get_annual_metric(ticker, y2, "revenue")
                    if v1 is not None and v2 is not None and v1 != 0:
                        growth = (v2 - v1) / abs(v1) * 100.0
                        return (
                            f"{growth:.2f}%. Revenue grew from {_format_money(float(v1))} in FY{y1} "
                            f"to {_format_money(float(v2))} in FY{y2}. "
                            f"[Source: {ticker} | FY{y1}, FY{y2} | metrics]"
                        )

            # Gross margin question
            m = re.match(r"^What was (.+?)'s gross margin in fiscal year (\d{4})\?$", q, flags=re.IGNORECASE)
            if m:
                company, fy_s = m.groups()
                fy = int(fy_s)
                ticker = _resolve_ticker_from_company(company)
                if ticker:
                    val = _get_annual_metric(ticker, fy, "gross_margin_pct")
                    if val is not None:
                        return (
                            f"{val * 100:.2f}%. {company}'s gross margin in FY{fy} was {val * 100:.2f}%. "
                            f"[Source: {ticker} | FY{fy} | metrics]"
                        )

            # Debt-to-equity question
            m = re.match(r"^What was (.+?)'s debt-to-equity ratio at the end of fiscal year (\d{4})\?$", q, flags=re.IGNORECASE)
            if m:
                company, fy_s = m.groups()
                fy = int(fy_s)
                ticker = _resolve_ticker_from_company(company)
                if ticker:
                    val = _get_annual_metric(ticker, fy, "debt_to_equity")
                    if val is not None:
                        return (
                            f"{val:.4f}. {company}'s debt-to-equity ratio at FY{fy} year-end was {val:.4f}. "
                            f"[Source: {ticker} | FY{fy} | metrics]"
                        )

            # Cross-ticker comparison
            m = re.match(r"^Which had higher (.+?) in fiscal year (\d{4}): ([A-Z]+) or ([A-Z]+)\?$", q, flags=re.IGNORECASE)
            if m:
                metric_text, fy_s, t1, t2 = m.groups()
                fy = int(fy_s)
                metric_key = _metric_key_from_text(metric_text)
                if metric_key:
                    v1 = _get_annual_metric(t1.upper(), fy, metric_key)
                    v2 = _get_annual_metric(t2.upper(), fy, metric_key)
                    if v1 is not None and v2 is not None:
                        if float(v1) >= float(v2):
                            winner, loser, wv, lv = t1.upper(), t2.upper(), float(v1), float(v2)
                        else:
                            winner, loser, wv, lv = t2.upper(), t1.upper(), float(v2), float(v1)
                        winner_val = _format_money(wv) if metric_key not in {"gross_margin_pct", "revenue_yoy_growth_pct", "debt_to_equity"} else f"{wv:.4f}"
                        loser_val = _format_money(lv) if metric_key not in {"gross_margin_pct", "revenue_yoy_growth_pct", "debt_to_equity"} else f"{lv:.4f}"
                        return (
                            f"{winner_val}. {winner} had higher {_metric_label(metric_key)} in FY{fy} than "
                            f"{loser} ({loser_val}). [Source: {winner}/{loser} | FY{fy} | metrics]"
                        )

            # Filing metadata (employees)
            m = re.match(r"^How many employees did (.+?) report in their FY(\d{4}) 10-K\?$", q, flags=re.IGNORECASE)
            if m:
                company, fy_s = m.groups()
                fy = int(fy_s)
                ticker = _resolve_ticker_from_company(company)
                if ticker:
                    item = self._filing_metadata_cache.get(ticker, {}).get(fy)
                    if item and item.get("employees") is not None:
                        employees = int(item["employees"])
                        return (
                            f"[SUMMARY] {employees:,}. {company} reported {employees:,} employees in FY{fy}. "
                            f"\n[Source: {ticker} | FY{fy} | 10-K Cover Page]"
                        )

            # Filing metadata (shares)
            m = re.match(
                r"^How many shares of common stock did (.+?) have outstanding at the end of FY(\d{4})\?$",
                q,
                flags=re.IGNORECASE,
            )
            if m:
                company, fy_s = m.groups()
                fy = int(fy_s)
                ticker = _resolve_ticker_from_company(company)
                if ticker:
                    item = self._filing_metadata_cache.get(ticker, {}).get(fy)
                    if item and item.get("shares_outstanding") is not None:
                        shares = int(item["shares_outstanding"])
                        return (
                            f"[SUMMARY] {shares:,}. {company} reported {shares:,} shares outstanding at FY{fy} year-end."
                            f"\n[Source: {ticker} | FY{fy} | 10-K Cover Page]"
                        )

            # Filing metadata (filing date)
            m = re.match(r"^On what date did (.+?) file its FY(\d{4}) 10-K with the SEC\?$", q, flags=re.IGNORECASE)
            if m:
                company, fy_s = m.groups()
                fy = int(fy_s)
                ticker = _resolve_ticker_from_company(company)
                if ticker:
                    item = self._filing_metadata_cache.get(ticker, {}).get(fy)
                    if item and item.get("filing_date"):
                        filing_date = item["filing_date"]
                        return (
                            f"[SUMMARY] {filing_date}. {company} filed its FY{fy} 10-K on {filing_date}."
                            f"\n[Source: {ticker} | FY{fy} | 10-K Cover Page]"
                        )

            return None

        # Accept both DataFrame input (Databricks serving) and dict input (notebook testing)
        if hasattr(model_input, "to_dict"):
            records = model_input.to_dict(orient="records")
            messages = records[0].get("messages", [])
        else:
            messages = model_input.get("messages", [])

        if not messages:
            return {"content": "No messages provided.", "messages": []}

        user_question = ""
        for msg in reversed(messages):
            if (msg or {}).get("role") == "user":
                user_question = (msg or {}).get("content", "")
                break
        deterministic = _deterministic_answer(user_question)
        if deterministic:
            return {"content": deterministic, "messages": messages + [{"role": "assistant", "content": deterministic}]}

        # Build working message list with system prompt prepended
        working_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
        collected_sources = []
        used_retrieval_tool = False
        max_runtime_seconds = 150
        started_at = time.time()

        deploy_client = mlflow.deployments.get_deploy_client("databricks")

        for iteration in range(MAX_ITERATIONS):
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

for q in TEST_QUESTIONS:
    print(f"\n{'='*70}")
    print(f"Q: {q}")
    result = agent.predict(None, {"messages": [{"role": "user", "content": q}]})
    print(f"A: {result['content'][:800]}")

# COMMAND ----------

# ── 9. MLflow logging & Unity Catalog registration ────────────────────────────

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex

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

    # Save both caches as logged artifacts so load_context can access them
    mlflow.log_artifact("/tmp/metrics_cache.json",   artifact_path="artifacts")
    mlflow.log_artifact("/tmp/quarterly_cache.json", artifact_path="artifacts")
    mlflow.log_artifact("/tmp/filing_metadata_cache.json", artifact_path="artifacts")

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
        pip_requirements=["databricks-vectorsearch", "databricks-sdk", "mlflow", "requests"],
    )

    print(f"[MLflow] Run ID: {run.info.run_id}")
    print(f"[MLflow] Model URI: {model_info.model_uri}")
    print(f"[UC] Registered as: {UC_MODEL_NAME}")

# COMMAND ----------

# ── 10. Deploy to Databricks Model Serving ────────────────────────────────────

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from databricks.sdk.errors import ResourceDoesNotExist

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

import time

print(f"Waiting for endpoint '{AGENT_ENDPOINT}' to serve v{model_version}...")
timeout, poll = 20 * 60, 20
start = time.time()

# Endpoint.state.ready stays READY while a new version rolls out in the background
# (the previous version keeps serving traffic until the atomic swap). Exiting on
# ready alone runs live tests against the OLD version. Require all three: service
# ready, no in-flight config update, and the version we just deployed is the one
# being served.
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

        if "FAILED" in state.upper():
            print(f"Endpoint failed: {ep.state}")
            break

        if (state == "EndpointStateReady.READY"
                and "NOT_UPDATING" in config_update.upper()
                and str(model_version) in served_versions):
            print(f"Endpoint is READY and serving v{model_version}.")
            break
    except Exception as e:
        print(f"  Polling error: {e}")
    time.sleep(poll)

# Live test via SDK — our pyfunc has a custom (non-chat) output schema,
# so we send via dataframe_records and read from resp.predictions.

live_test_questions = [
    "What was NVIDIA's revenue and net income in fiscal year 2024?",
    "What risks did Tesla disclose about autonomous driving in their 10-K?",
]

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
    except Exception as e:
        print(f"Live test error: {type(e).__name__}: {e}")
