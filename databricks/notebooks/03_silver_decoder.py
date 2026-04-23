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

# MAGIC %pip install --quiet sec-parser "numpy<2"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ── A) XBRL CompanyFacts → financial_statements ─────────────────────────────
# Re-read widgets — restartPython() wipes Python state but widget values persist.
CATALOG       = dbutils.widgets.get("catalog")
ENV           = dbutils.widgets.get("env")
START_DATE    = dbutils.widgets.get("start_date")
TICKER_FILTER = dbutils.widgets.get("ticker_filter")
TICKER_SUBSET = [t.strip() for t in TICKER_FILTER.split(",") if t.strip()] if TICKER_FILTER else []

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

# ── B) 10-K / 10-Q section extraction → filing_sections ─────────────────────
# Primary:  sec-parser DOM-aware extraction. Operates on the iXBRL HTML tree,
#           so it handles inline-XBRL span fragmentation, decodes HTML entities
#           natively (&#160;, &#8217;, &mdash; …), drops page headers/footers
#           by semantic class, and cleanly separates Item headings from body.
# Fallback: the legacy regex extractor on entity-decoded flat text. Retained as
#           a safety net for any future filing shape sec-parser might refuse.
#
# Output schema (unchanged): section_id, filing_id, ticker, fiscal_year,
# filing_type, section_name, section_text, word_count, parsed_at.
# Merge key: section_id = sha256(filing_id || section_name).

from pyspark.sql.functions import (
    udf, col, current_timestamp, lit, row_number, sha2, concat_ws, explode,
    count as spark_count,
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    ArrayType, StructType, StructField, StringType, IntegerType,
)
from delta.tables import DeltaTable
import re

# Defensive widget re-read so Section B can be run in isolation after
# restartPython wipes state.
CATALOG = dbutils.widgets.get("catalog")


# ── Canonical section taxonomy (mirrors VALID_SECTION_NAMES in 06_rag_agent) ─
# min_words for 10-Q "Risk Factors Updates" is intentionally low: most filers
# use 1-2 sentence pro-forma "no material change since the 10-K" boilerplate
# when nothing has changed, and we want to capture those as valid sections.
CANONICAL_10K = {
    "Business":     {"item_re": re.compile(r"^\s*item\s*1\b(?![a-c])", re.I), "required": True,  "min_words": 200},
    "Risk Factors": {"item_re": re.compile(r"^\s*item\s*1a\b",         re.I), "required": True,  "min_words": 400},
    "MD&A":         {"item_re": re.compile(r"^\s*item\s*7\b(?!a)",      re.I), "required": True,  "min_words": 400},
}
CANONICAL_10Q = {
    "MD&A":                 {"item_re": re.compile(r"^\s*item\s*2\b(?!\s*a)", re.I), "required": True,  "min_words": 150},
    "Risk Factors Updates": {"item_re": re.compile(r"^\s*item\s*1a\b",        re.I), "required": False, "min_words": 10},
}
SECTIONS_BY_FORM = {"10-K": CANONICAL_10K, "10-Q": CANONICAL_10Q}


# ── Legacy regex rules retained for the fallback path only ──────────────────
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
SECTION_RULES_BY_FORM = {"10-K": SECTION_RULES_10K, "10-Q": SECTION_RULES_10Q}

SGML_DOC_RE = re.compile(r"<DOCUMENT>.*?</DOCUMENT>", re.DOTALL)


def _extract_main_doc(sgml_text, form_type):
    """Unwrap the SEC SGML submission and return the main iXBRL HTML document.

    A `full-submission.txt` can contain many <DOCUMENT> blocks (cover page,
    exhibits, XBRL schemas). We match by <TYPE> to pick the 10-K or 10-Q
    body deterministically — safer than the previous 'first document wins'
    heuristic, which could grab a cover-page doc before the main filing.
    """
    for block in SGML_DOC_RE.findall(sgml_text):
        m_type = re.search(r"<TYPE>([^\s<]+)", block)
        if not m_type or m_type.group(1).strip().upper() != form_type.upper():
            continue
        m_text = re.search(r"<TEXT>(.*?)</TEXT>", block, re.DOTALL)
        if m_text:
            body = m_text.group(1)
            body = re.sub(r"^\s*<XBRL>", "", body, count=1)
            body = re.sub(r"</XBRL>\s*$", "", body, count=1)
            return body.strip()
    return ""


def _sec_parser_extract(html, form_type):
    """DOM-aware section extraction via sec-parser.

    sec-parser v0.58 ships one Edgar10QParser that handles both forms; the
    10-K parse emits benign 'Invalid section type for part2itemN' warnings
    (its section-type enum is scoped to 10-Q Items 1-6) which we suppress —
    the elements are still classified correctly as TitleElement / TextElement.
    """
    import warnings
    import sec_parser as sp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        elements = sp.Edgar10QParser().parse(html)

    skip_types = (
        sp.PageHeaderElement,
        sp.PageNumberElement,
        sp.EmptyElement,
        sp.IrrelevantElement,
        sp.NotYetClassifiedElement,
    )
    item_re = re.compile(r"^\s*(?:part\s+i?i?i?\s+)?item\s*\d+[a-c]?\b", re.I)

    # sec-parser only promotes some Items to TopSectionTitle (Items 1, 2, 5, 6
    # for 10-K; all Items for 10-Q). For 10-K Items 1A / 7 / 7A etc. we scan
    # TitleElement as well. The union, ordered by doc position, is our anchor
    # list.
    headings = []
    for i, el in enumerate(elements):
        if isinstance(el, sp.TopSectionTitle):
            headings.append((i, el))
        elif isinstance(el, sp.TitleElement) and item_re.match(el.text or ""):
            headings.append((i, el))
    headings.sort(key=lambda pair: pair[0])

    def _find_heading(rule_re):
        # Prefer descriptive full-text headings over TOC-style stubs.
        # MSFT's 10-Q places a bare "Item 2" TopSectionTitle (6 chars) near the
        # top of the document ahead of the real "ITEM 2. MANAGEMENT'S DISCUSSION
        # AND ANALYSIS OF FINANCIAL CONDITION ..." TitleElement deeper down.
        # First-match-wins returned the stub, so `_body_between` stopped at the
        # next heading one element later and produced a near-empty body that
        # failed the 150-word minimum — silently dropping MD&A for every MSFT
        # 10-Q. Filtering anchors whose text is a trivially short stub (≤ 12
        # chars) lets us pick the descriptive heading whose body actually
        # starts the section. Falls back to the original behavior if no
        # substantive match exists so previously-working filers are unaffected.
        matches = [(i, el) for i, el in headings if rule_re.search(el.text or "")]
        if not matches:
            return None, None
        substantive = [m for m in matches if len((m[1].text or "").strip()) > 12]
        chosen = substantive[0] if substantive else matches[0]
        return chosen

    def _body_between(start_idx):
        out = []
        for j in range(start_idx + 1, len(elements)):
            el = elements[j]
            if isinstance(el, sp.TopSectionTitle):
                break
            if isinstance(el, sp.TitleElement) and item_re.match(el.text or ""):
                break
            if isinstance(el, skip_types):
                continue
            if hasattr(el, "text") and el.text:
                out.append(el.text.strip())
        return "\n".join(out).strip()

    results = []
    for name, rule in SECTIONS_BY_FORM.get(form_type, {}).items():
        idx, heading = _find_heading(rule["item_re"])
        if heading is None:
            continue
        body = _body_between(idx)
        wc = len(body.split())
        if wc < rule["min_words"]:
            continue
        results.append({"section_name": name, "section_text": body, "word_count": wc})
    return results


def _regex_fallback_extract(raw_text, form_type):
    """Legacy regex extractor on entity-decoded, tag-stripped flat text.

    Used as a tier-2 fill for any REQUIRED section sec-parser missed — not
    as a whole-filing fallback. The merge in `extract_sections` keeps
    sec-parser's cleaner wins and only drops this extractor's output in
    for the still-missing sections. Retains the page-footer / ToC-collision
    weaknesses of the pre-sec-parser implementation, but html.unescape()
    upfront makes it reliable enough to recover JPM / BAC / MA / V 10-K
    MD&A and Risk Factors where sec-parser's classifier skipped them.
    """
    import html as _html

    text = _html.unescape(raw_text or "")
    text = re.sub(r"(?is)<img[^>]*src=[\"']data:image/[^>]*>", " ", text)
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>",   " ", text)
    text = re.sub(
        r"(?i)</?(div|p|br|tr|li|table|tbody|thead|tfoot|td|th|h1|h2|h3|h4|h5|h6)[^>]*>",
        "\n",
        text,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\t\x0B\f\r ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    rules = SECTION_RULES_BY_FORM.get(form_type, {})
    results = []
    for name, rule in rules.items():
        starts = sorted({m.start() for p in rule["start_patterns"] for m in re.finditer(p, text)})
        ends   = sorted({m.start() for p in rule["end_patterns"]   for m in re.finditer(p, text)})
        if not starts:
            continue
        best, best_score = None, -1
        doc_len = max(len(text), 1)
        for s in starts:
            end_candidates = [e for e in ends if e > s + 25]
            e = end_candidates[0] if end_candidates else min(len(text), s + rule["fallback_chars"])
            candidate = text[s:e].strip()
            wc = len(candidate.split())
            if wc < rule["min_words"]:
                continue
            score = wc + ((s / doc_len) * 250)
            if score > best_score:
                best_score = score
                best = {"section_name": name, "section_text": candidate, "word_count": wc}
        if best:
            results.append(best)
    return results


def extract_sections(content_bytes, filing_type):
    """Row UDF: returns (sections[], error).

    Two-tier extraction with a **per-section partial-fallback merge**:

    1. Run sec-parser over the DOM and collect whatever sections it finds.
       Each section is tagged ``extractor_used = "sec-parser"``.
    2. If any REQUIRED section is still missing (or sec-parser yielded zero
       sections outright), run the regex fallback. Only sections that
       sec-parser did NOT already find get added; sec-parser wins ties.
       These additions carry ``extractor_used = "regex-fallback"``.

    Rationale. The previous all-or-nothing fallback only kicked in when
    sec-parser returned exactly zero sections. For financial-institution
    filers (JPM, BAC, MA, PFE, some of V) sec-parser correctly promoted
    "Item 1. Business" to a TopSectionTitle but its classifier missed
    "Item 1A" and "Item 7" — so sec-parser returned {Business} and the
    pipeline accepted that as "extraction succeeded", silently dropping
    MD&A and Risk Factors. Per-section merge preserves sec-parser's clean
    wins and only enlists the regex fallback for the sections still missing,
    which is where its page-footer / ToC collision weaknesses do the least
    damage.

    Failure vocabulary (for `ingestion_errors.error_message`):
      * "Empty content"                        — null bronze row
      * "Unsupported filing_type: ..."         — non 10-K/10-Q
      * "Main iXBRL document not found in SGML wrapper"
                                               — full-submission.txt is
                                                 malformed or truncated
                                                 (see DDOG 2022-Q1)
      * "No sections found by either extractor"
                                               — structurally bare filing
                                                 (see MCD, which incorporates
                                                 its content from Part III /
                                                 proxy by reference)
      * "Missing required sections: X, Y"      — partial hit; after both
                                                 tiers ran, X and Y are
                                                 still absent
    """
    if content_bytes is None:
        return ([], "Empty content")
    try:
        raw = (
            content_bytes.decode("utf-8", errors="replace")
            if isinstance(content_bytes, (bytes, bytearray))
            else str(content_bytes)
        )
    except Exception as e:
        return ([], f"Decode error: {e}")

    if filing_type not in SECTIONS_BY_FORM:
        return ([], f"Unsupported filing_type: {filing_type}")

    html = _extract_main_doc(raw, filing_type)
    if not html:
        return ([], "Main iXBRL document not found in SGML wrapper")

    required = {n for n, r in SECTIONS_BY_FORM[filing_type].items() if r["required"]}
    merged = {}  # section_name -> dict(section_name, section_text, word_count, extractor_used)

    # Tier 1: sec-parser (DOM-aware)
    sec_parser_err = None
    try:
        for s in _sec_parser_extract(html, filing_type):
            s["extractor_used"] = "sec-parser"
            merged[s["section_name"]] = s
    except Exception as e:
        sec_parser_err = f"sec-parser error: {type(e).__name__}: {e}"

    # Tier 2: regex fallback, but only for sections sec-parser missed.
    # Runs when any required section is still absent OR when sec-parser
    # produced nothing at all — the latter keeps the old "zero-sections"
    # safety net in place for edge cases.
    fallback_err = None
    if (required - set(merged.keys())) or not merged:
        try:
            for s in _regex_fallback_extract(raw, filing_type):
                if s["section_name"] not in merged:
                    s["extractor_used"] = "regex-fallback"
                    merged[s["section_name"]] = s
        except Exception as e:
            fallback_err = f"regex-fallback error: {type(e).__name__}: {e}"

    sections = list(merged.values())
    if not sections:
        err = sec_parser_err or "No sections found by either extractor"
        if fallback_err:
            err = f"{err}; {fallback_err}"
        return ([], err)

    still_missing = sorted(required - set(merged.keys()))
    err = f"Missing required sections: {', '.join(still_missing)}" if still_missing else None
    return (sections, err)


# UDF schema carries `extractor_used` *inside* each section struct so the
# attribution is row-level on silver (not per-filing). With per-section merge
# a single filing can contribute both sec-parser and regex-fallback rows, and
# we want the chunker / VS index to be able to filter by extractor at section
# granularity.
extract_udf_schema = StructType([
    StructField("sections", ArrayType(StructType([
        StructField("section_name",   StringType()),
        StructField("section_text",   StringType()),
        StructField("word_count",     IntegerType()),
        StructField("extractor_used", StringType()),
    ]))),
    StructField("error", StringType()),
])
extract_udf = udf(extract_sections, extract_udf_schema)

df_bronze = (
    spark.table(f"{CATALOG}.finsage_bronze.filings")
    .filter(col("filing_type").isin("10-K", "10-Q"))
    .withColumn("rn", row_number().over(
        Window.partitionBy("filing_id").orderBy(col("ingested_at").desc())
    ))
    .filter(col("rn") == 1)
    .drop("rn")
)

# The UDF now consumes raw BINARY content directly — all SGML unwrap, entity
# decoding, and iXBRL tree parsing happens inside sec-parser, so the earlier
# chain of Spark SQL regexp_replace cleaning steps is gone.
df_extracted = (
    df_bronze
    .withColumn("udf_result", extract_udf(col("content"), col("filing_type")))
    .select(
        "filing_id", "ticker", "fiscal_year", "filing_type", "file_path",
        col("udf_result.sections").alias("sections"),
        col("udf_result.error").alias("error"),
    )
)
df_extracted.cache()

# Observability: section-level breakdown (not filing-level, because a single
# filing can now produce a mix — sec-parser wins for the sections it finds,
# regex-fallback fills the still-missing required sections). explode on a
# zero-length array drops the row cleanly, so error-only filings are excluded.
extractor_stats = (
    df_extracted
    .withColumn("sec", explode("sections"))
    .groupBy(col("sec.extractor_used").alias("extractor_used"))
    .agg(spark_count("*").alias("n_sections"))
    .collect()
)
if extractor_stats:
    print("[EXTRACTOR USAGE — per section]")
    for row in extractor_stats:
        print(f"  {row['extractor_used']:<16}: {row['n_sections']:>5} sections")

df_errors = df_extracted.filter(col("error").isNotNull())
n_errors = df_errors.count()
if n_errors > 0:
    print(f"Warning: {n_errors} filings hit section extraction errors. Logging to ingestion_errors.")
    (
        df_errors.select(
            sha2(concat_ws("||", col("file_path"), col("error")), 256).alias("error_id"),
            lit("silver_section_extraction").alias("source_system"),
            lit(None).cast("string").alias("source_url"),
            col("file_path"),
            lit("parse_failure").alias("error_type"),
            col("error").alias("error_message"),
            lit(0).alias("retry_count"),
            current_timestamp().alias("failed_at"),
        )
        .write.format("delta").mode("append")
        .saveAsTable(f"{CATALOG}.finsage_bronze.ingestion_errors")
    )

df_final_sections = (
    df_extracted.filter(col("sections").isNotNull())
    .withColumn("sec", explode("sections"))
    .select(
        # filing_id is already unique per SEC accession, so
        # (filing_id, section_name) yields a unique section_id across 10-K/10-Q.
        sha2(concat_ws("||", col("filing_id"), col("sec.section_name")), 256).alias("section_id"),
        "filing_id", "ticker", "fiscal_year", "filing_type",
        col("sec.section_name").alias("section_name"),
        col("sec.section_text").alias("section_text"),
        col("sec.word_count").alias("word_count"),
        # Per-row extractor attribution — lets downstream (chunker, VS index,
        # audit SQL) filter by extractor_used. With autoMerge enabled below,
        # the column lands on an existing silver table without a rebuild;
        # rows that predate this column stay NULL and are cleanly identifiable
        # as stale-regex-era survivors.
        col("sec.extractor_used").alias("extractor_used"),
        current_timestamp().alias("parsed_at"),
    )
)

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

df_extracted.unpersist()

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
