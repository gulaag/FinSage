"""Backend data access for the FinSage chat app.

Two responsibilities:
  1. query_agent()        — send a chat-completion-shaped payload to the
                            FinSage Model Serving endpoint and parse the
                            response into typed AgentResponse / Citation objects.
  2. fetch_section_text() — pull the full clean section text from
                            main.finsage_silver.filing_sections, used by tab 2
                            to render the source modal with the chunk
                            highlighted in its surrounding context.

Auth: relies on app-level service principal credentials injected by Databricks
Apps as DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET. WorkspaceClient and
the SQL connector both auto-detect via Config().
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Optional

from databricks import sql as dbsql
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from pydantic import BaseModel, Field

log = logging.getLogger("finsage-app.backend")

AGENT_ENDPOINT          = os.getenv("FINSAGE_AGENT_ENDPOINT", "finsage_agent_endpoint")
WAREHOUSE_ID            = os.getenv("DATABRICKS_WAREHOUSE_ID")
CATALOG                 = os.getenv("FINSAGE_CATALOG", "main")
SECTIONS_TABLE          = f"{CATALOG}.finsage_silver.filing_sections"
ANNUAL_METRICS_TABLE    = f"{CATALOG}.finsage_gold.company_metrics"
QUARTERLY_METRICS_TABLE = f"{CATALOG}.finsage_gold.company_metrics_quarterly"


# ─────────────────────────────────────────────────────────────────────────────
# Typed response models
# ─────────────────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """One retrieved chunk surfaced by the agent's search_filings tool."""
    ticker:       Optional[str]   = None
    fiscal_year:  Optional[int]   = None
    filing_type:  Optional[str]   = None  # "10-K" or "10-Q"
    section_name: Optional[str]   = None
    chunk_text:   str             = ""
    score:        Optional[float] = None

    @property
    def label(self) -> str:
        """Compact label for sidebar / chip rendering."""
        parts = [self.ticker or "?"]
        if self.fiscal_year is not None:
            parts.append(f"FY{int(self.fiscal_year)}")
        if self.filing_type:
            parts.append(self.filing_type)
        if self.section_name:
            parts.append(self.section_name)
        return " | ".join(parts)


class AgentResponse(BaseModel):
    """Parsed response from the FinSage Model Serving endpoint."""
    content:   str             = ""
    citations: list[Citation]  = Field(default_factory=list)
    # The agent's raw message history (tool calls + tool results + final
    # assistant turn). Used to drive the design-system agent-timeline card
    # so each tool invocation renders as a step.
    messages:  list[dict]      = Field(default_factory=list)
    error:     Optional[str]   = None


# ─────────────────────────────────────────────────────────────────────────────
# Singleton clients (Reflex re-imports state classes per request, so we
# memoize the heavy SDK clients at module scope)
# ─────────────────────────────────────────────────────────────────────────────

_workspace_client: Optional[WorkspaceClient] = None
_response_cache: "OrderedDict[str, AgentResponse]" = OrderedDict()
_response_cache_max = 256


def _get_workspace_client() -> WorkspaceClient:
    global _workspace_client
    if _workspace_client is None:
        _workspace_client = WorkspaceClient()
    return _workspace_client


def _is_likely_qualitative_prompt(text: str) -> bool:
    if not text:
        return False
    lo = text.lower()
    qual_markers = (
        "risk", "supply chain", "10-k", "10q", "10-q", "md&a",
        "business", "strategy", "competition", "regulation", "disclose",
    )
    numeric_markers = (
        "revenue", "net income", "operating income", "gross margin",
        "debt", "equity", "cash flow", "ratio", "growth", "yoy", "fy",
        "q1", "q2", "q3", "q4",
    )
    has_qual = any(m in lo for m in qual_markers)
    has_numeric = any(m in lo for m in numeric_markers)
    # Explicit risk/10-K narrative questions should route through retrieval even
    # if they mention a year token.
    if "risk" in lo or "10-k" in lo:
        return True
    return has_qual and not has_numeric


def _massage_messages_for_routing(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages
    wire = [dict(m) for m in messages]
    last = wire[-1]
    if (
        isinstance(last, dict)
        and last.get("role") == "user"
        and isinstance(last.get("content"), str)
        and _is_likely_qualitative_prompt(last.get("content") or "")
    ):
        # Lightweight routing hint: keeps user intent intact but nudges the
        # downstream tool loop toward narrative filing retrieval.
        if not re.search(r"search_filings", last["content"], flags=re.IGNORECASE):
            last["content"] = (
                last["content"].rstrip()
                + "\n\n[Routing hint: This is a qualitative filing-text question. "
                  "Use search_filings over 10-K/10-Q narrative sections, not metrics-only output.]"
            )
    return wire


def _tool_outputs_from_messages(msgs: list[dict]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pending: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function") or {}
                pending.append(str(fn.get("name") or "tool"))
        elif m.get("role") == "tool" and pending:
            name = pending.pop(0)
            out.append((name, str(m.get("content") or "")))
    return out


def _render_metadata_answer_from_tool(content: str, user_text: str) -> Optional[str]:
    if not content:
        return None
    no_data = re.search(
        r"No filing metadata(?: for ticker)?\s+'?([A-Z]{1,8})'?(?:\s+in\s+FY(\d{4}))?",
        content,
        flags=re.IGNORECASE,
    )
    if no_data:
        ticker = (no_data.group(1) or "").upper() or "This ticker"
        fy = no_data.group(2)
        period = f" in FY{fy}" if fy else ""
        return (
            f"{ticker} does not have available 10-K cover-page metadata{period} "
            "in the current FinSage metadata cache."
        )

    m = re.search(r"10-K metadata for\s+([A-Z]{1,8})\s+FY(\d{4})", content)
    if not m:
        return None
    ticker, fy = m.group(1), m.group(2)
    filing_date = None
    employees = None
    shares = None
    for ln in content.splitlines():
        ln = ln.strip()
        if ln.lower().startswith("filing date:"):
            filing_date = ln.split(":", 1)[1].strip()
        elif ln.lower().startswith("employees:"):
            employees = ln.split(":", 1)[1].strip()
        elif ln.lower().startswith("shares outstanding:"):
            shares = ln.split(":", 1)[1].strip()

    asks_employees = "employee" in user_text.lower()
    asks_shares = "shares" in user_text.lower() or "outstanding" in user_text.lower()
    asks_filing_date = "filing date" in user_text.lower() or "filed" in user_text.lower()

    def _fmt_int_like(s: Optional[str]) -> Optional[str]:
        if not s or s.upper() == "N/A":
            return s
        if re.fullmatch(r"\d+", s):
            return f"{int(s):,}"
        return s

    employees = _fmt_int_like(employees)
    shares = _fmt_int_like(shares)

    if asks_employees:
        if not employees or employees.upper() == "N/A":
            headline = f"{ticker}'s FY{fy} employee count is unavailable in the 10-K cover-page metadata (N/A)."
        else:
            headline = f"{ticker} reported {employees} employees in FY{fy}."
    elif asks_shares:
        if not shares or shares.upper() == "N/A":
            headline = f"{ticker}'s FY{fy} shares-outstanding value is unavailable in the 10-K cover-page metadata (N/A)."
        else:
            headline = f"{ticker} reported {shares} shares outstanding in FY{fy}."
    elif asks_filing_date:
        if not filing_date or filing_date.upper() == "N/A":
            headline = f"{ticker}'s FY{fy} filing date is unavailable in the 10-K cover-page metadata (N/A)."
        else:
            headline = f"{ticker}'s FY{fy} 10-K filing date is {filing_date}."
    else:
        headline = f"Here are {ticker}'s FY{fy} 10-K cover-page metadata values."

    details = []
    if filing_date:
        details.append(f"Filing Date: {filing_date}")
    if employees:
        details.append(f"Employees: {employees}")
    if shares:
        details.append(f"Shares Outstanding: {shares}")

    body = ""
    if details:
        body = "\n\n" + "\n".join(f"- {d}" for d in details)
    return f"{headline}{body}\n\n[Source: {ticker} | FY{fy} | 10-K Cover Page]"


# ─────────────────────────────────────────────────────────────────────────────
# Agent endpoint query
# ─────────────────────────────────────────────────────────────────────────────

def query_agent(messages: list[dict]) -> AgentResponse:
    """POST a chat-completion-style payload to finsage_agent_endpoint.

    The agent (v22+) accepts {"messages": [{role, content}, ...]} and returns
    {"content": str, "messages": list, "citations"?: list[dict]}. v22 omits
    the citations key — we tolerate both shapes.
    """
    if not messages:
        return AgentResponse(content="", error="No messages to send.")
    wire_messages = _massage_messages_for_routing(messages)

    # Stabilize UX for repeated identical prompts: if the same message history
    # is sent again, return the same parsed response from process-local cache.
    # This avoids visible response drift from decoder randomness between retries.
    try:
        cache_key = json.dumps(wire_messages, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        cache_key = ""
    if cache_key:
        cached = _response_cache.get(cache_key)
        if cached is not None:
            try:
                _response_cache.move_to_end(cache_key)
            except Exception:
                pass
            try:
                return cached.model_copy(deep=True)
            except AttributeError:
                return AgentResponse(**cached.dict())

    latest_user_text = ""
    for m in reversed(wire_messages):
        if isinstance(m, dict) and m.get("role") == "user":
            latest_user_text = str(m.get("content") or "")
            break
    qualitative_prompt = _is_likely_qualitative_prompt(latest_user_text)

    def _parse_prediction(pred_obj: Any) -> AgentResponse:
        if not isinstance(pred_obj, dict):
            return AgentResponse(content=str(pred_obj or ""), error=None)

        content = str(pred_obj.get("content", "") or "")
        raw_citations = pred_obj.get("citations") or []
        citations: list[Citation] = []
        for c in raw_citations:
            if not isinstance(c, dict):
                continue
            try:
                citations.append(Citation(**{k: c.get(k) for k in Citation.model_fields}))
            except Exception:
                log.warning("Skipped malformed citation: %s", c)

        raw_messages = pred_obj.get("messages") or []
        msgs: list[dict] = [m for m in raw_messages if isinstance(m, dict)]
        return AgentResponse(content=content, citations=citations, messages=msgs)

    def _used_search_filings(msgs: list[dict]) -> bool:
        for m in msgs:
            if not isinstance(m, dict):
                continue
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function") or {}
                if fn.get("name") == "search_filings":
                    return True
        return False

    try:
        client = _get_workspace_client()
        resp = client.serving_endpoints.query(
            name=AGENT_ENDPOINT,
            dataframe_records=[{"messages": wire_messages}],
        )
    except Exception as exc:
        log.exception("Agent endpoint query failed.")
        return AgentResponse(content="", error=f"Endpoint call failed: {exc}")

    raw = resp.predictions
    if isinstance(raw, list) and raw:
        pred = raw[0]
    else:
        pred = raw

    parsed = _parse_prediction(pred)

    # Qualitative guardrail: if the model ignored retrieval and answered with
    # metrics-only tools, force one retry with an explicit retrieval directive.
    if qualitative_prompt and not _used_search_filings(parsed.messages):
        retry_messages = wire_messages + [{
            "role": "user",
            "content": (
                "Important: answer this as a qualitative filing-text question. "
                "Use search_filings over 10-K/10-Q narrative sections and cite "
                "[Source: TICKER | FY#### | 10-K/10-Q | Section]. Do not answer "
                "with metrics-only tools."
            ),
        }]
        try:
            resp2 = client.serving_endpoints.query(
                name=AGENT_ENDPOINT,
                dataframe_records=[{"messages": retry_messages}],
            )
            raw2 = resp2.predictions
            pred2 = raw2[0] if isinstance(raw2, list) and raw2 else raw2
            parsed2 = _parse_prediction(pred2)
            if parsed2.content and (_used_search_filings(parsed2.messages) or parsed2.citations):
                parsed = parsed2
        except Exception:
            log.warning("Qualitative retry failed; using first response.", exc_info=True)

    # Metadata guardrail: keep cover-page Q&A concise and deterministic from
    # tool output, avoiding occasional "briefing template" spillover.
    outputs = _tool_outputs_from_messages(parsed.messages)
    if outputs and all(name == "get_filing_metadata" for name, _ in outputs):
        rendered = _render_metadata_answer_from_tool(outputs[-1][1], latest_user_text)
        if rendered:
            parsed.content = rendered

    # Cache successful responses only (never cache endpoint errors).
    if cache_key and not parsed.error:
        _response_cache[cache_key] = parsed
        try:
            _response_cache.move_to_end(cache_key)
            while len(_response_cache) > _response_cache_max:
                _response_cache.popitem(last=False)
        except Exception:
            pass

    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Section text lookup (tier-2 modal source)
# ─────────────────────────────────────────────────────────────────────────────

def _get_sql_connection():
    """Build a fresh DBSQL connection. Connections are cheap relative to
    cold-start latency, but we keep a simple LRU on the result so a single
    section text fetch doesn't requery on every modal open."""
    if not WAREHOUSE_ID:
        raise RuntimeError(
            "DATABRICKS_WAREHOUSE_ID is not set — add a sql-warehouse resource "
            "to the Databricks App config (see app.yaml)."
        )
    cfg = Config()
    return dbsql.connect(
        server_hostname    = cfg.host,
        http_path          = f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        credentials_provider = lambda: cfg.authenticate,
    )


@lru_cache(maxsize=512)
def fetch_section_text(
    ticker: str,
    fiscal_year: int,
    filing_type: Optional[str],
    section_name: str,
) -> Optional[str]:
    """Return the full clean section text for one (ticker, fy, filing_type,
    section) combination. None if not found.

    Cached — sections are immutable once landed in silver, so a process-
    lifetime cache is correct."""
    if not (ticker and section_name):
        return None
    try:
        conn = _get_sql_connection()
    except Exception as exc:
        log.warning("Cannot reach SQL warehouse: %s", exc)
        return None

    where = ["upper(ticker) = upper(%(ticker)s)",
             "fiscal_year   = %(fy)s",
             "section_name  = %(section)s",
             "extractor_used IS NOT NULL"]
    params: dict[str, Any] = {
        "ticker":  ticker,
        "fy":      int(fiscal_year),
        "section": section_name,
    }
    if filing_type:
        where.append("filing_type = %(ft)s")
        params["ft"] = filing_type

    sql = (
        f"SELECT section_text "
        f"FROM {SECTIONS_TABLE} "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY length(section_text) DESC "
        f"LIMIT 1"
    )

    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
    except Exception as exc:
        log.warning("filing_sections lookup failed for %s FY%s %s/%s: %s",
                    ticker, fiscal_year, filing_type, section_name, exc)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not row:
        return None
    text = row[0]
    return text if isinstance(text, str) else None


# ─────────────────────────────────────────────────────────────────────────────
# Structured metrics lookup (tier-2 modal source for [Source: ... | metrics])
# ─────────────────────────────────────────────────────────────────────────────

_METRIC_COLUMNS = [
    "ticker",
    "company_name",
    "fiscal_year",
    "fiscal_quarter",
    "revenue",
    "net_income",
    "gross_profit",
    "operating_income",
    "operating_cash_flow",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "total_debt",
    "rd_expense",
    "gross_margin_pct",
    "revenue_yoy_growth_pct",
    "debt_to_equity",
    "data_quality_score",
]


@lru_cache(maxsize=1024)
def fetch_metrics_row(
    ticker: str,
    fiscal_year: int,
    fiscal_quarter: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """Return one structured metrics row for the given period.

    - fiscal_quarter=None -> annual table (company_metrics, fiscal_quarter IS NULL)
    - fiscal_quarter=1..4 -> quarterly table (company_metrics_quarterly)
    """
    if not ticker or fiscal_year is None:
        return None
    try:
        conn = _get_sql_connection()
    except Exception as exc:
        log.warning("Cannot reach SQL warehouse for metrics lookup: %s", exc)
        return None

    table = QUARTERLY_METRICS_TABLE if fiscal_quarter is not None else ANNUAL_METRICS_TABLE
    cols_sql = ", ".join(_METRIC_COLUMNS)
    where = [
        "upper(ticker) = upper(%(ticker)s)",
        "fiscal_year = %(fy)s",
    ]
    params: dict[str, Any] = {
        "ticker": ticker,
        "fy": int(fiscal_year),
    }
    if fiscal_quarter is None:
        where.append("fiscal_quarter IS NULL")
    else:
        where.append("fiscal_quarter = %(fq)s")
        params["fq"] = int(fiscal_quarter)

    sql = (
        f"SELECT {cols_sql} "
        f"FROM {table} "
        f"WHERE {' AND '.join(where)} "
        f"LIMIT 1"
    )

    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
    except Exception as exc:
        log.warning(
            "metrics lookup failed for %s FY%s Q%s: %s",
            ticker, fiscal_year, fiscal_quarter, exc
        )
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not row:
        return None

    out: dict[str, Any] = {}
    for i, col_name in enumerate(_METRIC_COLUMNS):
        out[col_name] = row[i] if i < len(row) else None
    out["source_table"] = table
    return out
