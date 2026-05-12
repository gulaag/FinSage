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
CORPUS_MIN_FY = 2020
CORPUS_MAX_FY = 2026


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


def _normalize_source_labels_in_content(content: str) -> str:
    if not content:
        return content
    re_src = re.compile(r"\[Source:\s*([^\]]+?)\s*\]", flags=re.IGNORECASE)

    def _fix(match: re.Match[str]) -> str:
        raw = (match.group(1) or "").strip()
        parts = [p.strip() for p in raw.split("|") if p and p.strip()]
        if not parts:
            return match.group(0)
        ticker = (parts[0] or "").upper()
        fy = ""
        filing_type = ""
        metric_kind = False
        section_name = ""
        for p in parts[1:]:
            if re.fullmatch(r"FY\d{4}(?:\s*Q[1-4])?", p, flags=re.IGNORECASE):
                fy = p.upper().replace("  ", " ")
            elif re.fullmatch(r"10-[KQ]", p, flags=re.IGNORECASE):
                filing_type = p.upper()
            elif p.lower() == "metrics":
                metric_kind = True
            elif p.lower() == "10-k cover page":
                filing_type = filing_type or "10-K"
                section_name = "Cover Page"
            elif p in {"Business", "Risk Factors", "MD&A", "Risk Factors Updates"}:
                section_name = p
        if metric_kind and not filing_type:
            filing_type = "10-Q" if "Q" in fy else "10-K"
        out = [ticker]
        if fy:
            out.append(fy)
        if filing_type:
            out.append(filing_type)
        if metric_kind:
            out.append("metrics")
        elif section_name == "Cover Page":
            out.append("Cover Page")
        elif section_name:
            out.append(section_name)
        if len(out) < 2:
            return match.group(0)
        return f"[Source: {' | '.join(out)}]"

    return re_src.sub(_fix, content)


def _render_metrics_answer_from_tool(content: str, user_text: str, tool_name: str) -> Optional[str]:
    if not content:
        return None
    if tool_name not in {"get_company_metrics", "get_quarterly_metrics"}:
        return None
    ticker_match = re.search(r"for\s+([A-Z]{1,8})\s*\(", content)
    ticker = (ticker_match.group(1) if ticker_match else "").upper()
    if not ticker:
        return None

    asks = user_text.lower()
    wants_revenue = "revenue" in asks
    wants_net_income = "net income" in asks
    wants_margin = "margin" in asks
    wants_debt = "debt" in asks or "equity" in asks or "ratio" in asks
    wants_growth = "growth" in asks or "yoy" in asks

    if tool_name == "get_quarterly_metrics":
        block_re = re.compile(r"FY(\d{4})\s+Q([1-4])\s+\(period ending ([^)]+)\):([\s\S]*?)(?=\nFY\d{4}\s+Q[1-4]\s+\(|\Z)")
        blocks = list(block_re.finditer(content))
        if not blocks:
            return None
        fyq_need = re.search(r"FY(\d{4})\s*Q([1-4])", asks.upper())
        chosen = None
        if fyq_need:
            fy_need = int(fyq_need.group(1))
            q_need = int(fyq_need.group(2))
            for b in blocks:
                if int(b.group(1)) == fy_need and int(b.group(2)) == q_need:
                    chosen = b
                    break
        if chosen is None:
            chosen = blocks[-1]
        fy = int(chosen.group(1))
        fq = int(chosen.group(2))
        period_end = chosen.group(3).strip()
        payload = chosen.group(4)
        fields = {}
        for ln in payload.splitlines():
            if ":" not in ln:
                continue
            k, v = ln.split(":", 1)
            fields[k.strip().lower()] = v.strip()
        rev = fields.get("revenue", "N/A")
        ni = fields.get("net income", "N/A")
        op = fields.get("operating income", "N/A")
        margin = fields.get("gross margin", "N/A")
        yoy = fields.get("revenue yoy (same q)", "N/A")
        de = fields.get("debt/equity", "N/A")
        lead = f"{ticker} reported {rev} of revenue in FY{fy} Q{fq} (period ending {period_end})."
        context_bits = []
        if wants_net_income or (not wants_margin and not wants_debt and not wants_growth):
            context_bits.append(f"Net income was {ni}, and operating income was {op}.")
        if wants_margin or not wants_net_income:
            context_bits.append(f"Gross margin was {margin}, with same-quarter YoY revenue growth at {yoy}.")
        if wants_debt:
            context_bits.append(f"Debt-to-equity was {de}.")
        prose = " ".join(context_bits[:2]).strip()
        source = f"[Source: {ticker} | FY{fy} Q{fq} | 10-Q | metrics]"
        return f"{lead}\n\n{prose}\n\n{source}" if prose else f"{lead}\n\n{source}"

    block_re = re.compile(r"FY(\d{4}):([\s\S]*?)(?=\nFY\d{4}:|\Z)")
    blocks = list(block_re.finditer(content))
    if not blocks:
        return None
    fy_need = re.search(r"FY\s*(\d{4})|\b(20\d{2})\b", asks.upper())
    chosen = None
    if fy_need:
        fy_target = int(next(g for g in fy_need.groups() if g))
        for b in blocks:
            if int(b.group(1)) == fy_target:
                chosen = b
                break
    if chosen is None:
        chosen = blocks[-1]
    fy = int(chosen.group(1))
    payload = chosen.group(2)
    fields = {}
    for ln in payload.splitlines():
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        fields[k.strip().lower()] = v.strip()
    rev = fields.get("revenue", "N/A")
    ni = fields.get("net income", "N/A")
    op = fields.get("operating income", "N/A")
    margin = fields.get("gross margin", "N/A")
    yoy = fields.get("revenue yoy growth", "N/A")
    de = fields.get("debt/equity", "N/A")

    if wants_net_income and not wants_revenue:
        lead = f"{ticker}'s net income in FY{fy} was {ni}."
    elif wants_margin and not wants_revenue and not wants_net_income:
        lead = f"{ticker}'s gross margin in FY{fy} was {margin}."
    elif wants_debt and not wants_revenue and not wants_net_income:
        lead = f"{ticker}'s debt-to-equity ratio in FY{fy} was {de}."
    else:
        lead = f"{ticker}'s total revenue in FY{fy} was {rev}."

    context = []
    context.append(f"Net income was {ni}, and operating income was {op}.")
    if wants_growth or wants_revenue:
        context.append(f"Gross margin was {margin}, and revenue YoY growth was {yoy}.")
    if wants_debt:
        context.append(f"Debt-to-equity was {de}.")
    source = f"[Source: {ticker} | FY{fy} | 10-K | metrics]"
    return f"{lead}\n\n{' '.join(context[:2])}\n\n{source}"


def _render_metrics_no_data_answer(content: str, user_text: str) -> Optional[str]:
    if not content:
        return None

    m = re.search(r"No data for\s+([A-Z]{1,8})\s+in the requested fiscal year range\.", content, flags=re.IGNORECASE)
    if m:
        ticker = m.group(1).upper()
        year_m = re.search(r"\b(20\d{2})\b", user_text)
        if year_m:
            fy = year_m.group(1)
            return (
                f"[NON-FILING-FALLBACK] Caution: FinSage filing-backed coverage does not include {ticker} FY{fy} "
                f"(current corpus window is FY2020-FY2026). "
                f"If useful, I can provide best-effort general historical context as non-filing information.\n\n"
                "[Source: NON-FILING | FinSage filings do not cover this requested period | Due diligence required]"
            )
        return (
            f"[NON-FILING-FALLBACK] Caution: I don't have filing-backed {ticker} data for that requested fiscal period "
            "in this corpus (FY2020-FY2026).\n\n"
            "[Source: NON-FILING | FinSage filings do not cover this requested period | Due diligence required]"
        )

    m = re.search(r"No quarterly data for\s+([A-Z]{1,8})\s+matching the requested filters\.", content, flags=re.IGNORECASE)
    if m:
        ticker = m.group(1).upper()
        return (
            f"[NON-FILING-FALLBACK] Caution: I don't have a filing-backed quarterly row for {ticker} with those filters. "
            f"Try a different FY/Q combination within the available corpus period (FY2020+).\n\n"
            "[Source: NON-FILING | FinSage filings do not cover this requested period | Due diligence required]"
        )

    m = re.search(r"No metrics found for ticker\s+'?([A-Z]{1,8})'?", content, flags=re.IGNORECASE)
    if m:
        ticker = m.group(1).upper()
        return (
            f"[NON-FILING-FALLBACK] Caution: {ticker} is not in the current FinSage filing-backed metrics coverage set.\n\n"
            "[Source: NON-FILING | FinSage filings do not cover this requested period | Due diligence required]"
        )

    return None


def _force_non_filing_fallback_if_outside_corpus(content: str, user_text: str) -> Optional[str]:
    if not user_text:
        return None
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", user_text)]
    if not years:
        return None
    outside = [y for y in years if y < CORPUS_MIN_FY or y > CORPUS_MAX_FY]
    if not outside:
        return None
    target_fy = outside[0]
    cleaned = re.sub(r"\[Source:\s*[^\]]+\]\s*", "", content or "", flags=re.IGNORECASE).strip()
    if not cleaned:
        cleaned = "I can provide only best-effort general context for this period."
    return (
        f"[NON-FILING-FALLBACK] Caution: FinSage filing-backed coverage is FY{CORPUS_MIN_FY}-FY{CORPUS_MAX_FY}, "
        f"so FY{target_fy} is outside direct SEC-filing coverage in this demo.\n\n"
        f"{cleaned}\n\n"
        "[Source: NON-FILING | FinSage filings do not cover this requested period | Due diligence required]"
    )


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

    outputs = _tool_outputs_from_messages(parsed.messages)
    if not outputs:
        forced = _force_non_filing_fallback_if_outside_corpus(parsed.content or "", latest_user_text)
        if forced:
            parsed.content = forced

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
    is_non_filing_fallback = bool(re.search(r"\[NON-FILING-FALLBACK\]", parsed.content or "", flags=re.IGNORECASE))

    if outputs and all(name == "get_filing_metadata" for name, _ in outputs):
        rendered = _render_metadata_answer_from_tool(outputs[-1][1], latest_user_text)
        if rendered and not is_non_filing_fallback:
            parsed.content = rendered
    elif outputs and all(name in {"get_company_metrics", "get_quarterly_metrics"} for name, _ in outputs):
        no_data = _render_metrics_no_data_answer(outputs[-1][1], latest_user_text)
        if no_data and not is_non_filing_fallback:
            parsed.content = no_data
        else:
            rendered = _render_metrics_answer_from_tool(outputs[-1][1], latest_user_text, outputs[-1][0])
            if rendered and (len((parsed.content or "").strip()) < 320):
                parsed.content = rendered

    parsed.content = _normalize_source_labels_in_content(parsed.content or "")

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


def fetch_section_text(
    ticker: str,
    fiscal_year: int,
    filing_type: Optional[str],
    section_name: str,
) -> Optional[str]:
    """Return the full clean section text for one (ticker, fy, filing_type,
    section) combination. None if not found.

    No in-process cache — always re-query to guarantee fresh behavior."""
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
    if fiscal_quarter is not None:
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

    if not row and fiscal_quarter is None:
        sql_fallback = (
            f"SELECT {cols_sql} "
            f"FROM {table} "
            f"WHERE upper(ticker) = upper(%(ticker)s) AND fiscal_year = %(fy)s "
            f"LIMIT 1"
        )
        conn2 = None
        try:
            conn2 = _get_sql_connection()
            with conn2.cursor() as cur:
                cur.execute(sql_fallback, {"ticker": ticker, "fy": int(fiscal_year)})
                row = cur.fetchone()
        except Exception:
            row = None
        finally:
            try:
                conn2.close()
            except Exception:
                pass

    if not row:
        return None

    out: dict[str, Any] = {}
    for i, col_name in enumerate(_METRIC_COLUMNS):
        out[col_name] = row[i] if i < len(row) else None
    out["source_table"] = table
    return out
