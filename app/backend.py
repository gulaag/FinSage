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

import logging
import os
from functools import lru_cache
from typing import Any, Optional

from databricks import sql as dbsql
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from pydantic import BaseModel, Field

log = logging.getLogger("finsage-app.backend")

AGENT_ENDPOINT = os.getenv("FINSAGE_AGENT_ENDPOINT", "finsage_agent_endpoint")
WAREHOUSE_ID   = os.getenv("DATABRICKS_WAREHOUSE_ID")
CATALOG        = os.getenv("FINSAGE_CATALOG", "main")
SECTIONS_TABLE = f"{CATALOG}.finsage_silver.filing_sections"


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

    try:
        client = _get_workspace_client()
        resp = client.serving_endpoints.query(
            name=AGENT_ENDPOINT,
            dataframe_records=[{"messages": messages}],
        )
    except Exception as exc:
        log.exception("Agent endpoint query failed.")
        return AgentResponse(content="", error=f"Endpoint call failed: {exc}")

    raw = resp.predictions
    if isinstance(raw, list) and raw:
        pred = raw[0]
    else:
        pred = raw

    if not isinstance(pred, dict):
        return AgentResponse(content=str(pred or ""), error=None)

    content = str(pred.get("content", "") or "")
    raw_citations = pred.get("citations") or []
    citations: list[Citation] = []
    for c in raw_citations:
        if not isinstance(c, dict):
            continue
        try:
            citations.append(Citation(**{k: c.get(k) for k in Citation.model_fields}))
        except Exception:
            log.warning("Skipped malformed citation: %s", c)

    return AgentResponse(content=content, citations=citations)


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
