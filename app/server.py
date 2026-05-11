"""FinSage v1 — FastAPI server.

Serves the static HTML/CSS/JS frontend and wraps backend.py's Databricks
SDK calls as JSON REST endpoints:

  GET  /                — serves templates/index.html (the SPA shell)
  GET  /static/*        — serves CSS / JS / fonts
  POST /api/ask         — wraps query_agent(messages) → {content, citations, messages}
  GET  /api/section     — wraps fetch_section_text(ticker, fy, ft, section) → {section_text}
  GET  /healthz         — liveness probe

Same backend.py the Streamlit app used; no logic duplication. Databricks
Apps runs this via `uvicorn server:app --host 0.0.0.0 --port 8000`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import AgentResponse, fetch_metrics_row, fetch_section_text, query_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
log = logging.getLogger("finsage-server")

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"
INDEX_HTML = TEMPLATES_DIR / "index.html"

app = FastAPI(
    title="FinSage Research",
    description="SEC-filing intelligence over 30 publicly traded companies.",
    docs_url="/api/docs",
    redoc_url=None,
)

# ─────────────────────────────────────────────────────────────────────────────
# Static + index
# ─────────────────────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(str(INDEX_HTML), media_type="text/html")


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    return {"ok": True, "agent_endpoint": os.getenv("FINSAGE_AGENT_ENDPOINT", "finsage_agent_endpoint")}


# ─────────────────────────────────────────────────────────────────────────────
# Request / response models
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role:    str
    content: str


class AskRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)


class AskResponse(BaseModel):
    content:   str             = ""
    citations: list[dict]      = Field(default_factory=list)
    messages:  list[dict]      = Field(default_factory=list)
    error:     Optional[str]   = None


class SectionResponse(BaseModel):
    section_text: Optional[str] = None
    error:        Optional[str] = None


class MetricsResponse(BaseModel):
    row:         Optional[dict[str, Any]] = None
    source_table: Optional[str]           = None
    error:        Optional[str]           = None


# ─────────────────────────────────────────────────────────────────────────────
# API: ask the agent
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/ask", response_model=AskResponse)
def api_ask(req: AskRequest) -> AskResponse:
    if not req.messages:
        return AskResponse(error="Empty messages array")
    wire = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        resp: AgentResponse = query_agent(wire)
    except Exception as exc:
        log.exception("query_agent failed")
        return AskResponse(error=f"Endpoint call failed: {exc}")

    citations_out = []
    for c in resp.citations:
        try:
            citations_out.append(c.model_dump())
        except AttributeError:
            citations_out.append(dict(c))

    return AskResponse(
        content=resp.content,
        citations=citations_out,
        messages=resp.messages,
        error=resp.error,
    )


# ─────────────────────────────────────────────────────────────────────────────
# API: fetch full section text (modal drill-in)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/section", response_model=SectionResponse)
def api_section(
    ticker:       str           = Query(..., min_length=1, max_length=10),
    fiscal_year:  int           = Query(..., ge=2000, le=2100),
    section_name: str           = Query(..., min_length=1, max_length=64),
    filing_type:  Optional[str] = Query(None),
) -> SectionResponse:
    try:
        text = fetch_section_text(ticker, fiscal_year, filing_type, section_name)
    except Exception as exc:
        log.exception("fetch_section_text failed")
        return SectionResponse(error=f"Section fetch failed: {exc}")
    if text is None:
        return SectionResponse(error="Section text not found in silver for this row.")
    return SectionResponse(section_text=text)


@app.get("/api/metrics", response_model=MetricsResponse)
def api_metrics(
    ticker:        str           = Query(..., min_length=1, max_length=10),
    fiscal_year:   int           = Query(..., ge=2000, le=2100),
    fiscal_quarter: Optional[int] = Query(None, ge=1, le=4),
) -> MetricsResponse:
    try:
        row = fetch_metrics_row(ticker, fiscal_year, fiscal_quarter)
    except Exception as exc:
        log.exception("fetch_metrics_row failed")
        return MetricsResponse(error=f"Metrics fetch failed: {exc}")
    if not row:
        q_label = f" Q{fiscal_quarter}" if fiscal_quarter is not None else ""
        return MetricsResponse(error=f"Metrics row not found for {ticker.upper()} FY{fiscal_year}{q_label}.")
    source_table = row.get("source_table")
    clean_row = dict(row)
    clean_row.pop("source_table", None)
    return MetricsResponse(row=clean_row, source_table=source_table)


# ─────────────────────────────────────────────────────────────────────────────
# Local dev entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("DATABRICKS_APP_PORT", os.getenv("PORT", "8000")))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
