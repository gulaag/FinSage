"""FinSage app state — chat history, citations, and source-modal control.

State design notes:
  • `messages` is the full conversational history shown in tab 1 and replayed
    on the wire when a new turn is sent (the agent re-reads the system prompt
    every turn but uses prior turns as conversational context).
  • `latest_citations` is a computed var that surfaces only the most recent
    assistant turn's citations to tab 2 — older citations stay attached to
    their original assistant messages but aren't featured. This matches the
    chatbot UX where tab 2 follows the current Q&A, not history.
  • Modal state (`modal_*`) is mutually exclusive with the rest of the UI: a
    single modal renders one source at a time. Loading states are tracked
    explicitly so the modal can show a spinner while the SQL warehouse
    materializes the section text.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional

import reflex as rx

from .backend import AgentResponse, fetch_section_text, query_agent

log = logging.getLogger("finsage-app.state")

# Strip the agent's machine-readable citation tags from the prose shown in
# the chat bubble. The structured `citations` field on the assistant message
# is the user-facing surface for these — tab 2 renders them as cards. The
# wire `content` (sent back to the agent on subsequent turns) keeps the tags
# so the model retains its self-citation context across the conversation.
_SOURCE_LINE_RE = re.compile(r"^\s*\[Source:[^\]]+\]\s*$\n?", re.MULTILINE)
_INLINE_TAG_RE  = re.compile(r"\[(?:VERBATIM|SUMMARY)\]\s*", re.IGNORECASE)


def _clean_for_display(content: str) -> str:
    if not content:
        return ""
    cleaned = _SOURCE_LINE_RE.sub("", content)
    cleaned = _INLINE_TAG_RE.sub("", cleaned)
    return cleaned.strip()


class CitationView(rx.Base):
    """UI-shaped citation. All fields are typed Reflex Vars at render time."""
    ticker:        str   = ""
    fiscal_year:   int   = 0
    filing_type:   str   = ""
    section_name:  str   = ""
    chunk_text:    str   = ""
    chunk_preview: str   = ""        # truncated for card preview (240 chars)
    score:         float = 0.0
    score_pct:     str   = ""        # e.g. "84.2%" — pre-formatted for chip
    label:         str   = ""


class ChatMessage(rx.Base):
    role:            str
    content:         str
    display_content: str                 = ""   # citations + tags stripped, for UI
    citations:       list[CitationView] = []   # populated for assistant messages


def _citation_label(c: dict) -> str:
    parts: list[str] = [str(c.get("ticker") or "?")]
    fy = c.get("fiscal_year")
    if fy is not None:
        parts.append(f"FY{int(fy)}")
    if c.get("filing_type"):
        parts.append(str(c["filing_type"]))
    if c.get("section_name"):
        parts.append(str(c["section_name"]))
    return " | ".join(parts)


def _to_citation_view(c: Any) -> CitationView:
    """Coerce a Citation pydantic model OR a plain dict into a CitationView
    with all display-ready fields pre-computed."""
    if hasattr(c, "model_dump"):
        d = c.model_dump()
    elif isinstance(c, dict):
        d = c
    else:
        return CitationView()
    chunk_text = str(d.get("chunk_text") or "")
    preview = chunk_text[:240]
    if len(chunk_text) > 240:
        preview = preview.rstrip() + "…"
    score_val = float(d.get("score") or 0.0)
    score_pct = f"{score_val * 100:.1f}%" if score_val > 0 else ""
    return CitationView(
        ticker        = str(d.get("ticker") or ""),
        fiscal_year   = int(d.get("fiscal_year") or 0),
        filing_type   = str(d.get("filing_type") or ""),
        section_name  = str(d.get("section_name") or ""),
        chunk_text    = chunk_text,
        chunk_preview = preview,
        score         = score_val,
        score_pct     = score_pct,
        label         = _citation_label(d),
    )


def _build_highlighted_html(section: str, chunk: str) -> str:
    """Wrap the matched chunk inside the section text with a <mark> span.

    Strategy: HTML-escape both, attempt exact substring match, then fall back
    to the first 200-char anchor of the chunk. If neither succeeds, return
    the section text un-highlighted (still safer than a wrong highlight).
    Newlines become <br/> for browser rendering.
    """
    import html
    if not section:
        return ""
    section_esc = html.escape(section)
    if not chunk:
        return section_esc.replace("\n", "<br/>")
    chunk_esc = html.escape(chunk)
    idx = section_esc.find(chunk_esc)
    match_text = chunk_esc
    if idx == -1:
        anchor = chunk_esc[:200]
        idx = section_esc.find(anchor) if anchor else -1
        match_text = anchor
    if idx == -1:
        return section_esc.replace("\n", "<br/>")
    before = section_esc[:idx]
    after  = section_esc[idx + len(match_text):]
    body = (
        f"{before}"
        f'<mark class="finsage-highlight">{match_text}</mark>'
        f"{after}"
    )
    return body.replace("\n", "<br/>")


class State(rx.State):
    messages:    list[ChatMessage] = []
    input_text:  str               = ""
    is_loading:  bool              = False
    error_text:  str               = ""

    active_tab:  str               = "chat"

    modal_open:         bool = False
    modal_title:        str  = ""
    modal_section_text: str  = ""
    modal_chunk_text:   str  = ""
    modal_loading:      bool = False
    modal_error:        str  = ""

    appearance: str = "light"  # "light" | "dark" — toggled via header button

    # ───── computed vars ─────

    @rx.var
    def has_messages(self) -> bool:
        return len(self.messages) > 0

    @rx.var
    def latest_citations(self) -> list[CitationView]:
        """Citations from the most recent assistant turn, score-ordered."""
        for msg in reversed(self.messages):
            if msg.role == "assistant" and msg.citations:
                return list(msg.citations)
        return []

    @rx.var
    def latest_citation_count(self) -> int:
        return len(self.latest_citations)

    @rx.var
    def modal_highlighted_html(self) -> str:
        return _build_highlighted_html(self.modal_section_text, self.modal_chunk_text)

    # ───── event handlers ─────

    def toggle_appearance(self) -> None:
        self.appearance = "dark" if self.appearance == "light" else "light"

    def set_active_tab(self, tab: str) -> None:
        self.active_tab = tab

    def set_input_text(self, value: str) -> None:
        self.input_text = value

    def stage_question(self, question: str) -> None:
        """Populate the input box with a preset question. Used by the empty-
        state example chips; the second event in the on_click list calls
        send_message to dispatch the staged text."""
        self.input_text = question

    async def send_message(self):
        text = (self.input_text or "").strip()
        if not text or self.is_loading:
            return
        self.error_text = ""
        self.input_text = ""
        self.messages = self.messages + [ChatMessage(role="user", content=text, display_content=text)]
        self.is_loading = True
        yield  # flush UI: user bubble + spinner

        wire_history = [
            {"role": m.role, "content": m.content} for m in self.messages
        ]
        try:
            response: AgentResponse = await asyncio.to_thread(query_agent, wire_history)
        except Exception as exc:
            log.exception("send_message failed")
            self.is_loading = False
            self.error_text = f"Endpoint error: {exc}"
            return

        if response.error:
            self.is_loading = False
            self.error_text = response.error
            return

        citations_snapshot = [_to_citation_view(c) for c in response.citations]
        raw_content = response.content or "(no response)"
        assistant = ChatMessage(
            role="assistant",
            content=raw_content,
            display_content=_clean_for_display(raw_content),
            citations=citations_snapshot,
        )
        self.messages = self.messages + [assistant]
        self.is_loading = False
        # Auto-pivot to citations tab when retrieval-grounded answers come back —
        # makes the differentiator immediately visible to first-time users.
        if citations_snapshot and self.active_tab == "chat":
            self.active_tab = "chat"  # keep on chat; tab 2 has a badge to nudge

    def clear_chat(self) -> None:
        self.messages = []
        self.error_text = ""
        self.is_loading = False
        self.modal_open = False

    async def open_source_modal(self, idx: int):
        citations = self.latest_citations
        if idx < 0 or idx >= len(citations):
            return
        c: CitationView = citations[idx]
        self.modal_title        = c.label
        self.modal_chunk_text   = c.chunk_text
        self.modal_section_text = ""
        self.modal_error        = ""
        self.modal_loading      = True
        self.modal_open         = True
        yield

        ticker      = c.ticker
        fiscal_year = c.fiscal_year
        filing_type = c.filing_type or None
        section     = c.section_name
        if not (ticker and fiscal_year and section):
            self.modal_loading = False
            self.modal_error = "Citation is missing identifying metadata; showing the chunk only."
            return

        try:
            text: Optional[str] = await asyncio.to_thread(
                fetch_section_text,
                ticker, int(fiscal_year), filing_type, section,
            )
        except Exception as exc:
            log.exception("fetch_section_text failed")
            self.modal_loading = False
            self.modal_error = f"Could not load full section ({exc}); showing chunk preview."
            return

        if text:
            self.modal_section_text = text
        else:
            self.modal_error = (
                "Full section text not in silver for this row; showing the "
                "retrieved chunk only."
            )
        self.modal_loading = False

    def close_modal(self) -> None:
        self.modal_open       = False
        self.modal_section_text = ""
        self.modal_chunk_text = ""
        self.modal_error      = ""

    def set_modal_open(self, value: bool) -> None:
        if not value:
            self.close_modal()
        else:
            self.modal_open = True
