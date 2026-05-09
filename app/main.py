"""FinSage Streamlit entry point.

Single-file Streamlit app that:
  • Sends multi-turn chat to the FinSage Model Serving endpoint and renders
    professional 2-3 paragraph analyst answers in tab 1.
  • Surfaces every retrieved SEC filing chunk as a source card in tab 2.
  • Opens a modal on click that shows the FULL clean section text from
    main.finsage_silver.filing_sections with the retrieved chunk wrapped in
    <mark class="finsage-highlight"> for the tier-2 highlight effect.

Streamlit was chosen over Reflex purely because Streamlit is single-port-
native on Databricks Apps and pre-installed (1.38.0). The data layer
(backend.py) is framework-agnostic so a future migration to Reflex / Next.js
can reuse it without changes.
"""

from __future__ import annotations

import html as html_lib
import logging
import re
from typing import Any, Optional

import streamlit as st

from backend import AgentResponse, Citation, fetch_section_text, query_agent
from theme import APP_NAME, APP_TAGLINE, CUSTOM_CSS, EXAMPLE_QUESTIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
log = logging.getLogger("finsage-app")


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=f"{APP_NAME} — {APP_TAGLINE}",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Citation transforms (display-shaping)
# ─────────────────────────────────────────────────────────────────────────────

# Strip the agent's machine-readable citation tags from the prose shown in
# the chat bubble. The structured citations field is the user-facing surface
# for these — tab 2 renders them as cards.
_SOURCE_LINE_RE = re.compile(r"^\s*\[Source:[^\]]+\]\s*$\n?", re.MULTILINE)
_INLINE_TAG_RE  = re.compile(r"\[(?:VERBATIM|SUMMARY)\]\s*", re.IGNORECASE)


def clean_for_display(content: str) -> str:
    if not content:
        return ""
    cleaned = _SOURCE_LINE_RE.sub("", content)
    cleaned = _INLINE_TAG_RE.sub("", cleaned)
    return cleaned.strip()


def citation_label(c: Citation) -> str:
    parts: list[str] = [c.ticker or "?"]
    if c.fiscal_year is not None:
        parts.append(f"FY{int(c.fiscal_year)}")
    if c.filing_type:
        parts.append(c.filing_type)
    if c.section_name:
        parts.append(c.section_name)
    return " | ".join(parts)


def chunk_preview(text: str, max_len: int = 240) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "…"


def build_highlighted_html(section_text: str, chunk_text: str) -> str:
    """HTML-escape the section text, then wrap the chunk in <mark>. Falls
    back to a 200-char anchor when the exact chunk doesn't substring-match
    (e.g. minor whitespace differences). Newlines become <br/> for readability."""
    if not section_text:
        return ""
    section_esc = html_lib.escape(section_text)
    if not chunk_text:
        return section_esc.replace("\n", "<br/>")
    chunk_esc = html_lib.escape(chunk_text)
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


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────────────────────────────────────

def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("messages", [])              # list[dict]: role, content, display_content, citations
    ss.setdefault("modal_citation_idx", None)  # int | None
    ss.setdefault("staged_question", None)     # str | None — set by example chips
    ss.setdefault("active_tab", "chat")        # used for badge logic only


_init_state()


def latest_citations() -> list[Citation]:
    """Return the most recent assistant turn's citations as Citation objects."""
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant" and msg.get("citations"):
            return [
                c if isinstance(c, Citation) else Citation(**c)
                for c in msg["citations"]
            ]
    return []


def latest_citation_count() -> int:
    return len(latest_citations())


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f'<div class="finsage-brand">'
        f'  <span style="font-size: 1.6rem;">📈</span>'
        f'  <span class="finsage-brand-name">{APP_NAME}</span>'
        f'</div>'
        f'<div class="finsage-tagline">{APP_TAGLINE}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="finsage-divider" />', unsafe_allow_html=True)

    if st.button("New chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.modal_citation_idx = None
        st.session_state.staged_question = None
        st.rerun()

    st.markdown('<hr class="finsage-divider" />', unsafe_allow_html=True)

    with st.expander("About FinSage", expanded=False):
        st.markdown(
            "Grounded financial Q&A over 30 publicly traded companies, "
            "fiscal years 2020–2026.\n\n"
            "Every numerical answer comes from the SEC EDGAR XBRL "
            "CompanyFacts API, persisted in a Lakehouse Gold layer. "
            "Every qualitative answer is retrieved from 10-K and 10-Q "
            "section text via Databricks Vector Search.\n\n"
            "Click a citation in the **Sources** tab to drill into the "
            "original filing passage."
        )

    st.markdown(
        '<div style="position: absolute; bottom: 16px; left: 16px; right: 16px; '
        'color: var(--finsage-muted); font-size: 0.74rem; '
        'border-top: 1px solid var(--finsage-divider); padding-top: 12px;">'
        'Agent v24 · Llama 3.3 70B · Databricks'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Source modal (Streamlit 1.37+ st.dialog)
# ─────────────────────────────────────────────────────────────────────────────

def _badge_row_html(citation: Citation) -> str:
    """Build the colored-badge header row as a single HTML string."""
    parts: list[str] = [
        f'<span class="finsage-badge finsage-badge-ticker">{html_lib.escape(citation.ticker or "?")}</span>'
    ]
    if citation.fiscal_year:
        parts.append(f'<span class="finsage-badge finsage-badge-fy">FY{int(citation.fiscal_year)}</span>')
    if citation.filing_type:
        cls = "finsage-badge-filing-10k" if citation.filing_type == "10-K" else "finsage-badge-filing-10q"
        parts.append(f'<span class="finsage-badge {cls}">{html_lib.escape(citation.filing_type)}</span>')
    if citation.section_name:
        parts.append(f'<span class="finsage-badge finsage-badge-section">{html_lib.escape(citation.section_name)}</span>')
    return (
        '<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px; flex-wrap: wrap;">'
        + "".join(parts)
        + "</div>"
    )


@st.dialog(" ", width="large")
def show_source_dialog(citation: Citation):
    st.markdown(_badge_row_html(citation), unsafe_allow_html=True)
    st.caption("Source passage from the SEC filing. The retrieved chunk is highlighted in context.")
    st.markdown('<hr class="finsage-divider" />', unsafe_allow_html=True)

    if not (citation.ticker and citation.fiscal_year and citation.section_name):
        st.warning("Citation is missing identifying metadata — showing the chunk only.")
        st.markdown(
            f'<div class="finsage-section-body">'
            f'<mark class="finsage-highlight">{html_lib.escape(citation.chunk_text)}</mark>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Loading the full SEC filing section…"):
        section_text = fetch_section_text(
            citation.ticker,
            int(citation.fiscal_year),
            citation.filing_type,
            citation.section_name,
        )

    if section_text:
        highlighted = build_highlighted_html(section_text, citation.chunk_text)
        st.markdown(
            f'<div class="finsage-modal-scroll">'
            f'  <div class="finsage-section-body">{highlighted}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "Full section text isn't materialized in silver for this row. "
            "Showing the retrieved chunk only — the agent did still ground "
            "its answer in this passage."
        )
        st.markdown(
            f'<div class="finsage-section-body" style="background: var(--finsage-brand-soft); '
            f'padding: 16px; border-radius: 8px; border: 1px solid var(--finsage-brand-border);">'
            f'<mark class="finsage-highlight">{html_lib.escape(citation.chunk_text)}</mark>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _maybe_open_modal() -> None:
    """Open the dialog if a citation has been staged this run."""
    idx = st.session_state.get("modal_citation_idx")
    if idx is None:
        return
    citations = latest_citations()
    if 0 <= idx < len(citations):
        show_source_dialog(citations[idx])
    # Reset so the dialog doesn't reopen on the next rerun
    st.session_state.modal_citation_idx = None


# ─────────────────────────────────────────────────────────────────────────────
# Tab content renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_empty_chat() -> None:
    st.markdown(
        f'<div class="finsage-hero">'
        f'  <div style="font-size: 3rem;">📈</div>'
        f'  <div class="finsage-hero-title">Ask {APP_NAME} about any of 30 covered companies.</div>'
        f'  <div class="finsage-hero-subtitle">'
        f'    Annual + quarterly metrics, plus the underlying 10-K and 10-Q '
        f'    narrative. Every answer is grounded in SEC filings — open the '
        f'    <b>Sources</b> tab to see the exact passage.'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"example-{i}", use_container_width=True):
                st.session_state.staged_question = q
                st.rerun()


def _render_assistant_message(msg: dict) -> None:
    cite_count = len(msg.get("citations") or [])
    with st.chat_message("assistant", avatar="📊"):
        st.markdown(
            f'<div class="finsage-answer">{html_lib.escape(msg["display_content"])}</div>',
            unsafe_allow_html=True,
        )
        if cite_count > 0:
            st.markdown(
                f'<div style="margin-top: 12px; padding-top: 10px; '
                f'border-top: 1px solid var(--finsage-divider); '
                f'color: var(--finsage-muted); font-size: 0.82rem;">'
                f'🔗 {cite_count} source{"s" if cite_count != 1 else ""} cited '
                f'— open the <b>Sources</b> tab to inspect.'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_user_message(msg: dict) -> None:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(msg["display_content"])


def _render_chat_history() -> None:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            _render_user_message(msg)
        else:
            _render_assistant_message(msg)


def _render_chat_tab() -> None:
    if not st.session_state.messages and not st.session_state.staged_question:
        _render_empty_chat()
        return
    _render_chat_history()


def _filing_type_badge_html(ft: Optional[str]) -> str:
    if not ft:
        return ""
    cls = "finsage-badge-filing-10k" if ft == "10-K" else "finsage-badge-filing-10q"
    return f'<span class="finsage-badge {cls}">{html_lib.escape(ft)}</span>'


def _render_source_card(idx: int, c: Citation) -> None:
    score_pill = ""
    if c.score and c.score > 0:
        score_pill = (
            f'<span class="finsage-score-pill">'
            f'  📊 match {c.score * 100:.1f}%'
            f'</span>'
        )

    fy_badge = (
        f'<span class="finsage-badge finsage-badge-fy">FY{int(c.fiscal_year)}</span>'
        if c.fiscal_year else ""
    )
    section_badge = (
        f'<span class="finsage-badge finsage-badge-section">{html_lib.escape(c.section_name or "")}</span>'
        if c.section_name else ""
    )

    with st.container():
        st.markdown(
            f'<div class="finsage-source-card">'
            f'  <div class="finsage-source-card-header">'
            f'    <span class="finsage-badge finsage-badge-ticker">{html_lib.escape(c.ticker or "?")}</span>'
            f'    {fy_badge}'
            f'    {_filing_type_badge_html(c.filing_type)}'
            f'    {section_badge}'
            f'  </div>'
            f'  <div class="finsage-source-card-body">{html_lib.escape(chunk_preview(c.chunk_text))}</div>'
            f'  <div class="finsage-source-card-footer">'
            f'    {score_pill or "&nbsp;"}'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "🔍 View full source passage",
            key=f"view-src-{idx}",
            use_container_width=False,
        ):
            st.session_state.modal_citation_idx = idx
            st.rerun()


def _render_sources_tab() -> None:
    citations = latest_citations()
    if not citations:
        st.markdown(
            '<div class="finsage-hero">'
            '  <div style="font-size: 3rem;">📄</div>'
            '  <div class="finsage-hero-title">No sources yet</div>'
            '  <div class="finsage-hero-subtitle">'
            '    Ask FinSage a qualitative question — supply chain risks, '
            '    strategy, MD&A commentary — and the underlying SEC filing '
            '    passages will appear here, one card per retrieved passage.'
            '  </div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"### Sources for the latest answer  "
        f"<span style='color: var(--finsage-muted); font-weight: 400; font-size: 0.95rem;'>"
        f"{len(citations)} passages</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Click **View full source passage** on any card to see the entire SEC filing "
        "section with the retrieved chunk highlighted in context."
    )
    st.markdown('<hr class="finsage-divider" />', unsafe_allow_html=True)

    for idx, c in enumerate(citations):
        _render_source_card(idx, c)


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────

cite_count = latest_citation_count()
sources_label = f"📄 Sources ({cite_count})" if cite_count > 0 else "📄 Sources"

tab_chat, tab_sources = st.tabs(["💬 Chat", sources_label])

with tab_chat:
    _render_chat_tab()

with tab_sources:
    _render_sources_tab()


# ─────────────────────────────────────────────────────────────────────────────
# Modal trigger (must run after tabs so we have rendered content underneath)
# ─────────────────────────────────────────────────────────────────────────────

_maybe_open_modal()


# ─────────────────────────────────────────────────────────────────────────────
# Chat input + send (placed at bottom; Streamlit chat_input pins it visually)
# ─────────────────────────────────────────────────────────────────────────────

# An example-chip click stages a question; we dispatch it on this rerun
# rather than waiting for the user to type into chat_input.
def _dispatch_user_text(text: str) -> None:
    text = (text or "").strip()
    if not text:
        return

    user_msg = {
        "role": "user",
        "content": text,
        "display_content": text,
        "citations": [],
    }
    st.session_state.messages.append(user_msg)

    wire_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(text)

    with st.chat_message("assistant", avatar="📊"):
        with st.spinner("FinSage is consulting the filings…"):
            response: AgentResponse = query_agent(wire_history)

        if response.error:
            st.error(f"Endpoint error: {response.error}")
            # Pop the user message so the next turn doesn't re-send a
            # half-finished exchange.
            st.session_state.messages.pop()
            return

        raw_content = response.content or "(no response)"
        cleaned = clean_for_display(raw_content)
        citations_serialized = [
            c.model_dump() if hasattr(c, "model_dump") else dict(c)
            for c in (response.citations or [])
        ]
        st.session_state.messages.append({
            "role": "assistant",
            "content": raw_content,
            "display_content": cleaned,
            "citations": citations_serialized,
        })
        st.markdown(
            f'<div class="finsage-answer">{html_lib.escape(cleaned)}</div>',
            unsafe_allow_html=True,
        )
        if citations_serialized:
            st.markdown(
                f'<div style="margin-top: 12px; padding-top: 10px; '
                f'border-top: 1px solid var(--finsage-divider); '
                f'color: var(--finsage-muted); font-size: 0.82rem;">'
                f'🔗 {len(citations_serialized)} source(s) cited '
                f'— open the <b>Sources</b> tab to inspect.'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Rerun once so the new exchange lands in chat history AND the citation
    # count on the Sources tab badge updates.
    st.rerun()


# Drain a staged example question first
if st.session_state.staged_question:
    staged = st.session_state.staged_question
    st.session_state.staged_question = None
    _dispatch_user_text(staged)

prompt = st.chat_input("Ask about revenue, margins, risks, strategy…")
if prompt:
    _dispatch_user_text(prompt)
