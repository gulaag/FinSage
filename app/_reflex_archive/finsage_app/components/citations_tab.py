"""Tab 2 — sources panel.

Renders the structured citations[] returned by the v24 agent as a card grid.
Each card surfaces ticker, fiscal year, filing type, section name, similarity
score, and a 240-character preview of the retrieved chunk. Clicking a card
opens the source modal with the full clean section text and the chunk
highlighted in context (tier-2 highlight).
"""

from __future__ import annotations

import reflex as rx

from ..state import CitationView, State


def _empty_state() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.icon("file-search", size=44, color=rx.color("gray", 9)),
            rx.heading(
                "No sources yet",
                size="5",
                weight="medium",
                color=rx.color("gray", 12),
            ),
            rx.text(
                "Ask FinSage a qualitative question — supply chain risks, "
                "strategy, MD&A commentary — and the underlying SEC filing "
                "passages will appear here.",
                size="2",
                color=rx.color("gray", 11),
                text_align="center",
                style={"max_width": "420px", "line_height": "1.5"},
            ),
            rx.button(
                rx.icon("arrow-left", size=14),
                rx.text("Back to chat", size="2"),
                on_click=lambda: State.set_active_tab("chat"),
                variant="surface",
                color_scheme="iris",
                size="2",
                margin_top="8px",
            ),
            spacing="3",
            align="center",
        ),
        width="100%",
        height="100%",
        padding="60px 24px",
    )


def _filing_type_badge(c: CitationView) -> rx.Component:
    return rx.cond(
        c.filing_type != "",
        rx.badge(
            c.filing_type,
            color_scheme=rx.cond(c.filing_type == "10-K", "amber", "iris"),
            variant="soft",
            size="1",
        ),
        rx.box(),
    )


def _section_badge(c: CitationView) -> rx.Component:
    return rx.cond(
        c.section_name != "",
        rx.badge(c.section_name, color_scheme="gray", variant="soft", size="1"),
        rx.box(),
    )


def _score_chip(c: CitationView) -> rx.Component:
    return rx.cond(
        c.score_pct != "",
        rx.hstack(
            rx.icon("activity", size=11, color=rx.color("gray", 10)),
            rx.text("match " + c.score_pct, size="1", color=rx.color("gray", 10)),
            spacing="1",
            align="center",
        ),
        rx.box(),
    )


def _source_card(c: CitationView, idx: int) -> rx.Component:
    return rx.box(
        rx.vstack(
            # Header row: ticker + FY + filing type + section
            rx.hstack(
                rx.badge(c.ticker, color_scheme="iris", variant="solid", size="2"),
                rx.badge(
                    "FY" + c.fiscal_year.to_string(),
                    color_scheme="gray",
                    variant="surface",
                    size="1",
                ),
                _filing_type_badge(c),
                _section_badge(c),
                spacing="2",
                wrap="wrap",
                align="center",
            ),
            # Chunk preview
            rx.text(
                c.chunk_preview,
                size="2",
                color=rx.color("gray", 12),
                style={"line_height": "1.55"},
            ),
            # Footer: score + open button
            rx.hstack(
                _score_chip(c),
                rx.spacer(),
                rx.button(
                    rx.text("View source", size="1"),
                    rx.icon("external-link", size=12),
                    on_click=lambda: State.open_source_modal(idx),
                    variant="ghost",
                    color_scheme="iris",
                    size="1",
                ),
                width="100%",
                align="center",
            ),
            spacing="3",
            align="stretch",
        ),
        padding="18px",
        border=f"1px solid {rx.color('gray', 5)}",
        border_radius="12px",
        background=rx.color("gray", 1),
        cursor="pointer",
        on_click=lambda: State.open_source_modal(idx),
        _hover={
            "border_color":   rx.color("iris", 8),
            "background":     rx.color("gray", 2),
            "transform":      "translateY(-1px)",
            "box_shadow":     "0 4px 12px rgba(0,0,0,0.06)",
        },
        transition="all 0.15s ease",
    )


def _populated() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.heading(
                "Sources for the latest answer",
                size="5",
                weight="medium",
                color=rx.color("gray", 12),
            ),
            rx.spacer(),
            rx.text(
                State.latest_citation_count.to_string() + " passages",
                size="2",
                color=rx.color("gray", 11),
            ),
            width="100%",
            align="center",
            padding_bottom="8px",
        ),
        rx.text(
            "Click any card to see the full SEC filing section with the "
            "retrieved passage highlighted in context.",
            size="2",
            color=rx.color("gray", 11),
            margin_bottom="16px",
        ),
        rx.foreach(State.latest_citations, _source_card),
        spacing="3",
        align="stretch",
        width="100%",
        max_width="900px",
        padding="32px 28px 40px",
        margin="0 auto",
    )


def citations_tab() -> rx.Component:
    return rx.box(
        rx.cond(State.latest_citation_count > 0, _populated(), _empty_state()),
        width="100%",
        height="100%",
        overflow_y="auto",
    )
