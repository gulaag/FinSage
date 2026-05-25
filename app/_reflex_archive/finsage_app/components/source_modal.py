"""Source modal — tier-2 highlight view.

Opens when a user clicks a card in tab 2. Renders the full clean section
text (queried at click-time from main.finsage_silver.filing_sections) with
the retrieved chunk wrapped in <mark class="finsage-highlight">. The
highlight HTML is built server-side in State.modal_highlighted_html, with
HTML-escaping applied to all section content before our own tags are added.

Failure modes degrade gracefully:
  • Loading: spinner.
  • Section text missing in silver: chunk-only fallback view + amber callout.
  • SQL warehouse unreachable: same fallback.
"""

from __future__ import annotations

import reflex as rx

from ..state import State


def _modal_body() -> rx.Component:
    return rx.cond(
        State.modal_loading,
        rx.center(
            rx.vstack(
                rx.spinner(size="3", color=rx.color("iris", 11)),
                rx.text(
                    "Loading the full SEC filing section…",
                    size="2",
                    color=rx.color("gray", 11),
                ),
                spacing="3",
                align="center",
            ),
            padding="60px 0",
        ),
        rx.cond(
            State.modal_section_text != "",
            rx.box(
                rx.html(
                    State.modal_highlighted_html,
                    class_name="finsage-section-text",
                    style={"color": rx.color("gray", 12)},
                ),
                padding="0",
            ),
            rx.box(
                rx.text(
                    "Retrieved passage",
                    size="1",
                    color=rx.color("gray", 11),
                    weight="medium",
                    margin_bottom="6px",
                ),
                rx.box(
                    rx.text(
                        State.modal_chunk_text,
                        class_name="finsage-section-text",
                        style={"white_space": "pre-wrap"},
                    ),
                    padding="16px",
                    background=rx.color("amber", 2),
                    border=f"1px solid {rx.color('amber', 6)}",
                    border_radius="8px",
                ),
            ),
        ),
    )


def source_modal() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title(
                State.modal_title,
                weight="bold",
                color=rx.color("iris", 11),
            ),
            rx.dialog.description(
                "Source passage from the SEC filing. The retrieved chunk is "
                "highlighted in context.",
                size="1",
                color=rx.color("gray", 11),
                margin_bottom="16px",
            ),
            rx.cond(
                State.modal_error != "",
                rx.callout(
                    State.modal_error,
                    icon="info",
                    color_scheme="amber",
                    size="1",
                    margin_bottom="14px",
                ),
                rx.box(),
            ),
            rx.scroll_area(
                _modal_body(),
                type="hover",
                scrollbars="vertical",
                style={"height": "60vh", "padding_right": "12px"},
            ),
            rx.flex(
                rx.dialog.close(
                    rx.button(
                        "Close",
                        variant="soft",
                        color_scheme="gray",
                        size="2",
                    ),
                ),
                justify="end",
                margin_top="16px",
            ),
            style={"max_width": "920px", "max_height": "90vh"},
        ),
        open=State.modal_open,
        on_open_change=State.set_modal_open,
    )
