"""Tab 1 — conversational Q&A surface.

Layout:
  - Empty state: centered hero with example questions as clickable chips.
  - Populated: scrollable message stack + input form pinned at bottom.

Citation tags ([Source: ...], [VERBATIM], [SUMMARY]) are stripped from the
displayed prose by State (see _clean_for_display); the structured citations
panel in tab 2 is the user-facing surface for source attribution.
"""

from __future__ import annotations

import reflex as rx

from ..state import ChatMessage, State
from ..theme import EXAMPLE_QUESTIONS


def _user_bubble(msg: ChatMessage) -> rx.Component:
    return rx.flex(
        rx.box(
            rx.text(
                msg.display_content,
                size="3",
                style={"white_space": "pre-wrap", "line_height": "1.5"},
            ),
            class_name="finsage-chat-bubble-user",
        ),
        justify="end",
        width="100%",
        padding_y="4px",
    )


def _assistant_bubble(msg: ChatMessage) -> rx.Component:
    return rx.flex(
        rx.box(
            # `display_content` already has [Source:] lines removed; rendering
            # with white_space: pre-wrap preserves the LLM's paragraph breaks.
            rx.text(
                msg.display_content,
                size="3",
                style={"white_space": "pre-wrap", "line_height": "1.65"},
            ),
            rx.cond(
                msg.citations.length() > 0,
                rx.hstack(
                    rx.icon("link-2", size=12, color=rx.color("iris", 11)),
                    rx.text(
                        msg.citations.length().to_string() + " sources cited — open the Sources tab to inspect",
                        size="1",
                        color=rx.color("gray", 11),
                    ),
                    spacing="2",
                    align="center",
                    margin_top="14px",
                    padding_top="10px",
                    border_top=f"1px solid {rx.color('gray', 5)}",
                ),
                rx.box(),
            ),
            class_name="finsage-chat-bubble-assistant",
        ),
        justify="start",
        width="100%",
        padding_y="4px",
    )


def _message_view(msg: ChatMessage) -> rx.Component:
    return rx.cond(msg.role == "user", _user_bubble(msg), _assistant_bubble(msg))


def _loading_indicator() -> rx.Component:
    return rx.flex(
        rx.box(
            rx.hstack(
                rx.spinner(size="2", color=rx.color("iris", 11)),
                rx.text(
                    "FinSage is consulting the filings…",
                    size="2",
                    color=rx.color("gray", 11),
                    style={"font_style": "italic"},
                ),
                spacing="2",
                align="center",
            ),
            class_name="finsage-chat-bubble-assistant",
            padding="12px 18px",
        ),
        justify="start",
        width="100%",
        padding_y="4px",
    )


def _empty_state() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.icon("trending-up", size=44, color=rx.color("iris", 10)),
            rx.heading(
                "Ask FinSage about any of 30 covered companies.",
                size="6",
                weight="medium",
                color=rx.color("gray", 12),
                text_align="center",
                style={"max_width": "560px", "line_height": "1.3"},
            ),
            rx.text(
                "Annual + quarterly metrics, plus the underlying 10-K and 10-Q "
                "narrative. Every answer is grounded in SEC filings — click "
                "into Sources to see the exact passage.",
                size="2",
                color=rx.color("gray", 11),
                text_align="center",
                style={"max_width": "520px", "line_height": "1.5"},
            ),
            rx.box(height="20px"),
            rx.flex(
                rx.foreach(
                    EXAMPLE_QUESTIONS,
                    lambda q: rx.button(
                        q,
                        on_click=lambda: [State.stage_question(q), State.send_message],
                        variant="surface",
                        color_scheme="gray",
                        size="2",
                        style={
                            "white_space": "normal",
                            "text_align": "left",
                            "height": "auto",
                            "padding": "10px 14px",
                            "max_width": "240px",
                        },
                    ),
                ),
                wrap="wrap",
                gap="10px",
                justify="center",
                style={"max_width": "560px"},
            ),
            spacing="3",
            align="center",
        ),
        width="100%",
        height="100%",
        padding="40px 24px",
    )


def _input_bar() -> rx.Component:
    return rx.box(
        rx.form(
            rx.hstack(
                rx.input(
                    placeholder="Ask about revenue, margins, risks, strategy…",
                    value=State.input_text,
                    on_change=State.set_input_text,
                    size="3",
                    radius="large",
                    style={"flex_grow": "1"},
                    disabled=State.is_loading,
                ),
                rx.button(
                    rx.icon("send", size=16),
                    type="submit",
                    size="3",
                    color_scheme="iris",
                    disabled=State.is_loading | (State.input_text.length() == 0),
                ),
                spacing="2",
                width="100%",
            ),
            on_submit=State.send_message,
            reset_on_submit=False,
            width="100%",
        ),
        rx.cond(
            State.error_text != "",
            rx.callout(
                State.error_text,
                icon="triangle-alert",
                color_scheme="ruby",
                size="1",
                margin_top="8px",
            ),
            rx.box(),
        ),
        padding="16px 28px 24px",
        background=rx.color("gray", 1),
        border_top=f"1px solid {rx.color('gray', 5)}",
        width="100%",
    )


def _conversation() -> rx.Component:
    return rx.vstack(
        rx.foreach(State.messages, _message_view),
        rx.cond(State.is_loading, _loading_indicator(), rx.box()),
        spacing="2",
        align="stretch",
        width="100%",
        max_width="820px",
        padding="28px 28px 16px",
        margin="0 auto",
    )


def chat_tab() -> rx.Component:
    return rx.vstack(
        rx.box(
            rx.cond(State.has_messages, _conversation(), _empty_state()),
            flex_grow="1",
            width="100%",
            overflow_y="auto",
        ),
        _input_bar(),
        spacing="0",
        height="100%",
        width="100%",
        align="stretch",
    )
