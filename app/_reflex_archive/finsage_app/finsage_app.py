"""FinSage chat app — entry point.

Layout: 280-px sidebar (brand + tab nav + theme toggle) on the left, the
active tab content (chat or sources) filling the rest of the viewport.
A single source modal renders on top of either tab when the user clicks
through to inspect a specific filing passage.

The Reflex theme is wrapped around the page so light/dark appearance can
be toggled at runtime via State.appearance.
"""

from __future__ import annotations

import reflex as rx

from .components.chat_tab import chat_tab
from .components.citations_tab import citations_tab
from .components.sidebar import sidebar
from .components.source_modal import source_modal
from .state import State
from .theme import APP_NAME, CUSTOM_CSS, THEME_PARAMS


def _main_content() -> rx.Component:
    return rx.box(
        rx.cond(
            State.active_tab == "chat",
            chat_tab(),
            citations_tab(),
        ),
        flex_grow="1",
        height="100vh",
        display="flex",
        flex_direction="column",
        background=rx.color("gray", 1),
        overflow="hidden",
    )


def index() -> rx.Component:
    theme_kwargs = {**THEME_PARAMS}
    theme_kwargs["appearance"] = State.appearance  # bind to state, allow toggle
    return rx.theme(
        rx.fragment(
            rx.html(f"<style>{CUSTOM_CSS}</style>"),
            rx.flex(
                sidebar(),
                _main_content(),
                spacing="0",
                height="100vh",
                width="100vw",
                overflow="hidden",
                align="stretch",
            ),
            source_modal(),
        ),
        **theme_kwargs,
    )


app = rx.App(
    style={
        "font_family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    },
)

app.add_page(
    index,
    title=f"{APP_NAME} — Read 10-Ks at the speed of conversation",
    description="AI-grounded SEC filing intelligence. 30 companies, 10-K + 10-Q narrative and metrics.",
)
