"""Left rail: brand block, tab nav, new-chat shortcut, theme toggle.

Mirrors multibagg.ai's collapsible sidebar pattern but trimmed down — we
only have two views, so the nav doubles as the visual indication of which
tab is active.
"""

from __future__ import annotations

import reflex as rx

from ..state import State
from ..theme import APP_NAME, APP_TAGLINE


def _nav_button(label: str, icon_name: str, tab_id: str, badge_count=None) -> rx.Component:
    """Build the badge component at Python time so the None-vs-Var distinction
    doesn't have to be expressed inside an rx.cond (which would require Var
    boolean ops with mixed Python/Var operands)."""
    if badge_count is None:
        badge_component = rx.box()
    else:
        badge_component = rx.cond(
            badge_count > 0,
            rx.badge(
                badge_count.to_string(),
                color_scheme="iris",
                variant="solid",
                radius="full",
                size="1",
            ),
            rx.box(),
        )
    inner = rx.hstack(
        rx.icon(icon_name, size=16),
        rx.text(label, size="2", weight="medium"),
        rx.spacer(),
        badge_component,
        spacing="2",
        align="center",
        width="100%",
    )
    return rx.button(
        inner,
        on_click=lambda: State.set_active_tab(tab_id),
        variant=rx.cond(State.active_tab == tab_id, "soft", "ghost"),
        color_scheme=rx.cond(State.active_tab == tab_id, "iris", "gray"),
        size="3",
        width="100%",
        justify="start",
    )


def sidebar() -> rx.Component:
    return rx.vstack(
        # Brand block
        rx.vstack(
            rx.hstack(
                rx.icon("trending-up", size=24, color=rx.color("iris", 11)),
                rx.heading(APP_NAME, size="6", weight="bold", color=rx.color("iris", 11)),
                spacing="2",
                align="center",
            ),
            rx.text(
                APP_TAGLINE,
                size="1",
                color=rx.color("gray", 11),
                style={"line_height": "1.4"},
            ),
            spacing="1",
            align="start",
            padding="20px 24px 24px",
            width="100%",
        ),
        rx.divider(margin_y="4px"),

        # Nav
        rx.vstack(
            _nav_button("Chat",    "messages-square", "chat"),
            _nav_button("Sources", "file-text",       "sources", State.latest_citation_count),
            spacing="1",
            padding="8px 16px",
            width="100%",
            align="stretch",
        ),

        rx.spacer(),

        # New chat
        rx.box(
            rx.button(
                rx.icon("plus", size=14),
                rx.text("New chat", size="2"),
                on_click=State.clear_chat,
                variant="surface",
                color_scheme="gray",
                size="2",
                width="100%",
            ),
            padding="8px 16px",
            width="100%",
        ),

        # Theme toggle + footer
        rx.divider(margin_y="4px"),
        rx.hstack(
            rx.icon_button(
                rx.cond(State.appearance == "dark", rx.icon("sun"), rx.icon("moon")),
                on_click=State.toggle_appearance,
                variant="ghost",
                color_scheme="gray",
                size="2",
            ),
            rx.text(
                rx.cond(State.appearance == "dark", "Light mode", "Dark mode"),
                size="1",
                color=rx.color("gray", 11),
            ),
            rx.spacer(),
            rx.text("v24", size="1", color=rx.color("gray", 9)),
            spacing="2",
            align="center",
            padding="12px 20px 16px",
            width="100%",
        ),

        spacing="0",
        height="100vh",
        width="280px",
        background=rx.color("gray", 2),
        border_right=f"1px solid {rx.color('gray', 5)}",
        align="stretch",
    )
