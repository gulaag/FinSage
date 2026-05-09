"""Visual design tokens for FinSage.

The theme aims for a modern fintech-SaaS look — generous whitespace, soft
neutrals, deep indigo accent, fully responsive light/dark switching. Color
tokens map onto Reflex's radix-ui palette so dark mode works out of the box.
"""

from __future__ import annotations

# Reflex theme parameters — passed to rx.theme().
THEME_PARAMS = dict(
    appearance     = "light",       # default; user can flip via toggle
    accent_color   = "iris",        # indigo-blue — professional, financial
    gray_color     = "slate",       # neutral cool gray
    panel_background = "solid",
    radius         = "large",
    scaling        = "100%",
)

# Brand strings
APP_NAME    = "FinSage"
APP_TAGLINE = "Read 10-Ks at the speed of conversation"

# Example questions for the chatbot empty state
EXAMPLE_QUESTIONS = [
    "What was Apple's revenue in fiscal year 2024?",
    "Compare Microsoft and Alphabet's operating margins in 2023.",
    "Summarize NVIDIA's supply chain risks from their latest 10-K.",
    "What was Amazon's Q3 fiscal year 2024 net income?",
]

# Inline CSS for refinements Reflex's theme system doesn't cover —
# typography, code-block styling, scrollbar polish, mark highlight color.
CUSTOM_CSS = """
:root {
    --finsage-mark-bg: #fef3c7;
    --finsage-mark-text: #78350f;
    --finsage-chip-bg: var(--iris-3);
    --finsage-chip-border: var(--iris-7);
    --finsage-chip-text: var(--iris-11);
}

.dark, [data-theme="dark"], .radix-themes[data-is-root-theme="true"][appearance="dark"] {
    --finsage-mark-bg: #78350f;
    --finsage-mark-text: #fef3c7;
}

body, .rt-Theme {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

mark.finsage-highlight {
    background: var(--finsage-mark-bg);
    color: var(--finsage-mark-text);
    padding: 1px 4px;
    border-radius: 3px;
    font-weight: 500;
}

.finsage-citation-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    margin: 0 2px;
    background: var(--finsage-chip-bg);
    border: 1px solid var(--finsage-chip-border);
    color: var(--finsage-chip-text);
    border-radius: 4px;
    font-size: 0.78em;
    font-weight: 500;
    cursor: pointer;
    user-select: none;
    transition: filter 0.15s ease;
}

.finsage-citation-chip:hover {
    filter: brightness(1.06);
}

.finsage-section-text {
    font-family: 'Charter', 'Iowan Old Style', Georgia, serif;
    font-size: 0.95rem;
    line-height: 1.65;
    white-space: pre-wrap;
    word-wrap: break-word;
}

.finsage-chat-bubble-user {
    background: var(--iris-3);
    border: 1px solid var(--iris-6);
    color: var(--gray-12);
    border-radius: 14px 14px 4px 14px;
    padding: 12px 16px;
    max-width: 80%;
    align-self: flex-end;
}

.finsage-chat-bubble-assistant {
    background: var(--gray-2);
    border: 1px solid var(--gray-5);
    color: var(--gray-12);
    border-radius: 14px 14px 14px 4px;
    padding: 14px 18px;
    max-width: 90%;
    line-height: 1.6;
}

.finsage-chat-bubble-assistant p {
    margin: 0 0 12px 0;
}
.finsage-chat-bubble-assistant p:last-child {
    margin-bottom: 0;
}

::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: var(--gray-5);
    border-radius: 5px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--gray-7);
}
"""
