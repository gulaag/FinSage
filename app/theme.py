"""Visual design tokens and CSS overrides for the FinSage chat app.

Streamlit ships its own theme system (background / text / primary color via
`.streamlit/config.toml`); CUSTOM_CSS layers on top to polish elements
Streamlit's defaults can't reach — citation chips, source-card hover, mark
highlights inside the modal, scrollbar treatment.
"""

from __future__ import annotations

APP_NAME    = "FinSage"
APP_TAGLINE = "Read 10-Ks at the speed of conversation."

# Brand color (deep indigo — Radix iris 9 equivalent). Used in CSS overrides
# AND mirrored in .streamlit/config.toml so Streamlit's primary widgets pick
# up the same hue without us patching each one individually.
BRAND_PRIMARY = "#5b5bd6"
BRAND_PRIMARY_DARK = "#7c7cf2"

EXAMPLE_QUESTIONS = [
    "What was Apple's revenue in fiscal year 2024?",
    "Compare Microsoft and Alphabet's operating margins in 2023.",
    "Summarize NVIDIA's supply chain risks from their latest 10-K.",
    "What was Amazon's Q3 fiscal year 2024 net income?",
    "How did Tesla describe autonomous driving risks?",
    "What's JPMorgan's debt-to-equity ratio for fiscal year 2023?",
]

# Inline CSS injected via st.markdown(unsafe_allow_html=True). All selectors
# scoped to either Streamlit's actual class names or our own .finsage-* ones.
CUSTOM_CSS = """
<style>
:root {
    --finsage-brand: #5b5bd6;
    --finsage-brand-soft: rgba(91, 91, 214, 0.08);
    --finsage-brand-border: rgba(91, 91, 214, 0.22);
    --finsage-mark-bg: #fef3c7;
    --finsage-mark-text: #78350f;
    --finsage-card-bg: #ffffff;
    --finsage-card-border: #e5e7eb;
    --finsage-card-hover-border: #c4c4ee;
    --finsage-muted: #6b7280;
    --finsage-text: #1c1c1f;
    --finsage-divider: #e5e7eb;
}

@media (prefers-color-scheme: dark) {
    :root {
        --finsage-mark-bg: #78350f;
        --finsage-mark-text: #fef3c7;
        --finsage-card-bg: #1f1f23;
        --finsage-card-border: #2e2e35;
        --finsage-card-hover-border: #4a4ab0;
        --finsage-muted: #9ca3af;
        --finsage-text: #f3f4f6;
        --finsage-divider: #2e2e35;
        --finsage-brand-soft: rgba(124, 124, 242, 0.12);
        --finsage-brand-border: rgba(124, 124, 242, 0.32);
    }
}

#MainMenu      { visibility: hidden; }
footer         { visibility: hidden; }
header         { visibility: hidden; }

section.main > div.block-container {
    padding-top: 1.4rem;
    padding-bottom: 0.75rem;
    max-width: 1100px;
}

section[data-testid="stSidebar"] {
    background: var(--finsage-card-bg);
    border-right: 1px solid var(--finsage-divider);
}

.finsage-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px 4px;
}
.finsage-brand-name {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--finsage-brand);
    letter-spacing: -0.5px;
}
.finsage-tagline {
    color: var(--finsage-muted);
    font-size: 0.78rem;
    padding: 0 12px 16px;
    line-height: 1.45;
}

.finsage-source-card {
    background: var(--finsage-card-bg);
    border: 1px solid var(--finsage-card-border);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 14px;
    transition: all 0.15s ease;
}
.finsage-source-card:hover {
    border-color: var(--finsage-card-hover-border);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}
.finsage-source-card-header {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 10px;
}
.finsage-badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 9px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    line-height: 1.4;
    letter-spacing: 0.2px;
}
.finsage-badge-ticker {
    background: var(--finsage-brand);
    color: white;
}
.finsage-badge-fy {
    background: var(--finsage-card-border);
    color: var(--finsage-text);
}
.finsage-badge-filing-10k {
    background: rgba(245, 158, 11, 0.15);
    color: #b45309;
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.finsage-badge-filing-10q {
    background: var(--finsage-brand-soft);
    color: var(--finsage-brand);
    border: 1px solid var(--finsage-brand-border);
}
.finsage-badge-section {
    background: var(--finsage-brand-soft);
    color: var(--finsage-brand);
}
.finsage-source-card-body {
    color: var(--finsage-text);
    font-size: 0.92rem;
    line-height: 1.55;
    margin-bottom: 10px;
}
.finsage-source-card-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: var(--finsage-muted);
    font-size: 0.78rem;
}
.finsage-score-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

mark.finsage-highlight {
    background: var(--finsage-mark-bg) !important;
    color: var(--finsage-mark-text) !important;
    padding: 2px 4px;
    border-radius: 3px;
    font-weight: 500;
}

.finsage-section-body {
    font-family: 'Charter', 'Iowan Old Style', Georgia, serif;
    font-size: 0.96rem;
    line-height: 1.7;
    color: var(--finsage-text);
    word-wrap: break-word;
}

.finsage-answer {
    font-size: 1.0rem;
    line-height: 1.7;
    color: var(--finsage-text);
    white-space: pre-wrap;
}

.finsage-hero {
    text-align: center;
    padding: 36px 16px 20px;
    color: var(--finsage-text);
}
.finsage-hero-title {
    font-size: 1.5rem;
    font-weight: 600;
    margin: 18px 0 8px;
    line-height: 1.3;
}
.finsage-hero-subtitle {
    color: var(--finsage-muted);
    font-size: 0.94rem;
    line-height: 1.55;
    max-width: 540px;
    margin: 0 auto 22px;
}

button[data-baseweb="tab"] {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--finsage-brand) !important;
}

hr.finsage-divider {
    border: none;
    border-top: 1px solid var(--finsage-divider);
    margin: 14px 0;
}

.finsage-loading {
    color: var(--finsage-muted);
    font-style: italic;
    padding: 10px 0;
}
</style>
"""
