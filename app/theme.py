"""Visual design tokens, brand SVG, and CSS overrides for the FinSage chat app.

Streamlit ships its own theme system (background / text / primary color via
`.streamlit/config.toml`); CUSTOM_CSS layers on top to polish elements
Streamlit's defaults can't reach — landing hero, gradient wordmark, chip
buttons, citation chips, source-card hover, mark highlights inside the modal.

`brand_mark(size)` returns the SVG mark used in the sidebar (small) and the
landing hero (large). One source of truth, no PNG asset to ship.
"""

from __future__ import annotations

APP_NAME    = "FinSage"
APP_TAGLINE = "Be a PRO financial researcher."
APP_SUBLINE = (
    "Grounded SEC-filing intelligence over 30 publicly traded companies. "
    "Every number is sourced. Every quote is verifiable."
)

# Brand color (deep indigo — Radix iris 9 equivalent). Mirrored in
# .streamlit/config.toml so Streamlit's primary widgets pick up the same hue
# without us patching each one individually.
BRAND_PRIMARY       = "#5b5bd6"
BRAND_PRIMARY_LIGHT = "#8b8bf8"
BRAND_PRIMARY_DARK  = "#4a4ab8"

# Each example is (icon, question_text). The icon is rendered into the chip
# label only — the staged question forwarded to the agent is the bare text.
EXAMPLE_QUESTIONS: list[tuple[str, str]] = [
    ("💰", "What was Apple's revenue in fiscal year 2024?"),
    ("⚖️", "Compare Microsoft and Alphabet's operating margins in 2023."),
    ("⚠️", "Summarize NVIDIA's supply chain risks from their latest 10-K."),
    ("📊", "What was Amazon's Q3 fiscal year 2024 net income?"),
    ("🚗", "How did Tesla describe autonomous-driving risks in their latest 10-K?"),
    ("🏦", "What was JPMorgan's debt-to-equity ratio in fiscal year 2023?"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Brand mark — single rounded-square gradient SVG with a stylized "F".
# Sized inline so we can drop a tiny version in the sidebar and a large one
# in the landing hero without touching CSS.
# ─────────────────────────────────────────────────────────────────────────────

def brand_mark(size: int = 120, glow: bool = True) -> str:
    """Return the FinSage brand-mark SVG at the requested pixel size.

    Three layers stacked inside a single SVG:
      1. Indigo gradient fill on the rounded square (brand body)
      2. White-tint glassy highlight on the top half (premium feel)
      3. White stylized "F" (the wordmark)

    Each gradient gets a size-suffixed id so multiple instances on one page
    don't collide. `glow=True` adds an outer drop-shadow halo — used on the
    landing hero, off in the sidebar.
    """
    grad_id = f"finsage-grad-{size}"
    hl_id   = f"finsage-hl-{size}"
    glow_style = (
        ' style="filter: drop-shadow(0 14px 32px rgba(91, 91, 214, 0.38));"'
        if glow else ""
    )
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 120 120" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="FinSage logo"{glow_style}>'
        '  <defs>'
        f'    <linearGradient id="{grad_id}" x1="0%" y1="0%" x2="100%" y2="100%">'
        '      <stop offset="0%"   stop-color="#9999ff"/>'
        '      <stop offset="55%"  stop-color="#5b5bd6"/>'
        '      <stop offset="100%" stop-color="#3d3d9a"/>'
        '    </linearGradient>'
        f'    <linearGradient id="{hl_id}" x1="0%" y1="0%" x2="0%" y2="100%">'
        '      <stop offset="0%"   stop-color="rgba(255,255,255,0.22)"/>'
        '      <stop offset="55%"  stop-color="rgba(255,255,255,0)"/>'
        '    </linearGradient>'
        '  </defs>'
        f'  <rect width="120" height="120" rx="28" fill="url(#{grad_id})"/>'
        f'  <rect width="120" height="120" rx="28" fill="url(#{hl_id})"/>'
        '  <path d="M40 30 H82 V42 H52 V58 H76 V70 H52 V92 H40 Z" fill="white"/>'
        '</svg>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS injected via st.markdown(unsafe_allow_html=True). All selectors
# scoped to either Streamlit's actual class names or our own .finsage-* ones.
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --finsage-brand:           #5b5bd6;
    --finsage-brand-light:     #8b8bf8;
    --finsage-brand-dark:      #4a4ab8;
    --finsage-brand-soft:      rgba(91, 91, 214, 0.08);
    --finsage-brand-softer:    rgba(91, 91, 214, 0.04);
    --finsage-brand-border:    rgba(91, 91, 214, 0.22);
    --finsage-mark-bg:         #fef3c7;
    --finsage-mark-text:       #78350f;
    --finsage-page-bg:         #fbfbfd;
    --finsage-card-bg:         #ffffff;
    --finsage-card-border:     #e5e7eb;
    --finsage-card-hover-bord: #c4c4ee;
    --finsage-muted:           #6b7280;
    --finsage-muted-strong:    #4b5563;
    --finsage-text:            #1c1c1f;
    --finsage-divider:         #e9e9ee;
    --finsage-shadow-card:     0 1px 2px rgba(15, 15, 35, 0.04);
    --finsage-shadow-hover:    0 8px 24px rgba(91, 91, 214, 0.12);
}

@media (prefers-color-scheme: dark) {
    :root {
        --finsage-mark-bg:         #78350f;
        --finsage-mark-text:       #fef3c7;
        --finsage-page-bg:         #0c0c14;
        --finsage-card-bg:         #1a1a22;
        --finsage-card-border:     #2a2a35;
        --finsage-card-hover-bord: #4a4ab0;
        --finsage-muted:           #9ca3af;
        --finsage-muted-strong:    #cbd0d9;
        --finsage-text:            #f3f4f6;
        --finsage-divider:         #2a2a35;
        --finsage-brand-soft:      rgba(124, 124, 242, 0.12);
        --finsage-brand-softer:    rgba(124, 124, 242, 0.06);
        --finsage-brand-border:    rgba(124, 124, 242, 0.32);
        --finsage-shadow-card:     0 1px 2px rgba(0, 0, 0, 0.30);
        --finsage-shadow-hover:    0 8px 24px rgba(0, 0, 0, 0.45);
    }
}

/* ─── Streamlit chrome cleanup ─── */
#MainMenu      { visibility: hidden; }
footer         { visibility: hidden; }
header         { visibility: hidden; }

html, body, .stApp {
    background:
      radial-gradient(circle at 1px 1px, rgba(91,91,214,0.07) 1px, transparent 0)
          0 0 / 28px 28px,
      radial-gradient(1100px 520px at 78% -8%,  rgba(91,91,214,0.07), transparent 60%),
      radial-gradient(820px  460px at -10% 110%, rgba(139,139,248,0.06), transparent 60%),
      var(--finsage-page-bg) !important;
    color: var(--finsage-text);
    font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    font-feature-settings: 'cv02', 'cv03', 'cv04', 'cv11';
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
@media (prefers-color-scheme: dark) {
    html, body, .stApp {
        background:
          radial-gradient(circle at 1px 1px, rgba(139,139,248,0.08) 1px, transparent 0)
              0 0 / 28px 28px,
          radial-gradient(1100px 520px at 78% -8%,  rgba(91,91,214,0.10),  transparent 60%),
          radial-gradient(820px  460px at -10% 110%, rgba(139,139,248,0.08), transparent 60%),
          var(--finsage-page-bg) !important;
    }
}

section.main > div.block-container {
    padding-top: 1.4rem;
    padding-bottom: 0.75rem;
    max-width: 1100px;
}

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] {
    background: var(--finsage-card-bg);
    border-right: 1px solid var(--finsage-divider);
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.1rem;
}

.finsage-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 4px 0;
}
.finsage-brand-mark { display: inline-flex; }
.finsage-brand-mark svg { display: block; border-radius: 12px; }
.finsage-brand-text { display: flex; flex-direction: column; gap: 4px; }
.finsage-brand-name {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.6px;
    color: var(--finsage-text);
    line-height: 1.0;
}
.finsage-brand-eyebrow {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 1.6px;
    color: var(--finsage-muted);
    text-transform: uppercase;
}
.finsage-tagline {
    color: var(--finsage-muted-strong);
    font-size: 0.84rem;
    padding: 12px 4px 4px;
    line-height: 1.5;
}

.finsage-sidebar-footer {
    position: absolute;
    bottom: 16px; left: 16px; right: 16px;
    color: var(--finsage-muted);
    font-size: 0.74rem;
    line-height: 1.55;
    border-top: 1px solid var(--finsage-divider);
    padding-top: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
}
.finsage-sidebar-footer b { color: var(--finsage-muted-strong); font-weight: 600; }

.finsage-status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #10b981;
    margin-right: 6px;
    vertical-align: middle;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.18);
    animation: finsage-pulse-dot 2.4s ease-in-out infinite;
}
@keyframes finsage-pulse-dot {
    0%, 100% { box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.20); }
    50%      { box-shadow: 0 0 0 7px rgba(16, 185, 129, 0.0);  }
}

/* Sidebar expander — make the body actually look like a paragraph */
section[data-testid="stSidebar"] .streamlit-expanderHeader,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-weight: 600;
    color: var(--finsage-text);
}
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    border: 1px solid var(--finsage-divider);
    border-radius: 10px;
    background: var(--finsage-brand-softer);
}
section[data-testid="stSidebar"] [data-testid="stExpander"] p,
section[data-testid="stSidebar"] [data-testid="stExpander"] li {
    color: var(--finsage-muted-strong);
    font-size: 0.84rem;
    line-height: 1.55;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] strong {
    color: var(--finsage-text);
}
section[data-testid="stSidebar"] [data-testid="stExpander"] em {
    color: var(--finsage-brand);
    font-style: normal;
    font-weight: 600;
}

/* ─── Landing hero (empty chat state) ─── */
.finsage-landing {
    text-align: center;
    padding: 36px 16px 12px;
    animation: finsage-fade-up 0.7s ease-out both;
}
@keyframes finsage-fade-up {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0);    }
}

.finsage-landing-logo {
    position: relative;
    display: inline-block;
    margin: 0 auto 26px;
    animation: finsage-float 4.5s ease-in-out infinite;
    transition: transform 0.3s ease;
}
.finsage-landing-logo::before {
    content: "";
    position: absolute;
    inset: -28%;
    background: radial-gradient(closest-side,
        rgba(91, 91, 214, 0.32) 0%,
        rgba(91, 91, 214, 0.12) 45%,
        transparent 70%);
    border-radius: 50%;
    z-index: 0;
    pointer-events: none;
    animation: finsage-pulse 3.6s ease-in-out infinite;
}
.finsage-landing-logo svg {
    position: relative;
    z-index: 1;
}
.finsage-landing-logo:hover {
    transform: scale(1.04);
}
@keyframes finsage-float {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-6px); }
}
@keyframes finsage-pulse {
    0%, 100% { opacity: 0.7; transform: scale(1.0);  }
    50%      { opacity: 1.0; transform: scale(1.10); }
}

.finsage-landing-wordmark {
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -2.4px;
    line-height: 1.05;
    background: linear-gradient(110deg,
        var(--finsage-brand)       0%,
        var(--finsage-brand-light) 28%,
        #e0e0ff                    50%,
        var(--finsage-brand-light) 72%,
        var(--finsage-brand)       100%);
    background-size: 220% 100%;
    -webkit-background-clip: text;
            background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 10px;
    animation: finsage-shimmer 8s linear infinite;
}
@keyframes finsage-shimmer {
    from { background-position: 100% 0; }
    to   { background-position: -100% 0; }
}
.finsage-landing-tag {
    color: var(--finsage-muted-strong);
    font-size: 1.1rem;
    font-weight: 500;
    margin: 0 0 8px;
}
.finsage-landing-sub {
    color: var(--finsage-muted);
    font-size: 0.96rem;
    line-height: 1.6;
    max-width: 580px;
    margin: 0 auto 26px;
}

.finsage-trust {
    display: inline-flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    margin: 0 0 38px;
}
.finsage-trust-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border: 1px solid var(--finsage-card-border);
    border-radius: 999px;
    background: var(--finsage-card-bg);
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--finsage-muted-strong);
    box-shadow: var(--finsage-shadow-card);
    transition: all 0.18s ease;
}
.finsage-trust-pill:hover {
    border-color: var(--finsage-brand-border);
    color: var(--finsage-brand);
    transform: translateY(-1px);
    box-shadow: 0 6px 14px rgba(91, 91, 214, 0.10);
}

.finsage-section-label {
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 2.2px;
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--finsage-muted);
    margin: 6px 0 18px;
}
.finsage-section-label::before,
.finsage-section-label::after {
    content: "";
    display: inline-block;
    width: 36px;
    height: 1px;
    background: var(--finsage-divider);
    vertical-align: middle;
    margin: 0 12px;
}

/* ─── Generic Streamlit button → chip-style card ─── */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    background: var(--finsage-card-bg) !important;
    color: var(--finsage-text) !important;
    border: 1px solid var(--finsage-card-border) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    font-size: 0.94rem !important;
    font-weight: 500 !important;
    line-height: 1.5 !important;
    text-align: left !important;
    box-shadow: var(--finsage-shadow-card) !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    border-color: var(--finsage-brand) !important;
    background: var(--finsage-brand-soft) !important;
    color: var(--finsage-brand) !important;
    transform: translateY(-2px);
    box-shadow: var(--finsage-shadow-hover) !important;
}
.stButton > button:focus,
.stButton > button:focus-visible {
    outline: none !important;
    box-shadow: 0 0 0 3px var(--finsage-brand-soft),
                var(--finsage-shadow-hover) !important;
}

/* Sidebar buttons → centered, slightly tighter */
section[data-testid="stSidebar"] .stButton > button {
    text-align: center !important;
    padding: 10px 14px !important;
    font-size: 0.9rem !important;
}

/* ─── Source cards ─── */
.finsage-source-card {
    background: var(--finsage-card-bg);
    border: 1px solid var(--finsage-card-border);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: var(--finsage-shadow-card);
    transition: all 0.18s ease;
}
.finsage-source-card:hover {
    border-color: var(--finsage-card-hover-bord);
    transform: translateY(-1px);
    box-shadow: var(--finsage-shadow-hover);
}
.finsage-source-card-header {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 10px;
}
.finsage-source-card-body {
    color: var(--finsage-text);
    font-size: 0.92rem;
    line-height: 1.6;
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

/* ─── Badges (used in source cards + modal header) ─── */
.finsage-badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 9px;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 600;
    line-height: 1.45;
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

/* ─── Modal section text ─── */
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
.finsage-modal-scroll {
    max-height: 60vh;
    overflow-y: auto;
    padding-right: 8px;
}

/* ─── Assistant answer prose ─── */
.finsage-answer {
    font-size: 1.0rem;
    line-height: 1.75;
    color: var(--finsage-text);
    white-space: pre-wrap;
}

/* ─── Empty-state hero (used on Sources tab when no citations yet) ─── */
.finsage-hero {
    text-align: center;
    padding: 56px 16px 28px;
    color: var(--finsage-text);
}
.finsage-hero-icon {
    font-size: 2.4rem;
    opacity: 0.6;
}
.finsage-hero-title {
    font-size: 1.35rem;
    font-weight: 600;
    margin: 14px 0 8px;
    line-height: 1.3;
    color: var(--finsage-text);
}
.finsage-hero-subtitle {
    color: var(--finsage-muted);
    font-size: 0.94rem;
    line-height: 1.6;
    max-width: 540px;
    margin: 0 auto 22px;
}

/* ─── Chat surfaces ─── */
[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
    padding: 4px 0 6px;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    line-height: 1.7;
}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid var(--finsage-divider);
}
button[data-baseweb="tab"] {
    font-size: 0.96rem !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--finsage-brand) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--finsage-brand) !important;
}

/* ─── Misc ─── */
hr.finsage-divider {
    border: none;
    border-top: 1px solid var(--finsage-divider);
    margin: 14px 0;
}
</style>
"""
