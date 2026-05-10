"""Visual design tokens, brand SVG, and CSS overrides for the FinSage chat app.

Design language: editorial premium-product feel — warm surface tones,
editorial serif typography for the wordmark and section labels, refined
chip hover states, restrained animation. Indigo remains the brand primary;
a coral-copper secondary accent (warm signature hue) anchors the
hero and certain detail callouts so the page does not read as monochrome
indigo. All component class names are stable so main.py needs no churn.

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

# Indigo (brand primary) — Radix iris 9 equivalent. Mirrored in
# .streamlit/config.toml so Streamlit's native widgets pick up the same hue
# without us patching each one individually.
BRAND_PRIMARY       = "#5b5bd6"
BRAND_PRIMARY_LIGHT = "#8b8bf8"
BRAND_PRIMARY_DARK  = "#4a4ab8"

# Coral-copper secondary accent (warm editorial signature). Used sparingly:
# brand-mark inner glint, hero serif sub-accent, refusal pill underline.
BRAND_ACCENT        = "#d97757"
BRAND_ACCENT_SOFT   = "#f4a574"

# Each example is (icon, question_text). The icon renders in the chip label;
# the staged question forwarded to the agent is the bare text only.
EXAMPLE_QUESTIONS: list[tuple[str, str]] = [
    ("$",  "What was Apple's revenue in fiscal year 2024?"),
    ("⇄", "Compare Microsoft and Alphabet's operating margins in 2023."),
    ("△", "Summarize NVIDIA's supply chain risks from their latest 10-K."),
    ("Q", "What was Amazon's Q3 fiscal year 2024 net income?"),
    ("◎", "How did Tesla describe autonomous-driving risks in their latest 10-K?"),
    ("%", "What was JPMorgan's debt-to-equity ratio in fiscal year 2023?"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Brand mark — softer rounded square with indigo gradient body, glassy top
# highlight, coral inner glint, and a refined "F" wordmark. Sized inline so
# the same SVG renders cleanly at sidebar (42px) and hero (108px) sizes.
# ─────────────────────────────────────────────────────────────────────────────

def brand_mark(size: int = 108, glow: bool = True) -> str:
    """Return the FinSage brand-mark SVG at the requested pixel size.

    Layers stacked inside a single SVG:
      1. Indigo gradient fill on the rounded square (brand body)
      2. White-tint glassy highlight on the top half
      3. Coral inner glint anchored bottom-left (warm-copper signature)
      4. White stylized "F" in slightly thinner geometry than v1

    Each gradient gets a size-suffixed id so multiple instances on one page
    do not collide. `glow=True` adds an indigo drop-shadow halo — used on
    the landing hero, off in the sidebar.
    """
    grad_id  = f"finsage-grad-{size}"
    hl_id    = f"finsage-hl-{size}"
    glint_id = f"finsage-glint-{size}"
    glow_style = (
        ' style="filter: drop-shadow(0 16px 40px rgba(91, 91, 214, 0.32)) '
        'drop-shadow(0 4px 12px rgba(217, 119, 87, 0.18));"'
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
        f'    <radialGradient id="{glint_id}" cx="22%" cy="82%" r="42%">'
        '      <stop offset="0%"   stop-color="rgba(244,165,116,0.55)"/>'
        '      <stop offset="100%" stop-color="rgba(244,165,116,0)"/>'
        '    </radialGradient>'
        '  </defs>'
        f'  <rect width="120" height="120" rx="28" fill="url(#{grad_id})"/>'
        f'  <rect width="120" height="120" rx="28" fill="url(#{glint_id})"/>'
        f'  <rect width="120" height="120" rx="28" fill="url(#{hl_id})"/>'
        '  <path d="M40 30 H82 V41 H52 V57 H76 V68 H52 V92 H40 Z" fill="white" '
        'fill-opacity="0.97"/>'
        '</svg>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS injected via st.markdown(unsafe_allow_html=True). All selectors
# scoped to either Streamlit's actual class names or our own .finsage-* ones.
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600;8..60,700&display=swap');

:root {
    /* Brand */
    --finsage-brand:           #5b5bd6;
    --finsage-brand-light:     #8b8bf8;
    --finsage-brand-dark:      #4a4ab8;
    --finsage-brand-soft:      rgba(91, 91, 214, 0.08);
    --finsage-brand-softer:    rgba(91, 91, 214, 0.04);
    --finsage-brand-border:    rgba(91, 91, 214, 0.22);

    /* Coral-copper warmth (secondary accent) */
    --finsage-accent:          #d97757;
    --finsage-accent-soft:     rgba(217, 119, 87, 0.12);
    --finsage-accent-softer:   rgba(217, 119, 87, 0.06);
    --finsage-accent-border:   rgba(217, 119, 87, 0.28);

    /* Highlight inside source modal */
    --finsage-mark-bg:         #fef3c7;
    --finsage-mark-text:       #78350f;

    /* Surfaces (warm cream / parchment in light mode) */
    --finsage-page-bg:         #faf9f6;
    --finsage-card-bg:         #ffffff;
    --finsage-card-border:     #e8e6df;
    --finsage-card-hover-bord: #d2cfc4;
    --finsage-divider:         #ece9e0;

    /* Type */
    --finsage-text:            #1c1c1f;
    --finsage-muted-strong:    #4b5563;
    --finsage-muted:           #767472;

    /* Shadow scale (gentler than v1, no bouncy drop) */
    --finsage-shadow-card:     0 1px 2px rgba(28, 25, 23, 0.04);
    --finsage-shadow-hover:    0 6px 18px rgba(91, 91, 214, 0.10),
                               0 1px 3px  rgba(28, 25, 23, 0.04);

    /* Type families */
    --finsage-font-sans:       'Inter', system-ui, -apple-system, "Segoe UI",
                                Roboto, "Helvetica Neue", Arial, sans-serif;
    --finsage-font-serif:      'Source Serif 4', 'Charter', 'Iowan Old Style',
                                Georgia, serif;
}

@media (prefers-color-scheme: dark) {
    :root {
        --finsage-mark-bg:         #78350f;
        --finsage-mark-text:       #fef3c7;

        /* Warm dark surfaces — editorial-console feel, not cool blue-black */
        --finsage-page-bg:         #1f1d1a;
        --finsage-card-bg:         #2a2825;
        --finsage-card-border:     #3a3733;
        --finsage-card-hover-bord: #5a5550;
        --finsage-divider:         #36332f;

        --finsage-text:            #f3efe6;
        --finsage-muted-strong:    #c8c2b6;
        --finsage-muted:           #968f83;

        --finsage-brand-soft:      rgba(124, 124, 242, 0.14);
        --finsage-brand-softer:    rgba(124, 124, 242, 0.07);
        --finsage-brand-border:    rgba(124, 124, 242, 0.30);

        --finsage-accent-soft:     rgba(217, 119, 87, 0.18);
        --finsage-accent-softer:   rgba(217, 119, 87, 0.08);
        --finsage-accent-border:   rgba(217, 119, 87, 0.36);

        --finsage-shadow-card:     0 1px 2px rgba(0, 0, 0, 0.32);
        --finsage-shadow-hover:    0 6px 18px rgba(0, 0, 0, 0.42),
                                   0 1px 3px  rgba(0, 0, 0, 0.40);
    }
}

/* ─── Streamlit chrome cleanup ─── */
#MainMenu      { visibility: hidden; }
footer         { visibility: hidden; }
header         { visibility: hidden; }

html, body, .stApp {
    background:
      radial-gradient(1200px 540px at 82% -10%, rgba(91, 91, 214, 0.07), transparent 60%),
      radial-gradient(880px  480px at -8% 108%, rgba(217, 119, 87, 0.06), transparent 60%),
      var(--finsage-page-bg) !important;
    color: var(--finsage-text);
    font-family: var(--finsage-font-sans);
    font-feature-settings: 'cv02', 'cv03', 'cv04', 'cv11', 'ss01';
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
@media (prefers-color-scheme: dark) {
    html, body, .stApp {
        background:
          radial-gradient(1200px 540px at 82% -10%, rgba(124, 124, 242, 0.10), transparent 60%),
          radial-gradient(880px  480px at -8% 108%, rgba(217, 119, 87, 0.07),  transparent 60%),
          var(--finsage-page-bg) !important;
    }
}

section.main > div.block-container {
    padding-top: 1.4rem;
    padding-bottom: 0.75rem;
    max-width: 1080px;
}

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] {
    background: var(--finsage-card-bg);
    border-right: 1px solid var(--finsage-divider);
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
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
    font-family: var(--finsage-font-serif);
    font-size: 1.55rem;
    font-weight: 600;
    letter-spacing: -0.6px;
    color: var(--finsage-text);
    line-height: 1.0;
}
.finsage-brand-eyebrow {
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 1.6px;
    color: var(--finsage-muted);
    text-transform: uppercase;
}
.finsage-tagline {
    color: var(--finsage-muted-strong);
    font-size: 0.86rem;
    padding: 14px 4px 4px;
    line-height: 1.55;
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
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #10b981;
    margin-right: 6px;
    vertical-align: middle;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.18);
    animation: finsage-pulse-dot 2.6s ease-in-out infinite;
}
@keyframes finsage-pulse-dot {
    0%, 100% { box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.20); }
    50%      { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0.0);  }
}

/* Sidebar expander — make body actually look like a paragraph */
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
    font-size: 0.85rem;
    line-height: 1.6;
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
    padding: 28px 16px 8px;
    animation: finsage-fade-up 0.6s cubic-bezier(.22,.61,.36,1) both;
}
@keyframes finsage-fade-up {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0);    }
}

.finsage-landing-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: var(--finsage-accent);
    margin-bottom: 22px;
    padding: 6px 14px;
    background: var(--finsage-accent-softer);
    border: 1px solid var(--finsage-accent-border);
    border-radius: 999px;
}
.finsage-landing-eyebrow::before {
    content: "";
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--finsage-accent);
}

.finsage-landing-logo {
    position: relative;
    display: inline-block;
    margin: 0 auto 24px;
    transition: transform 0.4s cubic-bezier(.22,.61,.36,1);
}
.finsage-landing-logo svg { position: relative; z-index: 1; }
.finsage-landing-logo:hover { transform: scale(1.03); }

.finsage-landing-wordmark {
    font-family: var(--finsage-font-serif);
    font-size: 3.6rem;
    font-weight: 600;
    letter-spacing: -1.6px;
    line-height: 1.05;
    background: linear-gradient(110deg,
        var(--finsage-brand)       0%,
        var(--finsage-brand-light) 32%,
        var(--finsage-accent-soft) 50%,
        var(--finsage-brand-light) 68%,
        var(--finsage-brand)       100%);
    background-size: 220% 100%;
    -webkit-background-clip: text;
            background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 14px;
    animation: finsage-shimmer 12s linear infinite;
}
@keyframes finsage-shimmer {
    from { background-position: 100% 0; }
    to   { background-position: -100% 0; }
}
.finsage-landing-tag {
    color: var(--finsage-muted-strong);
    font-size: 1.08rem;
    font-weight: 500;
    margin: 0 0 8px;
    font-family: var(--finsage-font-serif);
    font-style: italic;
}
.finsage-landing-sub {
    color: var(--finsage-muted);
    font-size: 0.96rem;
    line-height: 1.65;
    max-width: 580px;
    margin: 0 auto 30px;
}

.finsage-trust {
    display: inline-flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    margin: 0 0 44px;
}
.finsage-trust-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 13px;
    border: 1px solid var(--finsage-card-border);
    border-radius: 999px;
    background: var(--finsage-card-bg);
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--finsage-muted-strong);
    box-shadow: var(--finsage-shadow-card);
    transition: all 0.18s ease;
}
.finsage-trust-pill b {
    font-weight: 600;
    color: var(--finsage-text);
    font-variant-numeric: tabular-nums;
    letter-spacing: 0.2px;
}
.finsage-trust-pill:hover {
    border-color: var(--finsage-accent-border);
    color: var(--finsage-accent);
    background: var(--finsage-accent-softer);
    transform: translateY(-1px);
}
.finsage-trust-pill:hover b {
    color: var(--finsage-accent);
}

.finsage-section-label {
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 2.4px;
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--finsage-muted);
    margin: 8px 0 22px;
    font-family: var(--finsage-font-sans);
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

/* ─── Streamlit button → editorial suggestion chip ─── */
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
    transition:
      background 0.18s ease,
      border-color 0.18s ease,
      color 0.18s ease,
      transform 0.18s cubic-bezier(.22,.61,.36,1) !important;
}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    border-color: var(--finsage-brand-border) !important;
    background: var(--finsage-brand-soft) !important;
    color: var(--finsage-brand) !important;
    transform: translateY(-1px);
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
    line-height: 1.65;
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
    font-variant-numeric: tabular-nums;
}

/* ─── Badges ─── */
.finsage-badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 9px;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 600;
    line-height: 1.45;
    letter-spacing: 0.2px;
    font-variant-numeric: tabular-nums;
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
    background: var(--finsage-accent-soft);
    color: var(--finsage-accent);
    border: 1px solid var(--finsage-accent-border);
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
    font-family: var(--finsage-font-serif);
    font-size: 1.0rem;
    line-height: 1.75;
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
    line-height: 1.78;
    color: var(--finsage-text);
    white-space: pre-wrap;
}
.finsage-answer h1, .finsage-answer h2, .finsage-answer h3 {
    font-family: var(--finsage-font-serif);
    font-weight: 600;
    letter-spacing: -0.3px;
}
.finsage-answer strong { color: var(--finsage-text); font-weight: 600; }

/* ─── Empty-state hero (Sources tab when no citations yet) ─── */
.finsage-hero {
    text-align: center;
    padding: 56px 16px 28px;
    color: var(--finsage-text);
}
.finsage-hero-icon {
    font-size: 2.4rem;
    opacity: 0.5;
}
.finsage-hero-title {
    font-family: var(--finsage-font-serif);
    font-size: 1.4rem;
    font-weight: 600;
    margin: 14px 0 8px;
    line-height: 1.3;
    color: var(--finsage-text);
}
.finsage-hero-subtitle {
    color: var(--finsage-muted);
    font-size: 0.94rem;
    line-height: 1.65;
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
    line-height: 1.75;
}

/* Chat input — slightly elevated, refined focus */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] [contenteditable="true"] {
    font-family: var(--finsage-font-sans) !important;
    font-size: 0.96rem !important;
}
[data-testid="stChatInput"] > div {
    border: 1px solid var(--finsage-card-border) !important;
    background: var(--finsage-card-bg) !important;
    border-radius: 14px !important;
    box-shadow: var(--finsage-shadow-card) !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--finsage-brand-border) !important;
    box-shadow: 0 0 0 3px var(--finsage-brand-soft) !important;
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

/* Scrollbar polish */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--finsage-card-border);
    border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover { background: var(--finsage-card-hover-bord); }
</style>
"""
