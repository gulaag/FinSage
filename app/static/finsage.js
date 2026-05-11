// FinSage v1 frontend — vanilla JS, no framework.
//
// State model:
//   state.threads        — array of {id, title, ticker, messages, createdAt}
//                          ordered newest-first. Each thread is one
//                          research conversation. Messages within a thread
//                          are full chat history; the canvas always renders
//                          ONLY the latest exchange (last user + next
//                          assistant) per the handoff design.
//   state.currentThreadId
//
// On "New research": stash any current non-empty thread, create a fresh
// empty thread, render. The sidebar shows every thread; the active one has
// the gold left rail.

const KNOWN_TICKERS = new Set([
  "AAPL","ABBV","AMZN","BAC","CRM","DDOG","F","GM","GOOGL","GS",
  "JNJ","JPM","KO","LCID","MA","MCD","MRK","MSFT","NET","NKE",
  "NVDA","PFE","PLTR","RIVN","SBUX","SNOW","TSLA","UNH","V","WMT"
]);
const NAME_TO_TICKER = {
  apple: "AAPL", abbvie: "ABBV", amazon: "AMZN", "bank of america": "BAC",
  salesforce: "CRM", datadog: "DDOG", ford: "F", "general motors": "GM",
  alphabet: "GOOGL", google: "GOOGL", goldman: "GS", "johnson & johnson": "JNJ",
  jpmorgan: "JPM", "coca-cola": "KO", coke: "KO", lucid: "LCID",
  mastercard: "MA", mcdonald: "MCD", merck: "MRK", microsoft: "MSFT",
  cloudflare: "NET", nike: "NKE", nvidia: "NVDA", pfizer: "PFE",
  palantir: "PLTR", rivian: "RIVN", starbucks: "SBUX", snowflake: "SNOW",
  tesla: "TSLA", unitedhealth: "UNH", visa: "V", walmart: "WMT",
};

// ────── DOM refs ──────
const app           = document.getElementById("app");
const sidebar       = document.getElementById("sidebar");
const sidebarColl   = document.getElementById("sidebarCollapse");
const sidebarReopen = document.getElementById("sidebarReopen");
const newBtn        = document.getElementById("newResearchBtn");
const composer      = document.getElementById("composer");
const composerInput = document.getElementById("composerInput");
const composerSend  = document.getElementById("composerSend");
const canvas        = document.getElementById("canvas");
const hero          = document.getElementById("hero");
const thread        = document.getElementById("thread");
const crumbCurrent  = document.getElementById("crumbCurrent");
const topbarTicker  = document.getElementById("topbarTicker");
const topbarStatus  = document.getElementById("topbarStatus");
const modalBackdrop = document.getElementById("modalBackdrop");
const modalBadges   = document.getElementById("modalBadges");
const modalBody     = document.getElementById("modalBody");
const modalClose    = document.getElementById("modalClose");
const examples      = document.querySelectorAll(".fs-example");
const threadList    = document.getElementById("threadList");
const canvasScroll  = canvas.parentElement;
let modalRequestSeq = 0;

// ────── State ──────
function makeThreadId() {
  return `t-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

function freshThread() {
  return {
    id: makeThreadId(),
    title: "New research",
    ticker: null,
    messages: [],
    createdAt: Date.now(),
    inflight: false,
    requestSeq: 0,
  };
}

const state = {
  threads: [freshThread()],
  currentThreadId: null,   // set after first render
};
state.currentThreadId = state.threads[0].id;

function getCurrentThread() {
  return state.threads.find(t => t.id === state.currentThreadId) || null;
}

function getThreadMessages() {
  const t = getCurrentThread();
  return t ? t.messages : [];
}

function createNewResearch() {
  // Keep the current thread in the list (so it shows in sidebar history)
  // and prepend a fresh empty one if and only if the current isn't already empty.
  const cur = getCurrentThread();
  if (cur && cur.messages.length === 0) {
    // Already on a clean empty thread; nothing to do
    return;
  }
  const t = freshThread();
  state.threads.unshift(t);
  state.currentThreadId = t.id;
}

function switchToThread(id) {
  if (state.threads.find(t => t.id === id)) {
    state.currentThreadId = id;
    closeModal();
  }
}

// ────── Helpers ──────
function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function tickerFromText(text) {
  if (!text) return null;
  const tokenMatches = text.match(/\b([A-Z]{1,5})\b/g) || [];
  for (const t of tokenMatches) {
    if (KNOWN_TICKERS.has(t)) return t;
  }
  const lo = text.toLowerCase();
  for (const [name, tk] of Object.entries(NAME_TO_TICKER)) {
    if (lo.includes(name)) return tk;
  }
  return null;
}

function italicizeEntity(text) {
  const safe = escapeHtml(text);
  const m = safe.match(/\b([A-Z]{1,5})\b/);
  if (m && KNOWN_TICKERS.has(m[1])) {
    return safe.replace(m[1], `<em>${m[1]}</em>`);
  }
  for (const [name, _] of Object.entries(NAME_TO_TICKER)) {
    const re = new RegExp(`\\b(${name})\\b`, "i");
    const mm = safe.match(re);
    if (mm) return safe.replace(re, `<em>${mm[1]}</em>`);
  }
  return safe;
}

function mdToHtml(text) {
  if (!text) return "";
  let out = String(text);
  out = out.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  out = out.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  out = out.replace(/^# (.+)$/gm, "<h2>$1</h2>");
  out = out.replace(/\*\*([^*]+?)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/(^|[^*])\*([^*\n]+?)\*(?!\*)/g, "$1<em>$2</em>");
  const blocks = out.split(/\n\s*\n+/).map(b => b.trim()).filter(Boolean);
  return blocks.map(b => {
    if (/^<h[123]>/.test(b)) return b;
    return `<p>${b.replace(/\n/g, "<br>")}</p>`;
  }).join("\n");
}

function wrapInlineNumbers(html) {
  const re = /(\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:B|M|K|bn|mn|billion|million|thousand|trillion))?)|(\d[\d,]*(?:\.\d+)?\s?%)|(\d[\d,]*(?:\.\d+)?\s?bps)/gi;
  const parts = html.split(/(<[^>]+>)/g);
  return parts.map(p => {
    if (p.startsWith("<")) return p;
    return p.replace(re, m => `<span class="fs-num">${m}</span>`);
  }).join("");
}

function matchCitation(srcStr, citations) {
  if (!citations || !citations.length) return null;
  const parts = srcStr.split("|").map(p => p.trim());
  const tickerGuess = (parts[0] || "").toUpperCase();
  let fyGuess = null, sectionGuess = null;
  for (let i = 1; i < parts.length; i++) {
    const p = parts[i];
    if (/^FY\d{4}/i.test(p)) {
      fyGuess = parseInt(p.replace(/^FY/i, "").slice(0, 4), 10);
    } else if (["Business", "Risk Factors", "MD&A", "Risk Factors Updates"].includes(p)) {
      sectionGuess = p;
    }
  }
  let best = null, bestScore = 0;
  for (const c of citations) {
    let score = 0;
    if (c.ticker && c.ticker.toUpperCase() === tickerGuess) score += 2;
    if (c.fiscal_year && fyGuess && parseInt(c.fiscal_year, 10) === fyGuess) score += 2;
    if (c.section_name && sectionGuess && c.section_name === sectionGuess) score += 1;
    if (score > bestScore) { bestScore = score; best = c; }
  }
  return bestScore >= 3 ? best : null;
}

function parseSourceLabel(srcStr) {
  const parts = srcStr.split("|").map(p => p.trim()).filter(Boolean);
  if (!parts.length) return {};
  const out = {
    label: srcStr,
    ticker: null,
    fiscal_year: null,
    fiscal_quarter: null,
    source_kind: "unknown", // "section" | "metrics" | "metadata" | "unknown"
    section_name: null,
  };
  out.ticker = (parts[0] || "").toUpperCase() || null;
  for (let i = 1; i < parts.length; i++) {
    const p = parts[i];
    const fyq = p.match(/^FY(\d{4})(?:\s*Q([1-4]))?$/i);
    if (fyq) {
      out.fiscal_year = parseInt(fyq[1], 10);
      out.fiscal_quarter = fyq[2] ? parseInt(fyq[2], 10) : null;
      continue;
    }
    if (["Business", "Risk Factors", "MD&A", "Risk Factors Updates"].includes(p)) {
      out.source_kind = "section";
      out.section_name = p;
      continue;
    }
    if (p.toLowerCase() === "metrics") {
      out.source_kind = "metrics";
      continue;
    }
    if (p.toLowerCase() === "10-k cover page") {
      out.source_kind = "metadata";
      continue;
    }
  }
  return out;
}

function transformCitations(content, citations) {
  const re = /\[Source:\s*([^\]]+?)\s*\]/g;
  const seen = new Map();
  const cites = {};
  let n = 1;
  const html = content.replace(re, (_m, src) => {
    src = src.trim();
    if (!seen.has(src)) {
      seen.set(src, n);
      const matched = matchCitation(src, citations);
      const parsed = parseSourceLabel(src);
      const sourceKind = matched ? "section" : (parsed.source_kind || "unknown");
      cites[n] = {
        label:        src,
        quote:        (matched && matched.chunk_text) || "",
        location:     (matched && matched.section_name) || "",
        ticker:       (matched && matched.ticker) || parsed.ticker || null,
        fiscal_year:  (matched && matched.fiscal_year) || parsed.fiscal_year || null,
        fiscal_quarter: parsed.fiscal_quarter || null,
        filing_type:  matched ? matched.filing_type : null,
        section_name: (matched && matched.section_name) || parsed.section_name || null,
        chunk_text:   matched ? matched.chunk_text : "",
        source_kind:  sourceKind,
      };
      n++;
    }
    const num = seen.get(src);
    const c = cites[num];
    const quote = c.quote
      ? `<span class="quote">&ldquo;${escapeHtml(c.quote.slice(0, 280))}&rdquo;</span>`
      : "";
    const locText =
      c.source_kind === "section"
        ? (c.location || "Open filing section")
        : c.source_kind === "metrics"
          ? "Open structured metrics"
          : c.source_kind === "metadata"
            ? "Open filing metadata"
          : "Reference";
    const loc = escapeHtml(locText);
    const canOpen =
      c.source_kind === "section" ||
      c.source_kind === "metrics" ||
      c.source_kind === "metadata";
    const popup =
      `<span class="pop">` +
        `<span class="src">${escapeHtml(c.label)}</span>` +
        quote +
        `<span class="loc"><span>${loc}</span><span>${canOpen ? "↗" : ""}</span></span>` +
      `</span>`;
    const clickableClass = canOpen ? " clickable" : "";
    return `<span class="fs-cite${clickableClass}" data-cite="${num}">${num}${popup}</span>`;
  });
  return { html, cites };
}

function stripInlineTags(s) {
  return (s || "").replace(/\[(?:VERBATIM|SUMMARY)\]\s*/gi, "");
}

// ────── Latest-exchange selector ──────
// Returns {userMsg, asstMsg} where asstMsg is the FIRST assistant message
// at index > userMsg's index, NOT the latest assistant overall. This is
// the fix for the "old answer shown during loading" bug.
function getLatestExchange() {
  const messages = getThreadMessages();
  let userIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") { userIdx = i; break; }
  }
  if (userIdx < 0) return { userMsg: null, asstMsg: null };
  const userMsg = messages[userIdx];
  let asstMsg = null;
  for (let j = userIdx + 1; j < messages.length; j++) {
    if (messages[j].role === "assistant") { asstMsg = messages[j]; break; }
  }
  return { userMsg, asstMsg };
}

// ────── Render: agent timeline ──────
function renderTimeline(rawMessages) {
  if (!rawMessages || !rawMessages.length) return "";
  const steps = [];
  const pending = [];
  for (const m of rawMessages) {
    if (!m || typeof m !== "object") continue;
    if (m.role === "assistant" && Array.isArray(m.tool_calls)) {
      for (const tc of m.tool_calls) {
        const fn = (tc && tc.function) || {};
        pending.push({
          name: fn.name || "tool",
          args: typeof fn.arguments === "string" ? fn.arguments : "",
        });
      }
    } else if (m.role === "tool" && pending.length) {
      const tc = pending.shift();
      const argsClean = (tc.args || "").slice(0, 180);
      const bodyLine =
        `Called <b>${escapeHtml(tc.name)}</b>` +
        (argsClean ? ` with <span class="fs-num">${escapeHtml(argsClean)}</span>` : "");
      const src = (m.content || "").split("\n", 1)[0].slice(0, 140);
      steps.push({ body: bodyLine, src });
    }
  }
  if (!steps.length) return "";
  const stepsHtml = steps.map(s => `
    <div class="fs-step">
      <div class="fs-marker">✓</div>
      <div class="body">${s.body}${s.src ? `<div class="src">${escapeHtml(s.src)}</div>` : ""}</div>
      <div class="ms"></div>
    </div>
  `).join("");
  return `
    <div class="fs-agent">
      <div class="fs-agent-head" data-toggle="agent">
        <span class="fs-eyebrow" style="margin:0">Reasoning</span>
        <span class="status">
          <span class="pulse done"></span>
          Synthesized across ${steps.length} step${steps.length !== 1 ? "s" : ""}
        </span>
        <span class="timing"></span>
        <span class="caret">▾</span>
      </div>
      <div class="fs-agent-body">${stepsHtml}</div>
    </div>
  `;
}

function extractMetricsEvidence(rawMessages) {
  if (!rawMessages || !rawMessages.length) return [];
  const out = [];
  const pending = [];
  for (const m of rawMessages) {
    if (!m || typeof m !== "object") continue;
    if (m.role === "assistant" && Array.isArray(m.tool_calls)) {
      for (const tc of m.tool_calls) {
        const fn = (tc && tc.function) || {};
        pending.push(fn.name || "tool");
      }
      continue;
    }
    if (m.role === "tool" && pending.length) {
      const toolName = pending.shift();
      if (toolName === "get_company_metrics" || toolName === "get_quarterly_metrics") {
        out.push({
          tool_name: toolName,
          content: String(m.content || ""),
        });
      }
    }
  }
  return out;
}

function extractMetadataEvidence(rawMessages) {
  if (!rawMessages || !rawMessages.length) return [];
  const out = [];
  const pending = [];
  for (const m of rawMessages) {
    if (!m || typeof m !== "object") continue;
    if (m.role === "assistant" && Array.isArray(m.tool_calls)) {
      for (const tc of m.tool_calls) {
        const fn = (tc && tc.function) || {};
        pending.push(fn.name || "tool");
      }
      continue;
    }
    if (m.role === "tool" && pending.length) {
      const toolName = pending.shift();
      if (toolName === "get_filing_metadata") {
        out.push({
          tool_name: toolName,
          content: String(m.content || ""),
        });
      }
    }
  }
  return out;
}

// ────── Render: TLDR + prose ──────
function renderTldrAndProse(transformedHtml) {
  const blocks = transformedHtml.split(/\n\s*\n+/).map(b => b.trim()).filter(Boolean);
  if (!blocks.length) return "";
  const firstHtml = mdToHtml(blocks[0]);
  const tldrInner = firstHtml.replace(/^<p>([\s\S]*)<\/p>$/, "$1");
  const restMd = blocks.slice(1).join("\n\n");
  let proseHtml = "";
  if (restMd.trim()) {
    proseHtml = mdToHtml(restMd);
    proseHtml = wrapInlineNumbers(proseHtml);
  }
  return `
    <div class="fs-tldr">
      <div class="l">Bottom line</div>
      <div class="body">${tldrInner}</div>
    </div>
    ${proseHtml ? `
      <div class="fs-h">
        <h3>Detailed answer</h3>
        <div class="rule"></div>
        <span class="badge">FinSage · v28</span>
      </div>
      <div class="fs-prose">${proseHtml}</div>
    ` : ""}
  `;
}

// ────── Render: sources strip ──────
function dedupeCitations(citations) {
  const seen = new Set();
  const out = [];
  for (const c of citations || []) {
    const key = [c.ticker, c.fiscal_year, c.filing_type, c.section_name].join("|");
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(c);
    if (out.length >= 8) break;
  }
  return out;
}

function renderSources(citations) {
  const cards = dedupeCitations(citations);
  if (!cards.length) return "";
  const items = cards.map((c, i) => {
    const badge = escapeHtml(c.filing_type || "FILING");
    const tickerStr = escapeHtml(c.ticker || "—");
    const fyStr = c.fiscal_year ? `FY${parseInt(c.fiscal_year, 10)}` : "";
    const titleStr = escapeHtml(c.section_name || "Section");
    const scoreStr = c.score ? `${(c.score * 100).toFixed(0)}% match` : "";
    return `
      <div class="fs-src-card" data-src-idx="${i}">
        <span class="badge">${badge}</span>
        <span class="ticker">${tickerStr}${fyStr ? " · " + fyStr : ""}</span>
        <span class="ttl">${titleStr}</span>
        <span class="meta"><span>${escapeHtml(scoreStr)}</span><span>↗</span></span>
      </div>
    `;
  }).join("");
  return `
    <div class="fs-h">
      <h3>Sources</h3>
      <div class="rule"></div>
      <span class="badge">${cards.length} filing passage${cards.length !== 1 ? "s" : ""}</span>
    </div>
    <div class="fs-src-grid">${items}</div>
  `;
}

// ────── Render: full canvas ──────
function renderThread() {
  const cur = getCurrentThread();
  if (!cur || cur.messages.length === 0) {
    // Empty state — show hero
    hero.style.display = "";
    thread.style.display = "none";
    thread.innerHTML = "";
    crumbCurrent.textContent = "New research";
    topbarTicker.style.display = "none";
    topbarStatus.style.display = "none";
    return;
  }

  const { userMsg, asstMsg } = getLatestExchange();

  hero.style.display = "none";
  thread.style.display = "";

  const now = new Date(cur.createdAt).toLocaleString("en-US", {
    month: "short", day: "numeric", year: "numeric",
    hour: "2-digit", minute: "2-digit", hour12: false,
  });

  let questionHtml = "";
  if (userMsg) {
    questionHtml = `
      <div class="fs-user-q">
        <div class="fs-eyebrow">Question · ${now}</div>
        <div class="fs-q-text">${italicizeEntity(userMsg.content)}</div>
      </div>
    `;
  }

  let answerHtml = "";
  // Critical: show LOADING when inflight AND no assistant for this user
  if (userMsg && !asstMsg && cur.inflight) {
    answerHtml = `
      <div class="fs-loading">
        <span class="dots"><span></span><span></span><span></span></span>
        FinSage is consulting the filings…
      </div>
    `;
  } else if (asstMsg && asstMsg.error) {
    answerHtml = `<div class="fs-error">${escapeHtml(asstMsg.error)}</div>`;
  } else if (asstMsg) {
    const cleaned = stripInlineTags(asstMsg.content);
    const { html: transformed, cites } = transformCitations(cleaned, asstMsg.citations || []);
    const timelineHtml = renderTimeline(asstMsg.raw_messages || []);
    const tldrProseHtml = renderTldrAndProse(transformed);
    const sourcesHtml = renderSources(asstMsg.citations || []);
    answerHtml = `${timelineHtml}${tldrProseHtml}${sourcesHtml}`;
    // Stash deduped cards on the thread so the source-card click handlers
    // can look up by idx without re-deduping
    cur._cards = dedupeCitations(asstMsg.citations || []);
    cur._inlineCites = cites || {};
    cur._metricsEvidence = extractMetricsEvidence(asstMsg.raw_messages || []);
    cur._metadataEvidence = extractMetadataEvidence(asstMsg.raw_messages || []);
  }

  thread.innerHTML = questionHtml + answerHtml;

  // Topbar updates from current thread state
  if (userMsg) {
    crumbCurrent.textContent =
      userMsg.content.slice(0, 80) + (userMsg.content.length > 80 ? "…" : "");
  }
  if (cur.ticker) {
    topbarTicker.style.display = "";
    topbarTicker.querySelector(".ticker").textContent = cur.ticker;
  } else {
    topbarTicker.style.display = "none";
  }
  if (asstMsg && asstMsg.citations && asstMsg.citations.length) {
    const filings = new Set();
    asstMsg.citations.forEach(c =>
      filings.add([c.ticker, c.fiscal_year, c.filing_type].join("|"))
    );
    topbarStatus.style.display = "";
    topbarStatus.querySelector(".text").textContent =
      `${filings.size} filing${filings.size !== 1 ? "s" : ""} indexed`;
  } else {
    topbarStatus.style.display = "none";
  }

  // Source card click handlers
  thread.querySelectorAll(".fs-src-card").forEach(card => {
    card.addEventListener("click", () => {
      const idx = parseInt(card.dataset.srcIdx, 10);
      const c = (cur._cards || [])[idx];
      if (c) openSourceModal(c);
    });
  });

  // Inline [Source: ...] citation chips can open either section drill-in
  // or structured metrics provenance.
  thread.querySelectorAll(".fs-cite[data-cite]").forEach(el => {
    el.addEventListener("click", () => {
      const idx = parseInt(el.dataset.cite, 10);
      const c = (cur._inlineCites || {})[idx];
      if (!c) return;
      if (c.source_kind === "section") openSourceModal(c);
      else if (c.source_kind === "metrics") openMetricsModal(c);
      else if (c.source_kind === "metadata") openMetadataModal(c);
    });
  });

  // Agent timeline collapse toggle
  thread.querySelectorAll('[data-toggle="agent"]').forEach(head => {
    head.addEventListener("click", () => {
      head.parentElement.classList.toggle("collapsed");
    });
  });

  canvasScroll.scrollTop = 0;
}

// ────── Render: sidebar thread list ──────
function renderSidebar() {
  // Group threads by recency. For demo simplicity: "Today" = all real
  // threads; "Sample" = demo placeholders that never become active. This
  // matches the handoff visual without faking persistence we don't have.
  const realThreads = state.threads;

  const today = realThreads.map(t => {
    const ticker = t.ticker || "—";
    const isCurrent = t.id === state.currentThreadId;
    const title = t.title || "New research";
    const when = isCurrent ? "now" : new Date(t.createdAt).toLocaleTimeString("en-US", {
      hour: "2-digit", minute: "2-digit", hour12: false,
    });
    return `
      <div class="fs-item${isCurrent ? ' active' : ''}" data-thread-id="${t.id}">
        ${escapeHtml(title)}
        <div class="meta"><span class="tag">${escapeHtml(ticker)}</span> · ${escapeHtml(when)}</div>
      </div>
    `;
  }).join("");

  const samples = `
    <div class="fs-section">Sample threads</div>
    <div class="fs-item">Apple Services growth & 10-K risks
      <div class="meta"><span class="tag">AAPL</span> · sample</div>
    </div>
    <div class="fs-item">NVIDIA supply chain exposure
      <div class="meta"><span class="tag">NVDA</span> · sample</div>
    </div>
    <div class="fs-item">MSFT vs GOOGL operating margin walk
      <div class="meta"><span class="tag">COMP</span> · sample</div>
    </div>
  `;

  threadList.innerHTML = `
    <div class="fs-section">Today</div>
    ${today}
    ${samples}
  `;

  // Bind clicks on real (non-sample) threads
  threadList.querySelectorAll('.fs-item[data-thread-id]').forEach(el => {
    el.addEventListener("click", () => {
      const id = el.dataset.threadId;
      if (id && id !== state.currentThreadId) {
        switchToThread(id);
        renderSidebar();
        renderThread();
      }
    });
  });
}

// ────── Modal: source drill-in ──────
function fmtMetricValue(key, val) {
  if (val == null) return "N/A";
  const pctFields = new Set(["gross_margin_pct", "revenue_yoy_growth_pct", "data_quality_score"]);
  if (pctFields.has(key)) return `${(Number(val) * 100).toFixed(1)}%`;
  if (key === "debt_to_equity") return `${Number(val).toFixed(2)}x`;
  if (typeof val === "number") {
    if (Math.abs(val) >= 1e9) return `$${(val / 1e9).toFixed(2)}B`;
    if (Math.abs(val) >= 1e6) return `$${(val / 1e6).toFixed(1)}M`;
    return `$${val.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
  }
  if (!Number.isNaN(Number(val)) && val !== "") return fmtMetricValue(key, Number(val));
  return String(val);
}

async function openSourceModal(c) {
  const reqSeq = ++modalRequestSeq;
  const fyStr = c.fiscal_year ? `FY${parseInt(c.fiscal_year, 10)}` : "";
  modalBadges.innerHTML = `
    <span class="fs-chip gold"><span class="ticker">${escapeHtml(c.ticker || "?")}${fyStr ? " · " + fyStr : ""}</span></span>
    ${c.filing_type ? `<span class="fs-chip"><span class="ticker">${escapeHtml(c.filing_type)}</span></span>` : ""}
    ${c.section_name ? `<span class="fs-chip"><span class="ticker">${escapeHtml(c.section_name)}</span></span>` : ""}
  `;
  modalBody.innerHTML = `<div class="fs-modal-loading">Loading SEC filing section…</div>`;
  modalBackdrop.classList.add("open");

  if (!c.ticker || !c.fiscal_year || !c.section_name) {
    modalBody.innerHTML = `<div class="fs-modal-error">Citation is missing identifying metadata.</div>`;
    return;
  }

  try {
    const params = new URLSearchParams({
      ticker: c.ticker,
      fiscal_year: String(parseInt(c.fiscal_year, 10)),
      section_name: c.section_name,
    });
    if (c.filing_type) params.set("filing_type", c.filing_type);
    const res = await fetch(`/api/section?${params.toString()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (reqSeq !== modalRequestSeq) return;
    const section = data.section_text || "";
    const chunk = c.chunk_text || "";
    if (!section) {
      modalBody.innerHTML = `
        <div class="fs-modal-error">
          Full section text isn't materialized in silver for this row.
          The agent still grounded its answer in this passage:
        </div>
        <div style="margin-top:14px"><mark>${escapeHtml(chunk)}</mark></div>
      `;
      return;
    }
    const sectionEsc = escapeHtml(section);
    const chunkEsc = escapeHtml(chunk);
    let highlighted = sectionEsc;
    if (chunkEsc) {
      let idx = sectionEsc.indexOf(chunkEsc);
      let matchText = chunkEsc;
      if (idx < 0 && chunkEsc.length > 200) {
        matchText = chunkEsc.slice(0, 200);
        idx = sectionEsc.indexOf(matchText);
      }
      if (idx >= 0) {
        highlighted =
          sectionEsc.slice(0, idx) +
          `<mark>${matchText}</mark>` +
          sectionEsc.slice(idx + matchText.length);
      }
    }
    modalBody.innerHTML = highlighted.replace(/\n/g, "<br>");
    setTimeout(() => {
      const mark = modalBody.querySelector("mark");
      if (mark) mark.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 80);
  } catch (err) {
    if (reqSeq !== modalRequestSeq) return;
    modalBody.innerHTML = `<div class="fs-modal-error">Failed to load section: ${escapeHtml(err.message || String(err))}</div>`;
  }
}

async function openMetricsModal(c) {
  const reqSeq = ++modalRequestSeq;
  const fyStr = c.fiscal_year ? `FY${parseInt(c.fiscal_year, 10)}` : "";
  const fqStr = c.fiscal_quarter ? `Q${parseInt(c.fiscal_quarter, 10)}` : "";
  modalBadges.innerHTML = `
    <span class="fs-chip gold"><span class="ticker">${escapeHtml(c.ticker || "?")}${fyStr ? " · " + fyStr : ""}${fqStr ? " · " + fqStr : ""}</span></span>
    <span class="fs-chip"><span class="ticker">metrics</span></span>
  `;
  modalBody.innerHTML = `<div class="fs-modal-loading">Loading structured metrics source…</div>`;
  modalBackdrop.classList.add("open");

  if (!c.ticker || !c.fiscal_year) {
    modalBody.innerHTML = `<div class="fs-modal-error">Metrics citation is missing ticker/year metadata.</div>`;
    return;
  }
  try {
    const params = new URLSearchParams({
      ticker: c.ticker,
      fiscal_year: String(parseInt(c.fiscal_year, 10)),
    });
    if (c.fiscal_quarter) params.set("fiscal_quarter", String(parseInt(c.fiscal_quarter, 10)));
    const res = await fetch(`/api/metrics?${params.toString()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (reqSeq !== modalRequestSeq) return;
    if (data.error) {
      // Fall back to tool-output evidence if SQL warehouse isn't configured
      // locally (common for localhost runs).
      const cur = getCurrentThread();
      const evidences = (cur && cur._metricsEvidence) || [];
      const fyNeedle = c.fiscal_quarter
        ? `FY${parseInt(c.fiscal_year, 10)} Q${parseInt(c.fiscal_quarter, 10)}`
        : `FY${parseInt(c.fiscal_year, 10)}`;
      const tickerNeedle = String(c.ticker || "").toUpperCase();
      const best = evidences.find(e => {
        const txt = (e.content || "").toUpperCase();
        return txt.includes(tickerNeedle) && txt.includes(fyNeedle.toUpperCase());
      }) || evidences[0];
      if (best && best.content) {
        modalBody.innerHTML = `
          <div style="margin-bottom:10px;color:#b8bcc7;font-size:13px">
            Structured metrics evidence from tool output (${escapeHtml(best.tool_name)}).
          </div>
          <pre style="white-space:pre-wrap;line-height:1.45;font-size:12.5px;color:#d8d9de;background:#14141d;border:1px solid #2a2a35;border-radius:8px;padding:12px">${escapeHtml(best.content)}</pre>
        `;
        return;
      }
      modalBody.innerHTML = `<div class="fs-modal-error">${escapeHtml(data.error)}</div>`;
      return;
    }
    const row = data.row || {};
    const sourceTable = data.source_table || "main.finsage_gold.company_metrics";
    const displayOrder = [
      "revenue",
      "net_income",
      "gross_profit",
      "operating_income",
      "operating_cash_flow",
      "total_assets",
      "total_liabilities",
      "total_equity",
      "total_debt",
      "rd_expense",
      "gross_margin_pct",
      "revenue_yoy_growth_pct",
      "debt_to_equity",
      "data_quality_score",
    ];
    const rowsHtml = displayOrder.map(key => `
      <tr>
        <td style="padding:6px 10px;border-bottom:1px solid rgba(255,255,255,0.08)">${escapeHtml(key)}</td>
        <td style="padding:6px 10px;border-bottom:1px solid rgba(255,255,255,0.08);text-align:right">${escapeHtml(fmtMetricValue(key, row[key]))}</td>
      </tr>
    `).join("");
    modalBody.innerHTML = `
      <div style="margin-bottom:10px;color:#b8bcc7;font-size:13px">
        Structured financial metrics provenance from
        <code>${escapeHtml(sourceTable)}</code>.
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:13px">
        <tbody>${rowsHtml}</tbody>
      </table>
    `;
  } catch (err) {
    if (reqSeq !== modalRequestSeq) return;
    modalBody.innerHTML = `<div class="fs-modal-error">Failed to load metrics source: ${escapeHtml(err.message || String(err))}</div>`;
  }
}

function openMetadataModal(c) {
  const fyStr = c.fiscal_year ? `FY${parseInt(c.fiscal_year, 10)}` : "";
  modalBadges.innerHTML = `
    <span class="fs-chip gold"><span class="ticker">${escapeHtml(c.ticker || "?")}${fyStr ? " · " + fyStr : ""}</span></span>
    <span class="fs-chip"><span class="ticker">10-K Cover Page</span></span>
  `;
  modalBackdrop.classList.add("open");
  const cur = getCurrentThread();
  const evidences = (cur && cur._metadataEvidence) || [];
  const fyNeedle = fyStr || "";
  const tickerNeedle = String(c.ticker || "").toUpperCase();
  const best = evidences.find(e => {
    const txt = (e.content || "").toUpperCase();
    return txt.includes(tickerNeedle) && (!fyNeedle || txt.includes(fyNeedle.toUpperCase()));
  }) || evidences[0];

  if (!best || !best.content) {
    modalBody.innerHTML = `<div class="fs-modal-error">No cover-page metadata tool output is available for this citation.</div>`;
    return;
  }
  modalBody.innerHTML = `
    <div style="margin-bottom:10px;color:#b8bcc7;font-size:13px">
      Filing metadata evidence from tool output (${escapeHtml(best.tool_name)}).
    </div>
    <pre style="white-space:pre-wrap;line-height:1.45;font-size:12.5px;color:#d8d9de;background:#14141d;border:1px solid #2a2a35;border-radius:8px;padding:12px">${escapeHtml(best.content)}</pre>
  `;
}

function closeModal() {
  modalRequestSeq++;
  modalBackdrop.classList.remove("open");
}

modalClose.addEventListener("click", closeModal);
modalBackdrop.addEventListener("click", e => {
  if (e.target === modalBackdrop) closeModal();
});
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && modalBackdrop.classList.contains("open")) closeModal();
});

// ────── Composer: submit ──────
async function dispatch(text) {
  text = (text || "").trim();
  if (!text) return;

  let cur = getCurrentThread();
  if (!cur) {
    cur = freshThread();
    state.threads.unshift(cur);
    state.currentThreadId = cur.id;
  }
  if (cur.inflight) return;

  cur.requestSeq = (cur.requestSeq || 0) + 1;
  const requestSeq = cur.requestSeq;
  cur.inflight = true;
  cur.messages.push({ role: "user", content: text });
  // First user message in this thread → title + ticker
  if (cur.messages.filter(m => m.role === "user").length === 1) {
    cur.title = text.length > 80 ? text.slice(0, 80) + "…" : text;
    cur.ticker = tickerFromText(text);
  }
  composerInput.value = "";
  autosizeTextarea();
  renderSidebar();
  renderThread();

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: cur.messages.map(m => ({ role: m.role, content: m.content })),
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (cur.requestSeq !== requestSeq) return;
    cur.messages.push({
      role: "assistant",
      content: data.content || "(no response)",
      citations: data.citations || [],
      raw_messages: data.messages || [],
      error: data.error || null,
    });
  } catch (err) {
    if (cur.requestSeq !== requestSeq) return;
    cur.messages.push({
      role: "assistant",
      content: "",
      citations: [],
      raw_messages: [],
      error: `Endpoint error: ${err.message || err}`,
    });
  } finally {
    if (cur.requestSeq === requestSeq) cur.inflight = false;
    renderSidebar();
    renderThread();
  }
}

composer.addEventListener("submit", e => {
  e.preventDefault();
  dispatch(composerInput.value);
});
composerInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    dispatch(composerInput.value);
  }
});
function autosizeTextarea() {
  composerInput.style.height = "auto";
  composerInput.style.height = Math.min(composerInput.scrollHeight, 160) + "px";
}
composerInput.addEventListener("input", autosizeTextarea);

// ────── Example chips ──────
examples.forEach(btn => {
  btn.addEventListener("click", () => {
    const q = btn.dataset.question || "";
    composerInput.value = q;
    autosizeTextarea();
    dispatch(q);
  });
});

// ────── New research ──────
newBtn.addEventListener("click", () => {
  createNewResearch();
  composerInput.value = "";
  autosizeTextarea();
  closeModal();
  renderSidebar();
  renderThread();
});

// ────── Sidebar collapse ──────
sidebarColl.addEventListener("click", () => app.classList.add("sidebar-collapsed"));
sidebarReopen.addEventListener("click", () => app.classList.remove("sidebar-collapsed"));

// ────── Cmd+N / Ctrl+N → New research ──────
document.addEventListener("keydown", e => {
  if ((e.metaKey || e.ctrlKey) && e.key === "n") {
    e.preventDefault();
    newBtn.click();
  }
});

// Initial render
renderSidebar();
renderThread();
