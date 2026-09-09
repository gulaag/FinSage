# FinSage

> **Trust is what you sell in Japan.**

In Japanese retail finance, a number without a source is worthless. FinSage is built around that single principle: every figure it returns is traceable to the exact SEC filing, fiscal year, and section it came from. It is a production-grade **agentic financial-intelligence system** that turns **80,000+ pages** of raw SEC filings into a **function-calling AI agent** that reasons like an analyst and cites like an auditor.

FinSage is not a thin wrapper around a chat model. It is a **governed reasoning system** — a tool-using agent with an owned orchestration loop, a retrieval-augmented generation (RAG) stack, hard hallucination controls, full MLflow tracing, and an offline evaluation harness that scores every deployed version against a locked gold set. The Medallion data pipeline underneath exists for one reason: to feed the agent audited, citation-grade evidence.

| Metric | Value |
|---|---|
| Filings processed | 728 (10-K + 10-Q) |
| Pages of source text | 80,000+ |
| Companies | 30 large-cap, 6 sectors |
| Fiscal years | 2020 to 2026 |
| Vector embeddings | 45,136 BGE-large chunks |
| Agent tools | 4 (function calling) |
| Evaluation scorers | 8 (2 LLM-as-judge + 6 programmatic) |
| Gold evaluation set | 100 stratified questions, SEC-verified |
| Answer accuracy | 87% on the 100-question gold set |
| Citation coverage | 100% of grounded answers |
| Hallucination rate | Near zero (structured metrics never come from RAG) |

### AI / ML engineering stack at a glance

**Agent:** function calling · tool use · ReAct-style reason–act–observe loop · owned orchestration harness (zero LangChain) · deterministic decoding · bounded iterations · wall-clock guardrails
**Retrieval (RAG):** Databricks Vector Search · approximate-nearest-neighbour (ANN) semantic search · BGE-large-en embeddings · similarity-threshold quality gate with permissive-retry fallback · metadata-filtered retrieval · token-aware, section-aware chunking · grounding + citation enforcement
**Reasoning model:** Claude Sonnet 4.6 (function-calling) on Databricks Model Serving
**Evaluation (LLMOps):** `mlflow.genai.evaluate` · LLM-as-a-judge · programmatic scorers · stratified gold set · regression tracking · preflight gating
**Observability & MLOps:** MLflow Tracing (RETRIEVER / TOOL spans) · Unity Catalog Model Registry · Model Serving (scale-to-zero) · automated redeploy job · Databricks Asset Bundle · GitHub Actions CI/CD

---

## Table of Contents

1. [Agent Architecture — the AI core](#1-agent-architecture--the-ai-core)
2. [Retrieval-Augmented Generation](#2-retrieval-augmented-generation)
3. [Guardrails and Hallucination Controls](#3-guardrails-and-hallucination-controls)
4. [Observability and MLOps](#4-observability-and-mlops)
5. [Evaluation — the 100-question harness](#5-evaluation--the-100-question-harness)
6. [Data Engineering: the Medallion substrate](#6-data-engineering-the-medallion-substrate)
7. [Databricks Asset Bundle](#7-databricks-asset-bundle)
8. [CI/CD with GitHub Actions](#8-cicd-with-github-actions)
9. [Branch Strategy](#9-branch-strategy)
10. [Local Development](#10-local-development)
11. [Directory Structure](#11-directory-structure)
12. [Environment and Secrets](#12-environment-and-secrets)
13. [Future Work](#13-future-work)
14. [License](#14-license)

---

## 1. Agent Architecture — the AI core

This is the heart of FinSage. The data pipeline exists to feed it. The agent is a **governed reasoning system, not a thin wrapper around a language model**, and it is engineered so that a wrong or unsourced answer is structurally hard to produce.

### 1.1 Design at a glance

| Property | Choice | Why it matters |
|---|---|---|
| Framework | `mlflow.pyfunc.PythonModel` | **Zero LangChain dependency.** The orchestration loop is owned, inspectable, and portable across serving runtimes. No hidden agent framework magic. |
| Reasoning model | Claude Sonnet 4.6 (function calling), on Databricks Model Serving | Strong tool selection and instruction following, hosted inside the same governed workspace as the data. |
| Control pattern | Function-calling agent loop (ReAct-style: reason → act → observe → repeat) | Lets a single question drive multiple tool calls until the agent has gathered enough grounded evidence to answer. |
| Decoding | `temperature = 0.0` (deterministic) | Reproducible answers; regressions are attributable to code, not sampling noise. |
| Registration | Unity Catalog model `main.finsage_gold.finsage_rag_agent` | Versioned, governed, and promotable like any other UC asset. |
| Serving | Model Serving endpoint `finsage_agent_endpoint` | Scales to zero when idle, redeployed automatically by a Databricks Job. |
| Observability | MLflow Tracing on every call | Each tool call and model call emits a span, so any answer can be replayed and audited. |

### 1.2 The agent control loop

Every question flows through **one deterministic loop**. There are no regex shortcuts and no hidden fast paths, so behavior is uniform and testable. Crucially, **the harness — not the model — executes the tools.** That separation is the control surface: the harness validates arguments, applies retrieval thresholds, retries, logs spans, and enforces the output contract.

```
User question
      │
      ▼
┌───────────────────────────────────────────────────────────────┐
│  System prompt (routing policy, refusal policy, citation       │
│  contract, worked examples) is prepended to the conversation   │
└───────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
        ┌──────── Agent loop, up to MAX_ITERATIONS = 5 ─────────┐
        │                                                        │
        │   1. REASON   model reads history + tool schemas       │
        │   2. ACT      model emits a structured tool call        │
        │   3. OBSERVE  harness validates args, runs the tool,    │
        │               applies thresholds, captures the result  │
        │   4. FEED     tool result is appended to the messages   │
        │                                                        │
        │   Repeat until the model stops calling tools, the       │
        │   iteration budget is hit, or a guardrail fires.        │
        └───────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
        ┌───────────────────────────────────────────────────────┐
        │  Citation-enforcement pass                              │
        │  guarantees every grounded answer carries a real,      │
        │  retrieved [Source: ...] line                           │
        └───────────────────────────┬───────────────────────────┘
                                     ▼
                          Answer shown to user
```

### 1.3 The four tools (function calling)

The agent is given **exactly four tools**, described to the model in OpenAI function-calling schema form and dispatched through an owned `TOOL_DISPATCH` registry. Fewer, well-scoped tools produce more reliable tool selection than many overlapping ones.

| Tool | Purpose | Source | MLflow span |
|---|---|---|---|
| `search_filings` | Semantic retrieval over 10-K / 10-Q narrative text for **qualitative** questions (strategy, risks, products, competition, regulation, supply chain). Filterable by ticker, section, fiscal year, and filing type. | Vector Search index `filing_chunks_index` | `RETRIEVER` |
| `get_company_metrics` | **Annual** structured financials: revenue, net income, operating income, gross profit, cash flow, assets, liabilities, equity, debt, R&D, margins, YoY growth, debt-to-equity. | Gold `company_metrics`, loaded to an in-memory cache at serving time | `TOOL` |
| `get_quarterly_metrics` | **Discrete** Q1/Q2/Q3 metrics plus same-quarter YoY trends. | Gold `company_metrics_quarterly` | `TOOL` |
| `get_filing_metadata` | Deterministic cover-page facts: filing date, employee count, shares outstanding. | Silver-derived metadata cache | `TOOL` |

**The single most important AI-design decision: numbers never come from RAG.** Financial figures are served from structured Gold tables through the metrics tools, so they are exact and cannot be hallucinated. Retrieval is reserved for **language**, where free-text semantic search is the right instrument. This split is the biggest single reason the hallucination rate is near zero — and it is a deliberate architectural choice most "RAG wrappers" never make.

---

## 2. Retrieval-Augmented Generation

`search_filings` is a real retrieval stack, not a one-line vector lookup.

- **Semantic search** runs as an **approximate-nearest-neighbour (ANN)** query against the Databricks Vector Search index `filing_chunks_index`, backed by **`databricks-bge-large-en`** embeddings over **45,136 chunks**.
- **Metadata-filtered retrieval:** queries can be scoped by `ticker`, `section_name`, `fiscal_year`, and `filing_type` (10-K vs 10-Q), so the agent never mixes years, companies, or annual/interim language. Column and filter handling is adapted at runtime to stay compatible with the serving index.
- **Retrieval quality gate:** results below a **similarity threshold of `0.4`** are dropped so weak, off-topic passages are never cited as evidence. If strict filtering removes *everything*, the harness performs **a single permissive retry at threshold `0.0`** so a hard query degrades gracefully instead of returning nothing.
- **Grounding + citation:** every surfaced passage is emitted with a structured `[Source: TICKER | FY#### | 10-K/10-Q | Section]` line, which the citation-enforcement pass guarantees reaches the final answer.

> Note on ranking: retrieval currently uses **similarity-scored ANN with a threshold quality gate and permissive-retry fallback**. There is no separate cross-encoder re-ranking stage today — a learned re-ranker is on the roadmap (see Future Work).

### Token-aware, section-aware chunking

The chunker (notebook 05) splits **each Silver section independently, never across section boundaries**, into **512-token chunks with 64 tokens of overlap** using the **`tiktoken`** tokenizer, distributed as a Spark pandas UDF. Section-aware chunking preserves the strong semantic structure of financial filings, and **deterministic SHA-256 `chunk_id`s** make re-indexing fully idempotent. The chunks are embedded with BGE-large and served through the Vector Search index that powers `search_filings`.

---

## 3. Guardrails and Hallucination Controls

Production safety is enforced by the harness, on the assumption that the model will occasionally misbehave.

| Guardrail | Setting | Protects against |
|---|---|---|
| Bounded agent loop | `MAX_ITERATIONS = 5` | Infinite reasoning loops and runaway cost |
| Wall-clock timeout | 150 seconds per request | Latency blowouts on pathological queries |
| Retrieval quality gate | Similarity threshold `0.4`, with a single permissive retry at `0.0` | Weak, off-topic passages being cited as evidence |
| Structured-metrics separation | Numbers served only from Gold tables, never from RAG | Fabricated or drifted financial figures |
| Citation enforcement | Post-processing pass attaches a real, retrieved `[Source: ...]` line if the model forgot | Ungrounded or unsourced answers reaching the user |
| Refusal policy | System prompt declines out-of-corpus tickers, future fiscal years, and missing data | Fabricated values for data the platform does not hold |
| Deterministic decoding | `temperature = 0.0` | Non-reproducible answers |

### The citation contract

Because trust is the product, the output format is a **contract**, not a preference:

- Metric answers end with `[Source: TICKER | FY#### | metrics]` (or `FY#### Q#` for quarterly).
- Filing-text answers are marked `[VERBATIM]` or `[SUMMARY]` and carry `[Source: TICKER | FY#### | 10-K/10-Q | Section]`.
- Cover-page facts carry `[Source: TICKER | FY#### | 10-K Cover Page]`.
- When data is absent, the agent refuses clearly and states why, rather than guessing.

### Prompt engineering, routing, and refusals

The system prompt is a governance artifact. It encodes a **tool-routing policy** (annual → `get_company_metrics`; quarterly → `get_quarterly_metrics`; cover-page → `get_filing_metadata`; qualitative narrative → `search_filings`), a **refusal policy** with explicit decline conditions, the **citation contract**, formula disclosure on first use, and worked examples — several of them refusal-shaped — so the model learns the exact shape of a correct answer and a correct refusal.

---

## 4. Observability and MLOps

Every `predict` call, every retrieval, and every metrics lookup emits an **MLflow span** — `search_filings` is traced with `span_type="RETRIEVER"`, the metrics/metadata tools with `span_type="TOOL"` — so any production answer can be traced end to end and replayed. This is also what lets the evaluation harness assert *retrieval groundedness* only on traces that actually invoked retrieval.

The agent is **registered in Unity Catalog** and deployed to a **Model Serving endpoint that scales to zero**. Redeployment is automated through a **Databricks Job** that runs the agent notebook end to end on serverless compute, so shipping a new version is a single governed action rather than a manual sequence.

---

## 5. Evaluation — the 100-question harness

FinSage treats agent quality as an engineering discipline with an **offline evaluation harness**, not a vibe check. This is the crown jewel of the AI engineering.

### 5.1 The gold set

Notebook `07_evaluation.py` scores the deployed agent against a **locked, stratified gold set of 100 questions** spanning all 30 tickers, every verified against SEC EDGAR. The set is stratified across annual numerical lookups, quarterly lookups, year-over-year comparisons, multi-company comparisons, filing-metadata / citation validation, and refusal tests, so no single behaviour dominates the score.

### 5.2 Eight scorers (LLM-as-judge + programmatic)

Each answer is graded by **eight scorers — two built-in LLM-as-a-judge scorers and six custom programmatic scorers** — run through `mlflow.genai.evaluate`:

| # | Scorer | Type | What it checks |
|---|---|---|---|
| 1 | Correctness | LLM judge | Factual correctness against ground truth |
| 2 | Cites ticker and year | LLM judge (Guidelines) | Presence of the required identifying citation |
| 3 | Numerical tolerance | Programmatic | Values within ±1%, with unit-aware (B/M/K) extraction |
| 4 | Citation format | Programmatic | Correct `[VERBATIM]` / `[SUMMARY]` and `[Source: ...]` shape |
| 5 | Refusal correctness | Programmatic | Refusals happen **for the right reason** (per-question expected context) |
| 6 | Tool routing correctness | Programmatic | The agent chose the **correct tool** for the question |
| 7 | Derived metric match | Programmatic | Derived ratios match independent of natural-language wording |
| 8 | Retrieval grounded when used | Programmatic | Answers cite the retrieved evidence — **skipped** when no `RETRIEVER` span exists |

The judge model is `databricks-meta-llama-3-3-70b-instruct`. The programmatic scorers are decoupled from LLM-judge phrasing on purpose: they catch value/routing regressions that an over-strict or over-lenient judge would misclassify.

### 5.3 Harness engineering

- **In-process model loading** via `unwrap_python_model()` returns the raw agent instance, bypassing pyfunc's dict coercion so the tool decorators emit real child spans *inside the eval process* — which is what makes retrieval-groundedness scoring possible.
- **Idempotent persistence** to two Delta tables (`eval_run_summaries`, one row per run; `eval_question_outcomes`, one row per (run, question, scorer)), so every run is versioned, comparable, and **regression-diffable across agent versions**.
- **Analysis module** (`src/evaluation/analysis.py`): pure-Spark `summarize_run`, `failure_breakdown`, `category_matrix`, `regression_diff`, and `question_flips`.
- **Preflight gating:** a 5-question smoke subset runs before the full 100, and a `<1s` pytest preflight suite (`tests/unit/test_eval_preflight.py`) catches dataset, schema, and scorer-wiring regressions *before a cluster is ever started*.

Current headline results: **87% accuracy, 100% citation coverage on grounded answers, and a near-zero hallucination rate.**

### Unit and preflight tests

Located in `tests/unit/`, these run as plain Python with no Spark dependency and cover XBRL concept normalization, the evaluation scorers, and the preflight suite:

```bash
pytest tests/unit/ -v
```

---

## 6. Data Engineering: the Medallion substrate

The agent's evidence is produced by a five-stage **Medallion pipeline** on Databricks, deployed as a Databricks Asset Bundle and orchestrated as a sequential job. All layers are Delta Lake tables in Unity Catalog under the `main` catalog. It exists to feed the agent audited data — so it is deliberately summarized here.

```
SEC EDGAR / CompanyFacts API
      │
      ▼
[01 Schema Setup] → [02 Bronze] → [03 Silver] → [04 Gold] → [05 Vector Chunker] → [06 Agent] → [07 Eval]
                     Auto Loader    sec-parser +   annual +      tiktoken +
                     + XBRL API     XBRL flatten   quarterly     VS index
```

- **Bronze (`main.finsage_bronze`)** — raw, append-only, auditable ingestion. `filings` (raw bytes via Auto Loader `cloudFiles`, checkpointed exactly-once, `availableNow=True`), `xbrl_companyfacts_raw` (SEC CompanyFacts JSON), `ingestion_errors`, and an idempotency ledger `sec_filings_download_log`. Change Data Feed is enabled for incremental downstream consumption.
- **Silver (`main.finsage_silver`)** — `financial_statements` flattens deeply nested XBRL CompanyFacts JSON into canonical metrics via `TARGET_CONCEPT_MAP` (SHA-256 `statement_id`, idempotent MERGE); `filing_sections` extracts named narrative sections using **`sec-parser`** (a DOM-aware iXBRL parser) with a per-section regex fallback. Operating on the parsed document tree — not flattened text — lets it decode HTML entities natively, survive inline-XBRL span fragmentation, and drop headers/footers by semantic role. Per-row extractor provenance is preserved.
- **Gold (`main.finsage_gold`)** — `company_metrics` (annual KPIs incl. margins, YoY growth, debt-to-equity, `data_quality_score`), `company_metrics_quarterly` (discrete Q1–Q3 derived from cumulative YTD XBRL by subtraction), `filing_section_chunks` (deterministic-ID token chunks), and the `filing_chunks_index` Vector Search index. Strict accounting discipline: full-year window for flow metrics, point-in-time for balance-sheet metrics, one canonical filing per (ticker, fiscal year) so amendments never mix into a row.

---

## 7. Databricks Asset Bundle

FinSage is deployed as a Databricks Asset Bundle (DAB). Job topology, cluster configuration, schedule, and environment promotions are version-controlled in `databricks.yml`. The bundle defines a sequential job whose tasks run in strict order via `depends_on`:

```yaml
bundle:
  name: finsage_pipeline

resources:
  jobs:
    finsage_daily_run:
      tasks:
        - task_key: schema_setup        # 01_schema_setup.py
        - task_key: bronze_autoloader   # 02_bronze_autoloader.py  (depends_on: schema_setup)
        - task_key: silver_decoder      # 03_silver_decoder.py     (depends_on: bronze_autoloader)
        - task_key: gold_metrics        # 04_gold_metrics.py       (depends_on: silver_decoder)
        - task_key: vector_chunker      # 05_vector_chunker.py     (depends_on: gold_metrics)
```

If any task fails, downstream tasks are skipped and a failure alert is sent. The job is scheduled in production and paused in the `dev` target.

### Bundle variables

| Variable | Default | Description |
|---|---|---|
| `catalog` | `main` | Unity Catalog catalog name |
| `env` | `dev` | Environment label (`dev` / `prod`) |
| `start_date` | `2020-01-01` | Earliest SEC filing date to ingest |
| `ticker_filter` | `""` | Comma-separated tickers to process (empty = all 30) |
| `notification_email` | configured per deploy | Recipient for job failure alerts |

### Common commands

```bash
databricks bundle validate            # syntax + workspace connection check
databricks bundle deploy              # deploy to dev (default target)
databricks bundle deploy -t prod      # deploy to prod
databricks bundle run finsage_daily_run
```

---

## 8. CI/CD with GitHub Actions

The workflow lives in `.github/workflows/deploy.yml` and gates production behind tests and validation.

```
push to main → unit-tests (pytest) → bundle-validate → deploy-prod
                     │ failure
                     ▼
             Pipeline halted, no deploy
```

| Trigger | unit-tests | bundle-validate | deploy-prod |
|---|---|---|---|
| push to `main` | yes | yes | yes |
| pull request to `main` | yes | no | no |
| push to `dev` or feature branches | no | no | no |

Authentication uses **OAuth Machine-to-Machine** with a service principal — no Personal Access Tokens anywhere. The three required secrets (`DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`) live in GitHub Actions secrets.

---

## 9. Branch Strategy

```
main          protected, requires PR + CI pass   → Databricks prod (automated)
  ▲
  │ merge
dev           integration testing                → Databricks dev (manual)
  ▲
  │ merge
feature/*     individual feature work            → none (CI runs tests on PRs)
```

---

## 10. Local Development

### Prerequisites

- Python 3.11 or later
- Databricks CLI v0.218 or later
- A Databricks workspace with Unity Catalog enabled

### Setup

```bash
# 1. Clone
git clone https://github.com/gulaag/FinSage.git
cd FinSage

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Dependencies
pip install -r requirements.txt

# 4. Authenticate (browser-based OAuth; no PAT required)
databricks auth login --host https://dbc-f33010ed-00fc.cloud.databricks.com/

# 5. Deploy to your personal dev environment
databricks bundle deploy

# 6. Trigger a run
databricks bundle run finsage_daily_run
```

---

## 11. Directory Structure

```
FinSage/
├── databricks.yml                     # Databricks Asset Bundle root configuration
├── databricks/
│   └── notebooks/
│       ├── 01_schema_setup.py         # Schemas, volumes, parallel SEC download
│       ├── 02_bronze_autoloader.py    # Auto Loader + CompanyFacts API ingestion
│       ├── 03_silver_decoder.py       # XBRL flattening + sec-parser section extraction
│       ├── 04_gold_metrics.py         # Annual metric aggregation + KPI derivation
│       ├── 04b_gold_quarterly_metrics.py  # Discrete quarterly metric derivation
│       ├── 05_vector_chunker.py       # tiktoken chunking + Vector Search index
│       ├── 06_rag_agent.py            # Function-calling agent + serving deployment
│       └── 07_evaluation.py           # 8-scorer evaluation harness
├── src/
│   ├── finsage/                       # Shared constants (TARGET_CONCEPT_MAP, etc.)
│   ├── ingestion/                     # SEC EDGAR downloader utilities
│   └── evaluation/                    # Gold dataset, scorers, persistence, analysis
├── app/                               # Chat frontend (FastAPI SPA + Streamlit) over the agent endpoint
├── terraform/
│   └── main.tf                        # Cluster policy, secret scope, service principal lookup
├── .github/
│   └── workflows/
│       └── deploy.yml                 # CI/CD pipeline
├── tests/
│   └── unit/                          # pytest suites (normalizer, scorers, preflight)
├── docs/                              # Architecture and decision records
├── requirements.txt
└── README.md
```

Notebook source files under `databricks/notebooks/` begin with `# Databricks notebook source`, so Databricks recognizes them as notebooks on deploy while they remain plain Python in Git.

---

## 12. Environment and Secrets

| Name | Location | Purpose |
|---|---|---|
| `DATABRICKS_HOST` | GitHub Secret | Workspace URL for CLI auth in CI |
| `DATABRICKS_CLIENT_ID` | GitHub Secret | Service principal client ID for M2M OAuth |
| `DATABRICKS_CLIENT_SECRET` | GitHub Secret | Service principal OAuth secret |
| `USER_AGENT` | Notebook widget default | Identifies FinSage to the SEC EDGAR API, as required by SEC terms |

Never commit secrets. Use GitHub Actions secrets for CI and Databricks Secret Scopes for runtime secrets accessible inside notebooks.

---

## 13. Future Work

### Pilot and validation

FinSage ran a pilot with **1,000 retail investors in Japan.** The feedback was consistent: **every number is cited**, so users trusted the answers because they could verify each figure against its source filing on the spot. That result confirmed the founding thesis — in this market, trust is the feature that sells — and it shapes the roadmap.

### Roadmap

- **Learned re-ranking.** Add a cross-encoder / LLM re-ranking stage on top of ANN retrieval to sharpen passage relevance beyond the current similarity-threshold gate.
- **Customer portfolio integration.** Let each user connect the portfolio they actually hold, so FinSage can personalize answers, comparisons, and alerts.
- **Live market feeds.** Extend the corpus beyond static filings to shareholder / earnings presentations and validated social signal (Twitter / X), while keeping the strict rule that any surfaced number stays traceable to a named source.
- **Knowledge graph layer.** Entity-relationship extraction from MD&A / Risk Factors for multi-hop reasoning over companies, risks, suppliers, and executives.

Each new source passes through the same trust discipline that defines the platform today: ingest it, ground answers in it, and cite it — or do not serve it at all.

---

## 14. License

Internal project, Arsaga Partners. All rights reserved.
