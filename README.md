# FinSage

> **Trust is what you sell in Japan.**

In Japanese retail finance, a number without a source is worthless. FinSage is built around that single principle: every figure it returns is traceable to the exact SEC filing, fiscal year, and section it came from. It is a production-grade financial intelligence platform that turns **80,000+ pages** of raw SEC filings into a function-calling AI agent that answers like an analyst and cites like an auditor.

FinSage ingests annual (10-K) and quarterly (10-Q) filings for 30 large-cap U.S. companies across 6 fiscal years, normalizes their XBRL financials, extracts their narrative sections, and serves the result through a governed, fully traced retrieval agent on Databricks.

| Metric | Value |
|---|---|
| Filings processed | 728 (10-K + 10-Q) |
| Pages of source text | 80,000+ |
| Companies | 30 large-cap, 6 sectors |
| Fiscal years | 2020 to 2026 |
| Vector embeddings | 45,136 BGE-large chunks |
| Agent tools | 4 (function calling) |
| Answer accuracy | 87% on a 100-question gold set |
| Citation coverage | 100% of grounded answers |
| Hallucination rate | Near zero |

---

## Table of Contents

1. [Agent Orchestration](#1-agent-orchestration)
2. [Future Work](#2-future-work)
3. [Data Engineering: The Medallion Pipeline](#3-data-engineering-the-medallion-pipeline)
4. [Databricks Asset Bundle](#4-databricks-asset-bundle)
5. [CI/CD with GitHub Actions](#5-cicd-with-github-actions)
6. [Branch Strategy](#6-branch-strategy)
7. [Local Development](#7-local-development)
8. [Testing and Evaluation](#8-testing-and-evaluation)
9. [Directory Structure](#9-directory-structure)
10. [Deployment Reference](#10-deployment-reference)
11. [Environment and Secrets](#11-environment-and-secrets)
12. [License](#12-license)

---

## 1. Agent Orchestration

This is the heart of FinSage. The data pipeline exists to feed it. The agent is a governed reasoning system, not a thin wrapper around a language model, and it is engineered so that a wrong or unsourced answer is structurally hard to produce.

### 1.1 Design at a glance

| Property | Choice | Why it matters |
|---|---|---|
| Framework | `mlflow.pyfunc.PythonModel` | Zero LangChain dependency. The orchestration loop is owned, inspectable, and portable across serving runtimes. |
| Reasoning model | Claude Sonnet 4.6 (function calling), served on Databricks Model Serving | Strong tool selection and instruction following, hosted inside the same governed workspace as the data. |
| Control pattern | ReAct loop (Reason, Act, Observe, repeat) | Lets a single question drive multiple tool calls until the agent has enough grounded evidence to answer. |
| Registration | Unity Catalog model `main.finsage_gold.finsage_rag_agent` | Versioned, governed, and promotable like any other UC asset. |
| Serving | Model Serving endpoint `finsage_agent_endpoint` | Scales to zero when idle, redeployed automatically by a Databricks Job. |
| Observability | MLflow tracing on every call | Each tool call and model call emits a span, so any answer can be replayed and audited. |

### 1.2 The ReAct control loop

Every question flows through one deterministic loop. There are no regex shortcuts and no hidden fast paths, so behavior is uniform and testable.

```
User question
      │
      ▼
┌───────────────────────────────────────────────────────────────┐
│  System prompt (routing, refusal policy, citation contract)    │
│  is prepended to the conversation                              │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
        ┌──────────── ReAct loop, up to 5 iterations ───────────┐
        │                                                        │
        │   1. REASON   model reads history + tool schemas       │
        │   2. ACT      model emits a structured tool call       │
        │   3. OBSERVE  harness runs the tool, captures result   │
        │   4. FEED     result is appended to the conversation   │
        │                                                        │
        │   Repeat until the model stops calling tools,          │
        │   or a guardrail fires.                                │
        └───────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │  Citation enforcement pass                            │
        │  guarantees every grounded answer carries a source    │
        └───────────────────────────┬───────────────────────────┘
                                    ▼
                          Answer shown to user
```

The harness, not the model, executes tools. That separation is the control surface: the harness validates arguments, applies thresholds, retries, logs, and enforces the output contract.

### 1.3 The four tools

The agent is given exactly four tools. Fewer, well-scoped tools produce more reliable tool selection than many overlapping ones.

| Tool | Purpose | Source |
|---|---|---|
| `search_filings` | Semantic search over 10-K and 10-Q narrative text for qualitative questions (strategy, risks, products, competition, regulation, supply chain). Filterable by ticker, section, fiscal year, and filing type. | Vector Search index `filing_chunks_index` |
| `get_company_metrics` | Annual structured financials: revenue, net income, operating income, gross profit, cash flow, assets, liabilities, equity, debt, R&D, margins, YoY growth, debt-to-equity. | Gold table `company_metrics`, loaded to an in-memory cache at serving time |
| `get_quarterly_metrics` | Discrete quarterly metrics for Q1, Q2, and Q3, plus same-quarter YoY trends. | Gold table `company_metrics_quarterly` |
| `get_filing_metadata` | Deterministic cover-page facts: filing date, employee count, shares outstanding. | Silver financial data |

A key architectural decision: **numbers never come from RAG.** Financial figures are served from structured Gold tables through the metrics tools, so they are exact and cannot be hallucinated. Retrieval is reserved for language, where free-text search is the right instrument. This split is the single biggest reason the hallucination rate is near zero.

### 1.4 Guardrails

Production safety is enforced by the harness, on the assumption that the model will occasionally misbehave.

| Guardrail | Setting | Protects against |
|---|---|---|
| Bounded tool loop | `MAX_ITERATIONS = 5` | Infinite reasoning loops and runaway cost |
| Wall-clock timeout | 150 seconds per request | Latency blowouts on pathological queries |
| Retrieval quality gate | Similarity threshold `0.4`, with a single permissive retry if strict filtering removes everything | Weak, off-topic passages being cited as evidence |
| Citation enforcement | Post-processing pass that attaches a real, retrieved `[Source: ...]` line if the model forgot | Ungrounded or unsourced answers reaching the user |
| Refusal policy | System prompt declines out-of-corpus tickers, future fiscal years, and missing data | Fabricated numbers for data the platform does not hold |
| Deterministic decoding | `temperature = 0.0` | Non-reproducible answers |

### 1.5 The citation contract

Because trust is the product, the output format is a contract, not a preference:

- Metric answers end with `[Source: TICKER | FY#### | metrics]` (or `FY#### Q#` for quarterly).
- Filing-text answers are marked `[VERBATIM]` or `[SUMMARY]` and carry `[Source: TICKER | FY#### | 10-K/10-Q | Section]`.
- Cover-page facts carry `[Source: TICKER | FY#### | 10-K Cover Page]`.
- When data is absent, the agent refuses clearly and states why, rather than guessing.

### 1.6 Observability and deployment

Every `predict` call, every retrieval, and every metrics lookup emits an MLflow span, so any production answer can be traced end to end and replayed. The agent is registered in Unity Catalog and deployed to a Model Serving endpoint that scales to zero. Redeployment is automated through a Databricks Job that runs the agent notebook end to end on serverless compute, so shipping a new version is a single governed action rather than a manual sequence.

---

## 2. Future Work

### 2.1 Pilot and validation

FinSage ran a pilot with **1,000 retail investors in Japan.** The feedback was strongly positive, and the reason was consistent: **every number is cited.** Users trusted the answers because they could verify each figure against its source filing on the spot. That result confirmed the founding thesis, that in this market trust is the feature that sells, and it directly shapes the roadmap below.

### 2.2 Roadmap

- **Customer portfolio integration.** Let each user connect the portfolio they actually hold, so FinSage can personalize answers, comparisons, and alerts to the specific companies a customer owns rather than treating every query as anonymous.
- **Live market feeds.** Extend the corpus beyond static SEC filings to include real-time and semi-structured sources:
  - Shareholder presentations
  - Investor and earnings presentations
  - Social signal from Twitter / X
  - Additional public sources as they are validated
  The goal is to complement the audited, citation-grade filing data with timely signal, while keeping the same strict rule that any number surfaced to a user remains traceable to a named source.

Each new source will pass through the same trust discipline that defines the platform today: ingest it, ground answers in it, and cite it, or do not serve it at all.

---

## 3. Data Engineering: The Medallion Pipeline

Everything the agent knows is produced by a five-stage Medallion pipeline on Databricks, deployed as a Databricks Asset Bundle and orchestrated as a sequential job. All layers are Delta Lake tables in Unity Catalog under the `main` catalog.

```
                   ┌──────────────────────────────────────────────────┐
                   │               GitHub Repository                   │
                   │        feature/* ──► dev ──► main                 │
                   └──────────────────────┬───────────────────────────┘
                                          │  push to main
                                          ▼
                   ┌──────────────────────────────────────────────────┐
                   │            GitHub Actions Workflow                │
                   │  1. pytest (unit + preflight)                     │
                   │  2. databricks bundle validate                    │
                   │  3. databricks bundle deploy -t prod              │
                   └──────────────────────┬───────────────────────────┘
                                          │  deploy
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Databricks Workspace (Production)                        │
│                                                                                │
│  SEC EDGAR / CompanyFacts API                                                  │
│        │                                                                       │
│        ▼                                                                       │
│  [01 Schema Setup] ──► [02 Bronze] ──► [03 Silver] ──► [04 Gold]              │
│                          Auto Loader     sec-parser +     annual metrics       │
│                          + XBRL API      XBRL flatten     + quarterly metrics  │
│                                                              │                 │
│                                                              ▼                 │
│                                                       [05 Vector Chunker]      │
│                                                       tiktoken + VS index      │
│                                                              │                 │
│                                                              ▼                 │
│                                                       [06 RAG Agent]           │
│                                                              │                 │
│                                                              ▼                 │
│                                                       [07 Evaluation]          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Bronze: raw, append-only ingestion (`main.finsage_bronze`)

The Bronze layer captures source data exactly as received, with no business logic, so it is fully auditable and replayable.

| Table | Description |
|---|---|
| `filings` | Raw filing bytes stored as `BINARY`, ingested by Databricks Auto Loader (`cloudFiles`) with checkpoint-based exactly-once delivery. |
| `xbrl_companyfacts_raw` | Raw JSON payloads from the SEC EDGAR CompanyFacts API, one snapshot per ticker per day. |
| `ingestion_errors` | Central error log for download, parse, and flattening failures across all layers. |
| `sec_filings_download_log` | Idempotency ledger that prevents re-downloading the same filing on a re-run. |

Change Data Feed is enabled on Bronze tables so downstream layers can consume changes incrementally. Auto Loader runs with `availableNow=True`, which gives batch-style behavior inside a scheduled job: process all newly available files, then stop.

### 3.2 Silver: cleaned and parsed (`main.finsage_silver`)

Two independent transformations turn raw bytes into structured, queryable rows.

| Table | Transformation |
|---|---|
| `financial_statements` | Flattens the deeply nested XBRL CompanyFacts JSON, mapping many raw US-GAAP concept names to a small set of canonical metrics (`revenue`, `net_income`, `equity`, and so on) via `TARGET_CONCEPT_MAP`. Deduplicated with a SHA-256 `statement_id` and written with an idempotent MERGE. |
| `filing_sections` | Extracts named narrative sections from 10-K and 10-Q filings. The primary extractor is `sec-parser`, a DOM-aware iXBRL parser; a per-section regex extractor fills any required section it misses. 10-K yields Business, Risk Factors, and MD&A; 10-Q yields MD&A and, when present, Risk Factors Updates. |

The section extractor is the hardest part of the pipeline. Operating on the parsed document tree rather than flattened text lets it decode HTML entities natively, survive inline-XBRL span fragmentation, and drop page headers and footers by semantic role. Each section records which extractor produced it, so provenance is preserved at row level.

### 3.3 Gold: analytics-ready metrics (`main.finsage_gold`)

The Gold layer produces wide, immediately usable tables with derived KPIs, plus the chunk table that feeds retrieval.

| Table | Description |
|---|---|
| `company_metrics` | One row per (ticker, fiscal year) for annual 10-K data. Contains the core financials plus gross margin, revenue YoY growth, debt-to-equity, and a `data_quality_score`. |
| `company_metrics_quarterly` | Discrete Q1, Q2, and Q3 metrics derived from 10-Q filings, including the subtraction logic needed to recover discrete quarters from cumulative year-to-date XBRL facts. |
| `filing_section_chunks` | Token-based chunks of Silver sections, each with a deterministic SHA-256 `chunk_id` for idempotent merges. |
| `filing_chunks_index` | Databricks Vector Search index (Delta Sync, triggered pipeline) backed by `databricks-bge-large-en`, holding 45,136 chunks filterable by ticker, section, and fiscal year. |

Gold applies strict accounting discipline: flow metrics require a full-year period window, balance-sheet metrics are treated as point-in-time, and a canonical filing is chosen per (ticker, fiscal year) by metric coverage so that amended filings never mix into the same row.

### 3.4 Vector: the retrieval foundation

The chunker splits each Silver section independently, never across section boundaries, into 512-token chunks with 64 tokens of overlap using the `tiktoken` tokenizer. Section-aware chunking preserves the strong semantic structure of financial filings, and deterministic chunk IDs make re-indexing idempotent. The resulting chunks are embedded with BGE-large and served through the Vector Search index that powers `search_filings`.

---

## 4. Databricks Asset Bundle

FinSage is deployed as a Databricks Asset Bundle (DAB). The job topology, cluster configuration, schedule, and environment promotions are version-controlled in `databricks.yml`.

The bundle defines a sequential job whose tasks run in strict order via `depends_on`:

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

If any task fails, downstream tasks are skipped and a failure alert is sent. The job runs on a schedule in production and is paused in the `dev` target to avoid accidental runs during development.

### Bundle variables

| Variable | Default | Description |
|---|---|---|
| `catalog` | `main` | Unity Catalog catalog name |
| `env` | `dev` | Environment label (`dev` / `prod`) |
| `start_date` | `2020-01-01` | Earliest SEC filing date to ingest |
| `ticker_filter` | `""` | Comma-separated tickers to process (empty = all 30) |
| `notification_email` | configured per deploy | Recipient for job failure alerts |

### Targets

| Target | Mode | Purpose |
|---|---|---|
| `dev` (default) | development | Personal deploys, job name prefixed with the deploying user, schedule paused |
| `prod` | production | Shared deploy, triggered only by CI/CD on push to `main` |

### Common commands

```bash
databricks bundle validate            # syntax + workspace connection check
databricks bundle deploy              # deploy to dev (default target)
databricks bundle deploy -t prod      # deploy to prod
databricks bundle run finsage_daily_run
```

---

## 5. CI/CD with GitHub Actions

The workflow lives in `.github/workflows/deploy.yml` and gates production behind tests and validation.

```
push to main
     │
     ▼
┌────────────────┐   failure   ┌───────────────────────────────────┐
│  unit-tests    │────────────►│  Pipeline halted, no deploy runs   │
│  (pytest)      │             └───────────────────────────────────┘
└───────┬────────┘
        │ success
        ▼
┌────────────────────────┐
│  bundle-validate       │  databricks bundle validate
└────────────┬───────────┘
             │ success
             ▼
┌────────────────────────┐
│  deploy-prod           │  databricks bundle deploy -t prod
└────────────────────────┘
```

| Trigger | unit-tests | bundle-validate | deploy-prod |
|---|---|---|---|
| push to `main` | yes | yes | yes |
| pull request to `main` | yes | no | no |
| push to `dev` or feature branches | no | no | no |

Authentication uses OAuth Machine-to-Machine with a service principal. Personal Access Tokens are not used anywhere in the pipeline. The three required secrets (`DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`) are stored in GitHub Actions secrets.

---

## 6. Branch Strategy

```
main          protected, requires PR + CI pass
                    ▲                 ▲
                    │ merge           │ merge
dev           integration testing ───┘
                    ▲
                    │ merge
feature/*     individual feature work
```

| Branch | Purpose | Deploys to |
|---|---|---|
| `feature/*` | Short-lived feature and fix branches | None (CI runs tests on PRs) |
| `dev` | Integration and staging | Databricks `dev` target (manual deploy) |
| `main` | Production-ready, merged via PR only | Databricks `prod` target (automated) |

---

## 7. Local Development

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

## 8. Testing and Evaluation

FinSage separates fast, local correctness checks from full agent evaluation.

### Unit and preflight tests

Located in `tests/unit/`, these run as plain Python with no Spark dependency:

```bash
pytest tests/unit/ -v
```

They cover XBRL concept normalization, the evaluation scorers, and a preflight suite that catches dataset, schema, and scorer-wiring regressions before a cluster is ever started.

### Agent evaluation harness

The evaluation notebook scores the deployed agent against a locked, stratified gold set of 100 questions spanning all 30 tickers, verified against SEC EDGAR. Seven scorers grade each answer:

| Scorer | What it checks |
|---|---|
| Correctness | LLM-judged factual correctness against ground truth |
| Cites ticker and year | Presence of the required identifying citation |
| Numerical tolerance | Values within tolerance, with unit-aware extraction |
| Citation format | Correct `[VERBATIM]` / `[SUMMARY]` and `[Source: ...]` shape |
| Refusal correctness | Refusals happen for the right reason |
| Tool routing correctness | The agent chose the correct tool for the question |
| Derived metric match | Derived ratios match independent of wording |

Results are persisted to two idempotent Delta tables (`eval_run_summaries` and `eval_question_outcomes`), so every run is versioned and comparable across agent versions. Current headline results: **87% accuracy, 100% citation coverage on grounded answers, and a near-zero hallucination rate.**

---

## 9. Directory Structure

```
FinSage/
├── databricks.yml                     # Databricks Asset Bundle root configuration
├── databricks/
│   └── notebooks/
│       ├── 01_schema_setup.py         # Schemas, volumes, and parallel SEC download
│       ├── 02_bronze_autoloader.py    # Auto Loader + CompanyFacts API ingestion
│       ├── 03_silver_decoder.py       # XBRL flattening + sec-parser section extraction
│       ├── 04_gold_metrics.py         # Annual metric aggregation + KPI derivation
│       ├── 04b_gold_quarterly_metrics.py  # Discrete quarterly metric derivation
│       ├── 05_vector_chunker.py       # tiktoken chunking + Vector Search index
│       ├── 06_rag_agent.py            # Function-calling agent + serving deployment
│       └── 07_evaluation.py           # 7-scorer evaluation harness
├── src/
│   ├── finsage/                       # Shared constants (TARGET_CONCEPT_MAP, etc.)
│   ├── ingestion/                     # SEC EDGAR downloader utilities
│   └── evaluation/                    # Ground-truth dataset, scorers, persistence, analysis
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

## 10. Deployment Reference

```bash
# Validate then deploy
databricks bundle validate
databricks bundle deploy            # dev
databricks bundle deploy -t prod    # prod (normally via CI)

# Update an existing deployment
git add .
git commit -m "feat: update silver section extraction"
git push origin main                # GitHub Actions handles the rest

# Monitor a run
databricks jobs list-runs --job-id <job-id>
databricks runs get-output --run-id <run-id>
```

---

## 11. Environment and Secrets

| Name | Location | Purpose |
|---|---|---|
| `DATABRICKS_HOST` | GitHub Secret | Workspace URL for CLI auth in CI |
| `DATABRICKS_CLIENT_ID` | GitHub Secret | Service principal client ID for M2M OAuth |
| `DATABRICKS_CLIENT_SECRET` | GitHub Secret | Service principal OAuth secret |
| `USER_AGENT` | Notebook widget default | Identifies FinSage to the SEC EDGAR API, as required by SEC terms |

Never commit secrets. Use GitHub Actions secrets for CI and Databricks Secret Scopes for runtime secrets accessible inside notebooks.

---

## 12. License

Internal project, Arsaga Partners. All rights reserved.
