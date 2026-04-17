# FinSage — SEC Filing Intelligence Platform

> **A production-grade Medallion data pipeline on Databricks**, deployed as a Databricks Asset Bundle (DAB) with full CI/CD via GitHub Actions.

FinSage ingests annual (10-K) and quarterly (10-Q) SEC filings for 30 large-cap U.S. companies, normalizes XBRL financial metrics, extracts narrative sections, and publishes a token-based vector index for Retrieval-Augmented Generation (RAG).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture / CI-CD Flow](#2-architecture--ci-cd-flow)
3. [Medallion Layers](#3-medallion-layers)
4. [Directory Structure](#4-directory-structure)
5. [Databricks Asset Bundle](#5-databricks-asset-bundle)
6. [CI/CD — GitHub Actions](#6-cicd--github-actions)
7. [Branch Strategy](#7-branch-strategy)
8. [Local Development](#8-local-development)
9. [Running Tests](#9-running-tests)
10. [Deployment Reference](#10-deployment-reference)
11. [Environment Variables and Secrets](#11-environment-variables-and-secrets)
12. [Notebook Code Reference](#12-notebook-code-reference)
13. [Future Work](#13-future-work)

---

## 1. Project Overview

FinSage is a production-grade data engineering platform that processes SEC EDGAR filings for a curated set of 30 large-cap U.S. companies. The pipeline is structured as a five-stage Medallion architecture (Bronze → Silver → Gold → Vector) and is deployed to Databricks as a Databricks Asset Bundle (DAB). The full job topology, cluster configuration, and environment promotions are version-controlled and deployed automatically through GitHub Actions CI/CD.

**What FinSage does:**

- Downloads 10-K and 10-Q SEC filings from SEC EDGAR using the `sec-edgar-downloader` library and the CompanyFacts API.
- Ingests raw filing files into a Delta Lake Bronze layer using Databricks Auto Loader (`cloudFiles` format).
- Normalizes XBRL financial data (27 raw XBRL concepts → 11 canonical metric names) in the Silver layer.
- Extracts three named narrative sections (Business Overview, Risk Factors, MD&A) from 10-K HTML filings using regex-based boundary detection.
- Aggregates financial KPIs, computes year-over-year growth rates, and assigns data quality scores in the Gold layer.
- Chunks Gold narrative text with `tiktoken` (512 tokens, 64-token overlap) and provisions a Databricks Vector Search Delta Sync index for downstream RAG use.

---

## 2. Architecture / CI-CD Flow

```
                   ┌──────────────────────────────────────────────────┐
                   │               GitHub Repository                   │
                   │   feature/* ──► dev ──► main                     │
                   └──────────────────────┬───────────────────────────┘
                                          │  push to main
                                          ▼
                   ┌──────────────────────────────────────────────────┐
                   │         GitHub Actions Workflow                   │
                   │  1. pytest (unit tests)                          │
                   │  2. databricks bundle validate                   │
                   │  3. databricks bundle deploy -t prod             │
                   └──────────────────────┬───────────────────────────┘
                                          │  deploy
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Databricks Workspace (Production)                     │
│                                                                             │
│  SEC EDGAR / SEC API ──► [01 Schema Setup]                                  │
│                               │                                             │
│                               ▼                                             │
│                         [02 Bronze]  ◄── Auto Loader (cloudFiles)           │
│                               │          + CompanyFacts API                 │
│                               ▼                                             │
│                         [03 Silver] ◄── XBRL Flattening + Section NLP      │
│                               │                                             │
│                               ▼                                             │
│                         [04 Gold]   ◄── Metric Aggregation + YoY Growth    │
│                               │                                             │
│                               ▼                                             │
│                         [05 Vector] ◄── tiktoken Chunking + VS Index        │
│                                                                             │
│   All layers are stored as Delta Lake tables in Unity Catalog (main)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Medallion Layers

### Bronze Layer — Raw Data Ingestion (`main.finsage_bronze`)

The Bronze layer is **append-only and auditable**. No business logic is applied.

| Table | Description |
|---|---|
| `filings` | Raw binary content of all SEC filing files. Ingested by Databricks Auto Loader (`cloudFiles` format). Checkpoint-based exactly-once delivery is guaranteed. |
| `xbrl_companyfacts_raw` | Raw JSON payloads from the SEC EDGAR CompanyFacts API (`/api/xbrl/companyfacts/CIK{cik}.json`). One row per ticker per day. |
| `ingestion_errors` | Logs download failures, HTTP errors, and parse exceptions from all layers. Used for observability and reprocessing. |
| `sec_filings_download_log` | Idempotency log for the `sec-edgar-downloader` parallel download job. Prevents duplicate API calls on re-runs. |

**Key design decisions:**

- `delta.enableChangeDataFeed = true` is enabled on all Bronze tables to support downstream Change Data Capture.
- The Auto Loader `availableNow=True` trigger causes the stream to behave as a batch job: it processes all newly available files and then stops.

### Silver Layer — Cleaned and Parsed (`main.finsage_silver`)

The Silver layer applies two independent transformations to Bronze data:

| Table | Transformation |
|---|---|
| `financial_statements` | Flattens XBRL CompanyFacts JSON using `TARGET_CONCEPT_MAP`, which maps 27 raw XBRL concept names to 11 canonical metric names (`revenue`, `net_income`, etc.). Deduplication uses a SHA-256 `statement_id`. Each run performs an idempotent MERGE. |
| `filing_sections` | Decodes 10-K HTML file bytes, removes HTML tags, Base64-encoded images, and scripts, then parses three named sections (**Business (Item 1)**, **Risk Factors (Item 1A)**, **MD&A (Item 7)**) using regex-based boundary detection. Section boundaries are deterministic and auditable. |

**Key design decisions:**

- `TARGET_CONCEPT_MAP` is the single source of truth for metric normalization. It is defined in `src/finsage/constants.py` and is independently unit-tested (see `tests/unit/test_normalizer.py`).
- Section extraction processes `10-K` filings only — quarterly reports do not have an Item 1/7 structure.

### Gold Layer — Analytical Metrics (`main.finsage_gold`)

The Gold layer produces **wide, analytics-ready** tables with derived KPIs per ticker and fiscal year.

| Table | Description |
|---|---|
| `company_metrics` | One row per `(ticker, fiscal_year)`. Contains 16 financial metrics, gross margin %, revenue YoY growth %, debt-to-equity ratio, and a `data_quality_score` (0–1) indicating how many of 9 core metrics are populated. |
| `filing_section_chunks` | Token-based chunks of Silver sections (512 tokens, 64-token overlap). Each chunk has a deterministic SHA-256 `chunk_id` for idempotent merges. |

**Key design decisions:**

- Strict accounting period alignment: flow metrics require `fiscal_period = 'FY'` and `duration_days` between 350 and 380.
- Canonical accession number selection: before aggregation, one `accession_number` per `(ticker, fiscal_year)` is selected based on required metric coverage.

### Vector Layer — RAG Index (`main.finsage_gold`)

| Resource | Description |
|---|---|
| `filing_section_chunks` (Gold) | Source table with `delta.enableChangeDataFeed = true`. |
| `filing_chunks_index` | Databricks Vector Search Delta Sync index backed by `databricks-bge-large-en`. Supports similarity search for downstream RAG agents. |

---

## 4. Directory Structure

```
FinSage/
├── databricks.yml                     # Databricks Asset Bundle root configuration
├── databricks/
│   ├── notebooks/
│   │   ├── 01_schema_setup.py         # DDL: schemas, tables, volumes + SEC filing download
│   │   ├── 02_bronze_autoloader.py    # Auto Loader + SEC API ingestion
│   │   ├── 03_silver_decoder.py       # XBRL flattening + section extraction
│   │   ├── 04_gold_metrics.py         # Metric aggregation + KPI derivation
│   │   └── 05_vector_chunker.py       # Chunking + Vector Search setup
│   └── workflows/                     # Reserved for future workflow YAML definitions
├── terraform/
│   └── main.tf                        # Cluster policy, secret scope, and SP lookup
├── .github/
│   └── workflows/
│       └── deploy.yml                 # CI/CD pipeline (pytest → validate → deploy)
├── tests/
│   └── unit/
│       └── test_normalizer.py         # pytest: TARGET_CONCEPT_MAP coverage
├── src/
│   ├── finsage/
│   │   ├── __init__.py
│   │   └── constants.py               # TARGET_CONCEPT_MAP and VALID_NORMALIZED_METRICS
│   ├── ingestion/
│   │   └── edgar_downloader.py        # SEC EDGAR downloader utilities
│   ├── evaluation/
│   │   └── ground_test.json           # Evaluation ground truth data
│   ├── agent/                         # Stub — see Future Work
│   ├── app/                           # Stub — see Future Work
│   ├── monitoring/                    # Stub — see Future Work
│   ├── processing/                    # Stub — see Future Work
│   ├── retrieval/                     # Stub — see Future Work
│   └── serving/                       # Stub — see Future Work
├── docs/
│   ├── finsage_presentation.html
│   ├── challenges_log.html
│   └── technical_decisions.html
├── assets/
│   └── screenshots/
│       └── pipeline.gif               # Live DAG view in the Databricks workspace
├── requirements.txt
└── README.md
```

> **Databricks notebook source files:** All `.py` files under `databricks/notebooks/` begin with `# Databricks notebook source`. When uploaded to the Databricks workspace by the DAB deploy step, Databricks recognizes them as interactive notebooks; in Git they remain plain Python files.

---

## 5. Databricks Asset Bundle

FinSage is deployed as a **Databricks Asset Bundle (DAB)**. The entire job topology — including the cluster definition, task DAG, schedule, and environment promotions — is defined in `databricks.yml` at the repository root.

### Bundle Configuration

The bundle defines one job (`finsage_daily_run`) with five tasks connected in a strict sequential DAG:

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

Tasks execute in **strict sequential order** via `depends_on`. If any task fails, all downstream tasks are skipped and an email alert is sent to the configured `notification_email`.

The job is scheduled daily at **06:00 UTC** in production. In the `dev` target, the schedule is set to `PAUSED` to prevent accidental cron runs during development.

### Bundle Variables

Variables are defined at the bundle level and injected into notebook tasks via `base_parameters`. Notebooks read them using `dbutils.widgets.get("variable_name")`. This pattern eliminates hard-coded environment-specific values in notebook code.

| Variable | Default | Description |
|---|---|---|
| `catalog` | `main` | Unity Catalog catalog name |
| `env` | `dev` | Environment label (`dev` / `prod`) |
| `start_date` | `2020-01-01` | Earliest SEC filing date to ingest |
| `ticker_filter` | `""` | Comma-separated tickers to process (empty = all 30) |
| `cluster_spark_version` | `14.3.x-scala2.12` | Databricks Runtime version |
| `cluster_node_type` | `i3.xlarge` | EC2 instance type (overridden to `i3.2xlarge` in prod) |
| `notification_email` | `digvijay@arsaga.jp` | Email address for job failure alerts |

### Targets

| Target | Mode | Purpose |
|---|---|---|
| `dev` (default) | `development` | Personal deploys; job name is prefixed with `[dev <username>]`; safe for iterative development. Schedule is paused. |
| `prod` | `production` | Shared deploy; no name prefix; triggered exclusively by CI/CD on push to `main`. |

**Note on the dev target:** The current `dev` target routes all tasks to an existing interactive cluster (`existing_cluster_id`) rather than spinning up a new job cluster. This is a workaround for environments where the deploying user account lacks the "Allow cluster creation" permission required for automated job compute. Once an administrator grants the required permission, replace the `existing_cluster_id` override with the top-level `job_clusters` definition.

### CLI Commands

```bash
# Validate the bundle (syntax + workspace connection check)
databricks bundle validate

# Deploy to dev (default target)
databricks bundle deploy

# Deploy to a specific target
databricks bundle deploy -t prod

# Manually trigger the job in dev
databricks bundle run finsage_daily_run

# Manually trigger the job in prod
databricks bundle run -t prod finsage_daily_run

# Destroy deployed resources (dev only — do not run in prod without approval)
databricks bundle destroy
```

### Live DAG — Databricks Workspace

The screenshot below shows the `finsage_daily_run` job as it appears in the Databricks Jobs UI after a successful `databricks bundle deploy`. All five tasks are connected as a strictly sequential DAG and share the `finsage_cluster` job cluster to avoid per-task cold-start overhead.

<img src="assets/screenshots/pipeline.gif" alt="FinSage pipeline DAG in the Databricks workspace — five sequential tasks: schema_setup → bronze_autoloader → silver_decoder → gold_metrics → vector_chunker, all running on the shared finsage_cluster. Scheduled daily at 06:00 UTC." style="max-width:100%;" />

> In development mode, the job is marked **[dev Digvijay]** — DAB automatically prefixes the deploying user's name to the job name to prevent collisions with the production `finsage_daily_run` job in the same shared workspace.

---

## 6. CI/CD — GitHub Actions

The workflow file is `.github/workflows/deploy.yml`.

### Pipeline Stages

```
push to main
     │
     ▼
┌────────────────┐     failure     ┌─────────────────────────────────────────┐
│  unit-tests    │────────────────►│  Pipeline halted. No deploy occurs.     │
│  (pytest)      │                  └─────────────────────────────────────────┘
└───────┬────────┘
        │ success
        ▼
┌────────────────────────┐
│  bundle-validate       │  databricks bundle validate
│  (Databricks CLI)      │
└────────────┬───────────┘
             │ success
             ▼
┌────────────────────────┐
│  deploy-prod           │  databricks bundle deploy -t prod
│  (Databricks CLI)      │
└────────────────────────┘
```

### Workflow Trigger Matrix

| Trigger | unit-tests | bundle-validate | deploy-prod |
|---|---|---|---|
| push to `main` | ✓ | ✓ | ✓ |
| pull request targeting `main` | ✓ | ✗ | ✗ |
| push to `dev` or feature branches | ✗ | ✗ | ✗ |

### Authentication

The Databricks CLI authenticates via **OAuth Machine-to-Machine (M2M)** using a service principal. Classic Personal Access Tokens (PATs) are not used anywhere in this pipeline.

The CLI reads the following environment variables automatically, calls the workspace OIDC token endpoint (`/oidc/v1/token`), receives a short-lived bearer token (~1 hour TTL), and uses it for all API calls.

| Secret | Value |
|---|---|
| `DATABRICKS_HOST` | `https://<your-workspace>.cloud.databricks.com` |
| `DATABRICKS_CLIENT_ID` | Application (client) ID of the `finsage-service-principal` |
| `DATABRICKS_CLIENT_SECRET` | OAuth client secret of the `finsage-service-principal` |

Add these under **Settings → Secrets and variables → Actions** in the GitHub repository.

**Service principal provisioning (requires Databricks admin access):**

1. Create a service principal in your identity provider (Entra ID / Okta).
2. Grant the SP the **Can Manage** role on the Databricks workspace.
3. Generate an OAuth client secret for the SP.
4. Add the three secrets to GitHub Actions.

---

## 7. Branch Strategy

```
main          ──── protected; requires PR + CI pass ──────────────────────────►
                         ▲                    ▲
                         │ merge              │ merge
dev           ──── integration testing ───────┘
                         ▲
                         │ merge
feature/*     ──── individual feature work ──────────────────────────────────►
```

| Branch | Purpose | Deploys to |
|---|---|---|
| `feature/*` | New features, bug fixes. Short-lived. | None (CI runs unit tests only on PRs) |
| `dev` | Integration testing; staging equivalent. | Databricks `dev` target (manual `bundle deploy`) |
| `main` | Production-ready code. Merged via PR only. | Databricks `prod` target (automated via GitHub Actions) |

### Release Process

1. Create a `feature/my-change` branch from `main`.
2. Develop and test locally (see §8).
3. Open a pull request targeting `main`. GitHub Actions automatically runs unit tests.
4. Once the PR is approved and CI passes, merge to `main`.
5. The CI/CD pipeline automatically validates and deploys to `prod`.

> **Hotfixes:** Branch directly from `main`, apply the fix, and open a PR. Do not bypass the PR process — the `bundle validate` gate is the last line of defense before production.

---

## 8. Local Development

### Prerequisites

- Python 3.11 or later
- [Databricks CLI v0.218 or later](https://docs.databricks.com/dev-tools/cli/databricks-cli.html)
- A Databricks workspace with Unity Catalog enabled

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/FinSage.git
cd FinSage

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure the Databricks CLI with OAuth U2M (browser-based login)
databricks auth login --host https://dbc-f33010ed-00fc.cloud.databricks.com/
# A browser window opens for a one-time login. No PAT required.
# In CI environments, authenticate using DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET (M2M OAuth).

# 5. Deploy to your personal dev environment
databricks bundle deploy       # Deploys to the 'dev' target by default

# 6. Trigger a manual run
databricks bundle run finsage_daily_run
```

---

## 9. Running Tests

Tests are located in `tests/unit/` and run as **plain Python with no Spark dependency** — only the standard library, `pytest`, and the `src/finsage` package are required.

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run with a coverage report
pytest tests/unit/ -v --cov=databricks/notebooks --cov-report=term-missing

# Run a single test file
pytest tests/unit/test_normalizer.py -v
```

### Test Coverage

| Test File | What It Covers |
|---|---|
| `test_normalizer.py` | `TARGET_CONCEPT_MAP` and `VALID_NORMALIZED_METRICS` (defined in `src/finsage/constants.py`). Asserts that all known XBRL concepts resolve to the expected canonical metric name, that unknown concepts return `None`, that the map contains no `None` values, and that all 11 expected financial categories are present in `VALID_NORMALIZED_METRICS`. |

---

## 10. Deployment Reference

### First-Time Setup

```bash
# Validate the bundle (YAML + workspace permission check)
databricks bundle validate

# Deploy to dev
databricks bundle deploy

# Deploy to prod (normally handled by CI, but can be triggered manually)
databricks bundle deploy -t prod
```

### Updating an Existing Deployment

```bash
# After modifying notebooks or databricks.yml:
git add .
git commit -m "feat: update silver section extraction regex"
git push origin main
# → GitHub Actions handles the rest
```

### Monitoring a Running Job

```bash
# List recent job runs
databricks jobs list-runs --job-id <job-id>

# View logs for a specific run
databricks runs get-output --run-id <run-id>
```

---

## 11. Environment Variables and Secrets

| Name | Location | Purpose |
|---|---|---|
| `DATABRICKS_HOST` | GitHub Secret | Workspace URL for CLI authentication in CI |
| `DATABRICKS_CLIENT_ID` | GitHub Secret | Service principal client ID for M2M OAuth in CI |
| `DATABRICKS_CLIENT_SECRET` | GitHub Secret | Service principal OAuth secret for M2M authentication in CI |
| `USER_AGENT` | Notebook widget default | Identifies FinSage to the SEC EDGAR API (required by SEC Terms of Service) |

> **Never commit secrets to Git.** `.gitignore` already excludes `.env` files. Use GitHub Secrets for CI and Databricks Secret Scopes (`databricks secrets`) for runtime secrets accessible within notebooks.

---

## 12. Notebook Code Reference

This section provides a detailed walkthrough of the five Databricks notebooks that form the FinSage pipeline. Each notebook has a specific role in the Medallion architecture, transforming SEC filing data from raw bytes into a searchable vector index.

---

### Notebook 01 — `01_schema_setup.py`: Schema Initialization & SEC Filing Download

This notebook is the entry point for the entire pipeline. It creates the infrastructure and downloads SEC filings for all 30 companies in parallel. It is **designed to be idempotent** — safe to re-run at any time without side effects.

#### Section 1: Runtime Parameter Configuration

```python
dbutils.widgets.text("catalog",       "main",       "Unity Catalog catalog")
dbutils.widgets.text("env",           "dev",        "Environment (dev/prod)")
dbutils.widgets.text("start_date",    "2020-01-01", "Earliest filing date")
dbutils.widgets.text("ticker_filter", "",           "Comma-separated tickers (empty=all)")
```

`dbutils.widgets` is Databricks' parameter injection mechanism. The DAB job passes these values via `base_parameters` in `databricks.yml`. Default values are used for interactive runs.

- `catalog`: Unity Catalog name (e.g., `main`)
- `env`: Runtime environment; switches behavior between `dev` and `prod`
- `start_date`: Only filings dated `2020-01-01` or later are ingested
- `ticker_filter`: Comma-separated list of tickers to process; empty means all 30 companies

```python
TICKER_SUBSET = [t.strip() for t in TICKER_FILTER.split(",") if t.strip()] if TICKER_FILTER else []
```

This list comprehension converts `"AAPL, MSFT, GOOGL"` into `["AAPL", "MSFT", "GOOGL"]`. The double check `t.strip()` and `if t.strip()` removes empty strings that result from trailing commas.

#### Section 2: Medallion Schema Creation (SQL)

```sql
CREATE SCHEMA IF NOT EXISTS main.finsage_bronze;
CREATE SCHEMA IF NOT EXISTS main.finsage_silver;
CREATE SCHEMA IF NOT EXISTS main.finsage_gold;
```

`IF NOT EXISTS` makes each statement idempotent. The three schemas represent the three data quality tiers (raw → cleaned → analytical).

#### Section 3: Download Log Table Creation (SQL)

```sql
CREATE TABLE IF NOT EXISTS main.finsage_bronze.sec_filings_download_log (
    ticker            STRING,
    form_type         STRING,
    last_successful_run DATE,
    status            STRING,
    retry_count       INT,
    error_message     STRING,
    updated_at        TIMESTAMP
) USING DELTA;
```

This table manages state across job runs to guarantee idempotency. It prevents a given `(ticker, form_type)` combination from being downloaded twice on the same day.

#### Section 4: Volume Creation

```sql
CREATE VOLUME IF NOT EXISTS main.finsage_bronze.raw_filings;
```

A Unity Catalog volume is a filesystem path backed by object storage. Raw SEC filing HTML files are stored here. The access path is `/Volumes/main/finsage_bronze/raw_filings/`.

#### Section 5: Configuration Constants and 30-Ticker List

```python
VOLUME_PATH = f"/Volumes/{CATALOG}/finsage_bronze/raw_filings"
USER_AGENT  = "Arsaga Partners digvijay@arsaga.jp"
LOG_TABLE   = f"{CATALOG}.finsage_bronze.sec_filings_download_log"
MAX_RETRIES = 3
MAX_CONCURRENT_WORKERS = 3

_ALL_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "GS", "BAC", "V", "MA",
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "WMT", "KO", "NKE", "MCD", "SBUX",
    "TSLA", "F", "GM", "RIVN", "LCID", "CRM", "SNOW", "PLTR", "NET", "DDOG"
]
```

- `USER_AGENT`: Included in all request headers per SEC EDGAR Terms of Service
- `MAX_CONCURRENT_WORKERS = 3`: SEC enforces a rate limit of 10 requests per second; 3 parallel workers provides a safe margin
- The 30 tickers span Technology, Finance, Healthcare, Consumer Staples, Automotive, and SaaS, enabling cross-sector analysis

#### Section 6: Pre-flight Idempotency Check

```python
today = date.today()
try:
    df_completed = spark.table(LOG_TABLE).filter(
        (col("status") == "SUCCESS") & (col("last_successful_run") == today)
    )
    completed_tasks = set([(row.ticker, row.form_type) for row in df_completed.collect()])
except Exception:
    completed_tasks = set()
```

Before downloads begin, this fetches all `(ticker, form_type)` pairs that already succeeded today as a set. The `try/except` safely handles the case where the log table is empty on first run. The set is referenced by worker threads to avoid Spark session thread-safety issues.

#### Section 7: Thread-Safe Download Worker

```python
def download_filing(ticker, form_type):
    if (ticker, form_type) in completed_tasks:
        return (ticker, form_type, "SKIPPED", 0, "")

    dl = Downloader("FinSage", USER_AGENT, VOLUME_PATH)
    success = False
    retries = 0
    error_msg = ""

    while not success and retries < MAX_RETRIES:
        old_stdout = sys.stdout
        sys.stdout = my_stdout = StringIO()
        try:
            dl.get(form_type, ticker, after="2020-01-01")
            output = my_stdout.getvalue()
            if "Error occurred while downloading" in output or "503" in output:
                raise Exception("SEC API Error/503 Detected.")
            success = True
        except Exception as e:
            retries += 1
            error_msg = str(e)
            time.sleep(10 * retries)
        finally:
            sys.stdout = old_stdout
```

This worker has several deliberate design choices:

1. **stdout capture**: The `sec-edgar-downloader` library prints errors to stdout rather than raising exceptions. `sys.stdout` is redirected to a `StringIO` buffer to capture and inspect the output for error keywords.
2. **Exponential backoff**: `time.sleep(10 * retries)` — 10 seconds after the first failure, 20 after the second, 30 after the third. This accommodates temporary SEC API load spikes.
3. **`finally` clause**: Restores stdout even when an exception occurs, preventing corrupted notebook output.
4. **Idempotency check**: The function's first action is to check `completed_tasks` and return early if the download already completed.

#### Section 8: Parallel Execution and Result Collection

```python
with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
    futures = [executor.submit(download_filing, t, f) for t in TICKERS for f in FORM_TYPES]
    for future in as_completed(futures):
        results.append(future.result())
```

`ThreadPoolExecutor` runs up to 3 threads in parallel. `as_completed()` yields results as each thread finishes. With 2 form types (10-K, 10-Q) × 30 tickers, up to 60 tasks are processed 3 at a time.

#### Section 9: Atomic State Update in Delta Lake

```python
if processed_results:
    df_updates = spark.createDataFrame(processed_results, schema=schema)
    df_updates = df_updates.withColumn(
        "last_successful_run",
        expr("IF(status = 'SUCCESS', current_date(), cast(null as date))")
    ).withColumn("updated_at", current_timestamp())

    dt_log = DeltaTable.forName(spark, LOG_TABLE)
    dt_log.alias("t").merge(
        df_updates.alias("s"),
        "t.ticker = s.ticker AND t.form_type = s.form_type"
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
```

A Delta Lake MERGE (upsert) updates state atomically. Existing `(ticker, form_type)` records are updated; new ones are inserted. `IF(status = 'SUCCESS', current_date(), null)` ensures `last_successful_run` is only updated on success, leaving failed records eligible for retry on the next run.

---

### Notebook 02 — `02_bronze_autoloader.py`: Bronze Layer Auto Loader & SEC API Ingestion

This notebook is the core of the Bronze layer. It ingests data from two independent sources: (1) physical files on the volume via Databricks Auto Loader streaming, and (2) XBRL JSON data from the SEC EDGAR CompanyFacts API in batch.

#### Section 1: Reset Flag (Emergency Use)

```python
RESET_PIPELINE = False
if RESET_PIPELINE:
    spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.finsage_bronze.filings")
    spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.finsage_bronze.ingestion_errors")
    spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.finsage_bronze.xbrl_companyfacts_raw")
    dbutils.fs.rm(f"/Volumes/{CATALOG}/finsage_bronze/checkpoints", recurse=True)
```

Setting `RESET_PIPELINE = True` drops all Bronze tables and Auto Loader checkpoints, enabling a full re-ingestion from scratch. This flag serves as a safety mechanism — its presence in code makes any intentional reset explicit and visible in code review.

#### Section 2: Bronze Table Creation (SQL)

```sql
CREATE TABLE IF NOT EXISTS main.finsage_bronze.filings (
    filing_id         STRING,
    ticker            STRING,
    filing_type       STRING,
    accession_number  STRING,
    fiscal_year       INT,
    file_path         STRING,
    content           BINARY,       -- raw binary content of the file
    file_size_bytes   LONG,
    ingestion_status  STRING,
    ingested_at       TIMESTAMP
) TBLPROPERTIES (delta.enableChangeDataFeed = true);
```

The `content BINARY` column stores the entire HTML file as binary. The Silver layer decodes it with `decode(col("content"), "UTF-8")`. `delta.enableChangeDataFeed = true` enables Change Data Capture (CDC) on this table, allowing downstream layers to track inserts, updates, and deletes.

#### Section 3: Auto Loader File Streaming (Core Logic)

```python
df_bronze = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("cloudFiles.schemaLocation", schema_location)
    .option("recursiveFileLookup", "true")
    .load(volume_path)
    .withColumn("file_path",        col("_metadata.file_path"))
    .withColumn("ticker",           split(col("file_path"), "/").getItem(6))
    .withColumn("filing_type",      split(col("file_path"), "/").getItem(7))
    .withColumn("accession_number", split(col("file_path"), "/").getItem(8))
    .withColumn("year_short",       split(col("accession_number"), "-").getItem(1))
    .withColumn("fiscal_year",      concat(lit("20"), col("year_short")).cast("int"))
    .withColumn(
        "filing_id",
        concat_ws("-", col("ticker"), col("filing_type"), col("fiscal_year"), col("accession_number"))
    )
    ...
)
```

**How Auto Loader (`cloudFiles`) works:**

- `format("cloudFiles")`: Databricks' native incremental stream processor. Automatically detects new files in a directory.
- `cloudFiles.format = "binaryFile"`: Reads files as raw binary. Auto Loader provides file metadata (path, size, modification time) in the built-in `_metadata` column.
- `cloudFiles.schemaLocation`: Persists schema information to a checkpoint path and manages schema evolution (e.g., new columns) automatically.
- `recursiveFileLookup = true`: Searches subdirectories recursively.

**Metadata extraction from the file path:**

The path structure after `sec-edgar-downloader` completes is:
`/Volumes/main/finsage_bronze/raw_filings/sec-edgar-filings/AAPL/10-K/0000320193-21-000105/...`

`split(col("file_path"), "/")` splits on slashes, then index positions extract each element:
- Index 6 → ticker (e.g., `AAPL`)
- Index 7 → form type (e.g., `10-K`)
- Index 8 → accession number (e.g., `0000320193-21-000105`)

The `"-21-"` segment (index 1) of the accession number is the two-digit filing year. `concat(lit("20"), col("year_short")).cast("int")` converts `"21"` → `"2021"` → `2021`.

**Stream write configuration:**

```python
df_bronze.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", checkpoint_path)
    .option("badRecordsPath",     bad_records_path)
    .option("mergeSchema",        "true")
    .trigger(availableNow=True)
    .toTable(target_table)
```

- `outputMode("append")`: Bronze is append-only; existing data is never overwritten.
- `checkpointLocation`: Ensures exactly-once processing. The same file cannot be ingested twice.
- `badRecordsPath`: Failed records are written to a separate path without stopping the stream.
- `trigger(availableNow=True)`: Processes all currently available files and then stops — batch-like behavior suitable for a daily scheduled job.

#### Section 4: SEC EDGAR CompanyFacts API Ingestion

**Step 0 — Skip check for today's data:**

```python
df_existing = spark.sql("""
    SELECT ticker FROM main.finsage_bronze.xbrl_companyfacts_raw
    WHERE to_date(fetched_at) = current_date() AND api_status = 'success'
""")
already_fetched_today = [row["ticker"] for row in df_existing.collect()]
```

Tickers already successfully fetched today are skipped. The API is called at most once per ticker per day.

**Step 1 — Build ticker-to-CIK mapping:**

```python
company_map_url = "https://www.sec.gov/files/company_tickers.json"
company_map_resp = session.get(company_map_url, headers=HEADERS, timeout=30)
company_map = company_map_resp.json()

for item in company_map.values():
    ticker = item.get("ticker", "").upper()
    if ticker in TICKERS:
        ticker_to_cik[ticker] = str(item.get("cik_str", "")).zfill(10)
```

The SEC publishes a mapping of all registered company tickers to their CIK (Central Index Key) numbers. CIK is the SEC's unique numeric identifier for each registrant. `zfill(10)` zero-pads to 10 digits (e.g., `320193` → `0000320193`), the format required by SEC API URLs.

**Step 2 — Fetch CompanyFacts JSON:**

```python
source_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
resp = session.get(source_url, headers=HEADERS, timeout=45)
if resp.status_code == 200:
    payload = resp.json()
    api_rows.append(Row(
        snapshot_id=str(uuid4()), ticker=ticker, cik=cik,
        entity_name=payload.get("entityName"), source_url=source_url,
        raw_json=resp.text, api_status="success",
        ...
    ))
```

`raw_json=resp.text` stores the entire JSON payload as a string in Bronze. The Silver layer parses it with `json.loads()`. `uuid4()` generates a unique `snapshot_id` to distinguish multiple daily snapshots of the same ticker.

**Steps 3 & 4 — Schema definition and Delta write:**

```python
api_schema = StructType([
    StructField("snapshot_id",      StringType(),  True),
    StructField("raw_json",         StringType(),  True),
    StructField("http_status_code", IntegerType(), True),
    ...
])
spark.createDataFrame(api_rows, schema=api_schema)
    .withColumn("fetched_at", current_timestamp())
    .write.format("delta").mode("append")
    .saveAsTable(f"{CATALOG}.finsage_bronze.xbrl_companyfacts_raw")
```

An explicit Spark schema ensures type consistency. `.mode("append")` upholds the Bronze append-only principle. `current_timestamp()` records the Spark server time as `fetched_at`.

---

### Notebook 03 — `03_silver_decoder.py`: Silver Layer XBRL Decoding & Text Extraction

This notebook is the parsing engine of the pipeline. It extracts structured information from raw Bronze data through two completely independent paths: (A) XBRL numeric data, and (B) 10-K text sections.

#### Part A: XBRL CompanyFacts → `financial_statements` Table

**`TARGET_CONCEPT_MAP` — the core of normalization:**

```python
TARGET_CONCEPT_MAP = {
    "Revenues":                                                    "revenue",
    "SalesRevenueNet":                                             "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax":         "revenue",
    "RevenuesNetOfInterestExpense":                                "revenue",
    "NetIncomeLoss":                                               "net_income",
    "GrossProfit":                                                 "gross_profit",
    "OperatingIncomeLoss":                                         "operating_income",
    "NetCashProvidedByUsedInOperatingActivities":                  "operating_cash_flow",
    "Assets":                                                      "total_assets",
    "StockholdersEquity":                                          "equity",
    "LongTermDebt":                                                "long_term_debt",
    "ResearchAndDevelopmentExpense":                               "rd_expense",
    ...
}
```

This map is the canonical source of truth for metric normalization. Because SEC XBRL filings use different tag names for the same concept across companies and time periods, this map unifies 27 XBRL concept names into 11 normalized metric names. The map is defined in `src/finsage/constants.py` and covered by tests in `tests/unit/test_normalizer.py`.

**`flatten_companyfacts` function — unwrapping the deeply nested JSON:**

```python
def flatten_companyfacts(row):
    out = []
    payload = json.loads(row.raw_json)
    us_gaap = payload.get("facts", {}).get("us-gaap", {})
    for concept, concept_body in us_gaap.items():
        normalized_line_item = TARGET_CONCEPT_MAP.get(concept)
        if not normalized_line_item:
            continue  # Skip XBRL concepts not in TARGET_CONCEPT_MAP
        units_map = concept_body.get("units", {})
        for unit, entries in units_map.items():
            for e in entries:
                filing_type = e.get("form")
                fiscal_year = e.get("fy")
                if filing_type not in ("10-K", "10-Q") or fiscal_year is None:
                    continue
                out.append(Row(
                    ticker=row.ticker,
                    normalized_line_item=normalized_line_item,
                    value=float(e.get("val")),
                    fiscal_year=int(fiscal_year),
                    fiscal_period=e.get("fp"),
                    ...
                ))
    return out
```

The SEC CompanyFacts JSON is deeply nested:

```
facts → us-gaap → Revenues → units → USD → [{filed: "2021-10-29", fy: 2021, fp: "FY", val: 365817000000, ...}]
```

This function is called via `rdd.flatMap()`, expanding one Bronze row (one JSON blob) into many normalized financial metric rows.

**Deterministic deduplication via SHA-256:**

```python
.withColumn(
    "statement_id",
    sha2(concat_ws(
        "||",
        coalesce(col("ticker"),      lit("")),
        coalesce(col("accession"),   lit("")),
        coalesce(col("raw_line_item"), lit("")),
        coalesce(col("unit"),        lit("")),
        coalesce(col("period_end"),  lit("")),
    ), 256)
)
```

`statement_id` is a hash of `(ticker + accession + XBRL concept + unit + period end date)`. The `||` separator prevents hash collisions that could arise if different field combinations produced the same concatenated string.

**Window function to select the most recent snapshot:**

```python
window_spec = Window.partitionBy("statement_id").orderBy(
    col("source_fetched_at").desc(),
    col("filing_date").desc_nulls_last(),
)
df_financials_latest = (
    df_financials
    .withColumn("rn", row_number().over(window_spec))
    .filter(col("rn") == 1)
    .drop("rn")
)
```

Among rows with the same `statement_id` (from multiple daily API snapshots), only the most recent is retained. `row_number()` assigns a rank within each group; `.filter(col("rn") == 1)` keeps only the top-ranked row.

**Idempotent MERGE write:**

```python
if spark.catalog.tableExists(silver_table):
    DeltaTable.forName(spark, silver_table).alias("t").merge(
        df_financials_latest.alias("s"), "t.statement_id = s.statement_id"
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
else:
    df_financials_latest.write.format("delta").saveAsTable(silver_table)
```

If the table already exists, a MERGE (upsert) updates existing records and inserts new ones. On first run, a direct write is used. This ensures no data duplication regardless of how many times the pipeline is re-run.

#### Part B: 10-K Text Sections → `filing_sections` Table

**`SECTION_RULES` — regex-based boundary definitions:**

```python
SECTION_RULES = {
    "Business": {
        "start_patterns": [r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+1\b(?!\s*[ab]\b)"],
        "end_patterns":   [r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+1a\b",
                           r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+2\b"],
        "min_words": 250, "fallback_chars": 250000,
    },
    "Risk Factors": {
        "start_patterns": [r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+1a\b"],
        ...
        "min_words": 400,
    },
    "MD&A": {
        "start_patterns": [r"(?im)^[\s>\-\.\(\)\d]{0,12}item\s+7\b(?!\s*a\b)"],
        ...
    },
}
```

Regex breakdown:

- `(?im)`: Case-insensitive (`i`) and multiline (`m`) — `^` matches the start of each line.
- `[\s>\-\.\(\)\d]{0,12}`: Tolerates up to 12 leading noise characters (whitespace, `>`, `-`, digits) — necessary because 10-K HTML formatting varies across filers.
- `item\s+1\b`: One or more whitespace characters followed by a word boundary, preventing a match on `item 1a`.
- `(?!\s*[ab]\b)`: Negative lookahead — does not match "item 1a" or "item 1b", anchoring the start to Item 1 (Business) only.

**HTML cleaning pipeline:**

```python
df_processed = (
    df_bronze_clean
    .withColumn("raw_text",       decode(col("content"), "UTF-8"))
    .withColumn("main_doc",       expr("substring_index(raw_text, '</DOCUMENT>', 1)"))
    .withColumn("no_images",      regexp_replace(col("main_doc"),  r"(?is)<img[^>]*src=[\"']data:image/[^>]*>", " "))
    .withColumn("no_script",      regexp_replace(col("no_images"), r"(?is)<script[^>]*>.*?</script>", " "))
    .withColumn("no_style",       regexp_replace(col("no_script"), r"(?is)<style[^>]*>.*?</style>", " "))
    .withColumn("text_with_breaks", regexp_replace(col("no_style"),
        r"(?i)</?(div|p|br|tr|li|table|...)[^>]*>", "\n"))
    .withColumn("no_html",        regexp_replace(col("text_with_breaks"), "<[^>]+>", " "))
    .withColumn("clean_text",     regexp_replace(col("no_html"),   "\u00a0", " "))
    ...
)
```

The cleaning pipeline proceeds in stages:

1. `decode(content, "UTF-8")`: Converts binary to text.
2. `substring_index(raw_text, '</DOCUMENT>', 1)`: SEC EDGAR files may contain multiple embedded documents; only the first (the main filing document) is extracted.
3. Removal of Base64-encoded inline images — large and irrelevant to text processing.
4. Removal of `<script>` and `<style>` tags (JavaScript and CSS are noise).
5. Structural tags (`<div>`, `<p>`, `<br>`, `<tr>`, etc.) are replaced with newlines to preserve paragraph structure.
6. Remaining HTML tags are replaced with spaces.
7. Non-breaking spaces (`\u00a0`) are normalized to regular spaces.

**Section extraction UDF:**

```python
split_udf = udf(
    extract_sections_hardened,
    StructType([
        StructField("sections", ArrayType(StructType([
            StructField("section_name", StringType()),
            StructField("section_text", StringType()),
            StructField("word_count",   IntegerType()),
        ]))),
        StructField("error", StringType()),
    ])
)
```

Spark's `udf()` integrates this Python function into distributed processing. The return schema is explicit: `sections` (an array of section records) and `error` (an error message or null).

**`_choose_best_block` — best-match selection logic:**

```python
def _choose_best_block(text, rule):
    starts = _collect_positions(rule["start_patterns"], text)
    ends   = _collect_positions(rule["end_patterns"],   text)
    if not starts:
        return None
    doc_len, best, best_score = max(len(text), 1), None, -1
    for s in starts:
        end_candidates = [e for e in ends if e > s + 25]
        e = end_candidates[0] if end_candidates else min(len(text), s + rule["fallback_chars"])
        candidate  = text[s:e].strip()
        word_count = len(candidate.split())
        if word_count < rule["min_words"]:
            continue
        score = word_count + ((s / doc_len) * 250)
        if score > best_score:
            best_score = score
            best = {"section_text": candidate, "word_count": word_count, ...}
    return best
```

Complex filings like 10-Ks contain "Item 1" in both the table of contents and the body. A scoring system prevents selecting a table of contents entry as the section start:

- Blocks with fewer than `min_words` (250 or 400) are excluded (table of contents entries have very few words).
- `score = word_count + ((s / doc_len) * 250)`: Blocks positioned later in the document (the body appears after the table of contents) score higher.
- The highest-scoring block is selected as the true section.

---

### Notebook 04 — `04_gold_metrics.py`: Gold Layer Financial Metric Aggregation

This notebook refines Silver normalized financial metrics into a wide, immediately analytics-ready table. It applies strict quality filters, selects canonical accession numbers, and computes derived KPIs including year-over-year growth rates.

#### Section 1: Time-Period Filtering

```python
df = (
    spark.table(silver_table)
    .filter(col("filing_type").rlike("^10-K"))
    .filter(col("fiscal_period") == "FY")
    .filter(col("fiscal_year") >= 2020)
    .withColumn("duration_days", datediff(col("period_end_dt"), col("period_start_dt")))
    ...
)
```

- `filing_type.rlike("^10-K")`: Annual filings only; `rlike` supports regex, so amended filings (e.g., `10-K/A`) are also included.
- `fiscal_period == "FY"`: XBRL "FY" denotes a complete fiscal year; Q1–Q4 quarterly data is excluded.
- `fiscal_year >= 2020`: Analysis is scoped to the most recent 5 years.
- `duration_days`: Computed to verify that the period is exactly one year (350–380 days), filtering out quarterly or multi-year periods.

#### Section 2: Concept Priority-Based Deduplication

```python
concept_priority = (
    when((col("normalized_line_item") == "revenue") &
         (col("raw_line_item") == "RevenueFromContractWithCustomerExcludingAssessedTax"), lit(1))
    .when((col("normalized_line_item") == "revenue") &
          (col("raw_line_item") == "SalesRevenueNet"), lit(2))
    .when((col("normalized_line_item") == "revenue") &
          (col("raw_line_item") == "Revenues"), lit(3))
    ...
    .otherwise(lit(99))
)
```

When multiple XBRL concepts map to the same normalized metric (e.g., `revenue`), this defines which concept takes precedence. Priority 1 is highest. `RevenueFromContractWithCustomerExcludingAssessedTax` (the post-ASC 606 standard) is the preferred revenue concept because it reflects the current accounting standard.

#### Section 3: Period Fit Validation

```python
.withColumn("annual_fit_score", when(
    col("is_duration_metric") &
    col("duration_days").between(350, 380) &
    (col("period_end_year") == col("fiscal_year")),
    lit(1)
).otherwise(lit(0)))
.withColumn("instant_fit_score", when(
    col("is_instant_metric") &
    (col("period_end_year") == col("fiscal_year")),
    lit(1)
).otherwise(lit(0)))
```

Two types of metrics are handled differently:

- **Flow metrics** (`is_duration_metric`): Revenue, profit, etc. — cumulative values over a period. The period must be exactly one year (350–380 days). Periods shorter than 350 days are likely quarterly; periods longer than 380 days may be transition or restatement periods.
- **Instant metrics** (`is_instant_metric`): Total assets, liabilities, etc. — point-in-time balances at the end of the fiscal year.

#### Section 4: Canonical Accession Number Selection

```python
df_accession_quality = (
    df.withColumn("usable_fact_flag", when(..., lit(1)).otherwise(lit(0)))
    .groupBy("ticker", "company_name", "fiscal_year", "accession")
    .agg(
        spark_sum(when(required_metric_flag == 1, col("usable_fact_flag")).otherwise(lit(0)))
            .alias("required_metric_hits"),
        countDistinct(when(col("usable_fact_flag") == 1, col("normalized_line_item")))
            .alias("distinct_metric_coverage"),
        spark_max("filing_date_dt").alias("latest_filing_date"),
    )
)

accession_window = Window.partitionBy("ticker", "fiscal_year").orderBy(
    col("required_metric_hits").desc(),
    col("distinct_metric_coverage").desc(),
    col("latest_filing_date").desc(),
)
```

When multiple accession numbers exist for the same `(ticker, fiscal_year)` (e.g., original and amended filings), the "best" filing is selected by:

1. `required_metric_hits`: Number of the 6 required metrics covered (highest priority).
2. `distinct_metric_coverage`: Total number of usable distinct metrics.
3. `latest_filing_date`: Most recent filing as a tiebreaker.

This guarantees that a single canonical filing is used for aggregation, preventing data from different amended filings from mixing.

#### Section 5: Metric Aggregation (Pivot)

```python
df_base = (
    df_best_fact
    .groupBy("ticker", "company_name", "fiscal_year")
    .agg(
        spark_max(when(col("normalized_line_item") == "revenue",        col("value"))).alias("revenue"),
        spark_max(when(col("normalized_line_item") == "net_income",     col("value"))).alias("net_income"),
        spark_max(when(col("normalized_line_item") == "gross_profit",   col("value"))).alias("gross_profit_raw"),
        spark_max(when(col("normalized_line_item") == "total_assets",   col("value"))).alias("total_assets"),
        spark_max(when(col("normalized_line_item") == "equity",         col("value"))).alias("equity"),
        ...
    )
)
```

`spark_max(when(condition, value))` is the idiomatic Spark conditional aggregation pattern for pivoting normalized rows into named columns. Each `normalized_line_item` value becomes a separate column in the resulting wide table.

#### Section 6: Derived Metric Computation

```python
df_metrics = (
    df_base
    .withColumn("gross_profit",
        coalesce(col("gross_profit_raw"), col("revenue") - col("cost_of_revenue")))
    .withColumn("total_debt",
        when(col("short_term_debt").isNull() & col("long_term_debt").isNull(), lit(None).cast("double"))
        .otherwise(coalesce(col("short_term_debt"), lit(0.0)) + coalesce(col("long_term_debt"), lit(0.0))))
    .withColumn("gross_margin_pct",
        when(col("revenue").isNotNull() & (col("revenue") != 0) & col("gross_profit").isNotNull(),
             col("gross_profit") / col("revenue")))
)
```

- `gross_profit`: Uses the directly reported XBRL value if available; falls back to `revenue - cost_of_revenue` via `coalesce`.
- `total_debt`: If both short-term and long-term debt are null, the result is null (no data); otherwise, null values are treated as zero before summing.
- `gross_margin_pct`: Guards against division by zero with an explicit `revenue != 0` check.

**Year-over-year (YoY) growth calculation:**

```python
yoy_window = Window.partitionBy("ticker").orderBy("fiscal_year")
df_metrics = (
    df_metrics
    .withColumn("prior_year_revenue", lag("revenue").over(yoy_window))
    .withColumn("revenue_yoy_growth_pct",
        when(col("prior_year_revenue").isNotNull() & (col("prior_year_revenue") != 0),
             (col("revenue") - col("prior_year_revenue")) / col("prior_year_revenue")))
    .withColumn("debt_to_equity",
        when(col("equity").isNotNull() & (col("equity") != 0) & col("total_debt").isNotNull(),
             col("total_debt") / col("equity")))
)
```

`lag("revenue")` retrieves the prior fiscal year's revenue for the same ticker. `partitionBy("ticker")` ensures the window operates within each company. The growth rate formula is `(current − prior) / prior`, guarded against null and zero divisors.

#### Section 7: Data Quality Score and Final Write

```python
validated_metric_count = (
    when(col("revenue").isNotNull(),             lit(1)).otherwise(lit(0)) +
    when(col("net_income").isNotNull(),          lit(1)).otherwise(lit(0)) +
    ...
    when(col("rd_expense").isNotNull(),          lit(1)).otherwise(lit(0))
)

df_gold = (
    df_metrics
    .withColumn("data_quality_score", validated_metric_count / lit(9.0))
    ...
)
```

For each of the 9 core metrics, a value of 1 is added if the metric is non-null. Dividing by 9.0 yields a score in the range [0, 1]. A `data_quality_score` of 1.0 means all 9 metrics are populated. This score allows downstream consumers to filter for high-completeness records.

---

### Notebook 05 — `05_vector_chunker.py`: Vector Chunk Generation & Vector Search Index

This notebook builds the RAG (Retrieval-Augmented Generation) foundation of FinSage. It splits text sections into LLM-compatible chunks and registers them in a Databricks Vector Search index for semantic similarity search.

#### Section 1: Chunking Configuration

```python
SOURCE_TABLE          = "main.finsage_silver.filing_sections"
TARGET_TABLE          = "main.finsage_gold.filing_section_chunks"
EMBEDDING_MODEL       = "text-embedding-3-large"
CHUNK_TOKENS          = 512
CHUNK_OVERLAP_TOKENS  = 64
CHUNK_VERSION         = f"tok_{CHUNK_TOKENS}_{CHUNK_OVERLAP_TOKENS}_v1"
```

- `CHUNK_TOKENS = 512`: Maximum tokens per chunk. While `text-embedding-3-large` supports up to 8,192 tokens, 512 tokens balances retrieval precision and context retention.
- `CHUNK_OVERLAP_TOKENS = 64`: Overlap between consecutive chunks prevents semantic loss at chunk boundaries.
- `CHUNK_VERSION`: A version string encoding the chunking configuration. Changing chunking parameters updates this string, distinguishing new chunks from existing ones in idempotent merges.

#### Section 2: Lazy Initialization of the tiktoken Encoder

```python
_ENCODING = None

def get_encoding():
    global _ENCODING
    if _ENCODING is None:
        try:
            _ENCODING = tiktoken.encoding_for_model(EMBEDDING_MODEL)
        except KeyError:
            _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING
```

`tiktoken` is OpenAI's tokenizer library. Lazy initialization (creating the encoder only on first call) avoids serialization issues in Spark's distributed execution environment. `cl100k_base` (used by GPT-4) is the fallback encoding.

#### Section 3: Deterministic Chunk ID Generation

```python
def deterministic_chunk_id(
    filing_id: str,
    section_name: str,
    chunk_index: int,
    chunk_text: str,
    chunk_version: str,
) -> str:
    payload = {
        "filing_id":    str(filing_id),
        "section_name": str(section_name),
        "chunk_index":  int(chunk_index),
        "chunk_text":   chunk_text,
        "chunk_version": chunk_version,
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

- `json.dumps(sort_keys=True)`: Sorts dictionary keys to ensure the same content always produces the same JSON string.
- `separators=(",", ":")`: Removes whitespace from the JSON output for compact, unambiguous serialization.
- SHA-256 hash: Identical inputs always produce the same ID. This allows the pipeline to be re-run without changing chunk IDs, enabling accurate Vector Search index MERGEs.

#### Section 4: Token-Based Chunking via Pandas UDF

```python
@F.pandas_udf(chunk_array_schema)
def chunk_sections_udf(
    section_text_col: pd.Series,
    filing_id_col: pd.Series,
    section_name_col: pd.Series,
) -> pd.Series:
    enc  = get_encoding()
    step = CHUNK_TOKENS - CHUNK_OVERLAP_TOKENS  # = 512 - 64 = 448

    for text, filing_id, section_name in zip(section_text_col, filing_id_col, section_name_col):
        normalized = normalize_text(str(text))
        token_ids  = enc.encode(normalized)

        row_chunks  = []
        chunk_index = 0
        for start in range(0, len(token_ids), step):
            end        = min(start + CHUNK_TOKENS, len(token_ids))
            chunk_ids  = token_ids[start:end]
            chunk_text = enc.decode(chunk_ids).strip()
            cid = deterministic_chunk_id(...)
            row_chunks.append({
                "chunk_id":    cid,
                "chunk_index": chunk_index,
                "chunk_text":  chunk_text,
                "token_count": len(chunk_ids),
                ...
            })
            chunk_index += 1
            if end == len(token_ids):
                break
        out.append(row_chunks)
```

`@F.pandas_udf` is a vectorized UDF that receives data as `pd.Series` batches rather than row-by-row, providing significantly better performance than a standard Spark UDF.

The chunking algorithm:

- `step = 448`: Each chunk starts 448 tokens after the previous one (512 − 64 overlap).
- The full text is tokenized into a list of token IDs by `tiktoken`.
- Each slice `token_ids[start:end]` is decoded back to text using `enc.decode()`.
- The loop terminates when `end == len(token_ids)` (last chunk reached).

Visual representation (512 tokens, 64-token overlap):

```
[Chunk 0: tokens   0–511]
[Chunk 1: tokens 448–959]
[Chunk 2: tokens 896–1407]
              ^^^--- 64-token overlap
```

#### Section 5: Data Quality Guards

```python
dup_count = (
    df_chunks.groupBy("chunk_id").count()
    .filter(F.col("count") > 1).limit(1).count()
)
if dup_count > 0:
    raise RuntimeError("Duplicate chunk_id detected. Aborting write.")

bad_rows = (
    df_chunks.filter(
        F.col("chunk_text").isNull() |
        (F.col("token_count") <= 0) |
        (F.col("chunk_index") < 0)
    ).limit(1).count()
)
if bad_rows > 0:
    raise RuntimeError("Invalid chunk rows detected. Aborting write.")
```

Two assertions run before any write:

1. **Duplicate `chunk_id` check**: Verifies the deterministic ID logic is functioning correctly. Duplicate IDs would cause incorrect behavior in the Vector Search index MERGE.
2. **Invalid row check**: Fails fast if any chunk has a null `chunk_text`, a non-positive `token_count`, or a negative `chunk_index`.

`.limit(1).count()` is used deliberately — it returns 1 as soon as a single matching row is found, making the check fast even on large datasets.

#### Section 6: Vector Search Endpoint and Index Provisioning

```python
VECTOR_SEARCH_ENDPOINT_NAME  = "finsage_vs_endpoint"
INDEX_NAME                   = "main.finsage_gold.filing_chunks_index"
EMBEDDING_MODEL_ENDPOINT     = "databricks-bge-large-en"
PIPELINE_TYPE                = "TRIGGERED"
```

- `finsage_vs_endpoint`: The Databricks Vector Search endpoint (a compute resource similar to a cluster).
- `databricks-bge-large-en`: BGE (BAAI General Embedding) Large English model, served by Databricks Model Serving, which converts text to vector embeddings.
- `TRIGGERED` pipeline: Index updates are triggered explicitly rather than running continuously — cost-efficient for a daily batch pipeline.

**Retry logic with exponential backoff and jitter:**

```python
def _retryable_call(fn, retries: int = 8, base_sleep: float = 1.5, max_sleep: float = 20.0):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt == retries:
                break
            backoff = min(max_sleep, base_sleep * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
            time.sleep(backoff)
    raise last_exc
```

`2 ** (attempt - 1)` implements exponential backoff (1.5s, 3s, 6s, 12s, 20s, ...). Adding `random.uniform(0, 0.5)` jitter prevents multiple clients from retrying simultaneously (the "thundering herd" problem).

**Endpoint online polling:**

```python
def wait_for_endpoint_online(vsc, endpoint_name, timeout_sec):
    start = time.time()
    while True:
        if time.time() - start > timeout_sec:
            raise TimeoutError(...)
        ep    = _retryable_call(lambda: vsc.get_endpoint(name=endpoint_name))
        state = _normalize_state(_nested_get(ep, ("endpoint_status", "state"), ("status", "state")))
        if state == "ONLINE":
            return
        if state in {"FAILED", "ERROR"}:
            raise RuntimeError(...)
        time.sleep(POLL_SEC)  # polls every 15 seconds
```

Vector Search endpoints take several minutes to start. This function polls every 15 seconds, waiting up to 30 minutes. `_nested_get` handles differences in API response structure across SDK versions.

**Similarity search demonstration:**

```python
def search_financial_filings(query: str, num_results: int = 3):
    index   = vsc.get_index(endpoint_name="finsage_vs_endpoint", index_name=INDEX_NAME)
    results = index.similarity_search(
        query_text=query,
        columns_to_return=["ticker", "fiscal_year", "section_name", "chunk_text"],
        num_results=num_results,
    )
    docs = results.get("result", {}).get("data_array", [])
    return "\n---\n".join(
        f"[{d[0]} | {d[1]} | {d[2]}]\n{d[3]}" for d in docs
    )

print(search_financial_filings("What did Apple say about supply chain or manufacturing risks?"))
```

Once the index is built, natural language queries can retrieve the most semantically relevant filing sections. `similarity_search` embeds the query text using the same BGE Large model and returns the nearest neighbors by cosine similarity. This is the retrieval (R) component of a RAG system.

---

## 13. Future Work

The following items are defined in the repository structure or configuration files but are **not yet implemented**:

- **`src/agent/`, `src/app/`, `src/monitoring/`, `src/processing/`, `src/retrieval/`, `src/serving/`**: These directories exist as stubs. A RAG agent, application serving layer, and monitoring module are planned but not yet built.
- **`databricks/workflows/`**: Reserved for future Databricks Workflow YAML definitions (e.g., Delta Live Tables pipelines).
- **Terraform remote state backend**: `terraform/main.tf` includes a commented-out `azurerm` backend block. Remote state storage must be configured before Terraform is used in a multi-engineer or CI context.
- **Production `run_as` service principal**: `databricks.yml` includes commented-out `run_as` configuration for the `prod` target. Once the `finsage-service-principal` is provisioned in the Databricks Admin Console with its numeric application ID, this should be uncommented to decouple job execution from a human identity.
- **Cluster policy integration into DAB**: `terraform/main.tf` creates a `finsage_cluster_policy`, but `databricks.yml` does not yet reference it via `policy_id`. This is noted as future work in the Terraform file.
- **Terraform in CI**: Terraform is not part of the GitHub Actions pipeline. Infrastructure changes (cluster policies, secret scopes) must currently be applied manually.

---

## License

Internal project — Arsaga Partners. All rights reserved.
