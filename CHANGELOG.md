# Changelog

All notable changes to FinSage are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned
- Terraform remote state backend (Azure Blob Storage)
- Integration test execution as a separate CI job (gated on Databricks connectivity)
- `bundle run` step in CI for smoke-testing after prod deploy
- Databricks cluster policy ID wired into `databricks.yml` from Terraform output

---

## [1.1.0] — 2026-03-25

### Added
- **Runtime parameterisation** — all five notebooks now read `catalog`, `env`,
  `start_date`, and `ticker_filter` via `dbutils.widgets`, injected by the DAB job
  via `base_parameters`.
- **`src/finsage/constants.py`** — single source of truth for `TARGET_CONCEPT_MAP`.
  Tests now import from this module; the duplicate map in `test_normalizer.py` is removed.
- **`tests/conftest.py`** — pytest configuration: `sys.path` injection so unit tests
  import the `finsage` package without requiring a wheel install; `spark` session
  fixture for integration tests; `integration` marker registration.
- **`tests/integration/test_bronze_schema.py`** — integration test suite covering
  table existence, schema correctness, and data quality for all three Medallion layers.
- **`terraform/main.tf`** — foundational Terraform skeleton: Databricks provider,
  cluster policy, secret scope, and service principal data source.
- **`CHANGELOG.md`** — this file.
- **`run_as`** block in `databricks.yml` — prod job now declares the service principal
  that executes tasks, decoupling job execution from the deploying user's identity.
- **`pause_status: PAUSED`** override for the `dev` target — prevents the daily cron
  from firing in development environments.
- **`notification_email`** variable in `databricks.yml` — removes hardcoded personal
  email from the job definition.

### Changed
- `databricks.yml` variables expanded: `catalog`, `env`, `ticker_filter`,
  `start_date`, `notification_email` are now first-class bundle variables
  with per-target overrides.
- `tests/unit/test_normalizer.py` imports `TARGET_CONCEPT_MAP` and
  `VALID_NORMALIZED_METRICS` from `src.finsage.constants` — eliminates map duplication.
- `requirements.txt` pinned to minimum versions; added `pytest`, `pytest-cov`,
  `tiktoken`, `sec-edgar-downloader`.

### Security
- OAuth M2M authentication enforced in CI: `DATABRICKS_TOKEN` (PAT) replaced with
  `DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET` in both workflow jobs.

### Fixed
- `05_vector_chunker.py` — duplicate `search_financial_filings` definition removed;
  incorrect `columns_to_return` parameter corrected to `columns`.
- Missing newline at end of `05_vector_chunker.py`.

---

## [1.0.0] — 2026-03-25

### Added
- Initial production-grade Databricks Asset Bundle configuration (`databricks.yml`)
  with `dev` and `prod` targets.
- GitHub Actions CI/CD pipeline (`deploy.yml`): unit tests → bundle validate →
  deploy to prod, triggered on push to `main`.
- Five Databricks notebook source files implementing the four-layer Medallion
  architecture (Bronze → Silver → Gold → Vector).
- Unit test suite (`tests/unit/test_normalizer.py`) with 20 pytest assertions.
- Professional README with ASCII architecture diagrams, CLI cheatsheet,
  branch strategy table, and deployment reference.
