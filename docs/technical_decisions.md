# Technical Decisions

This document captures the rationale behind key architectural and scope decisions for FinSage, a financial intelligence platform on Databricks.

---

## 1. Scope: 30 Companies, ~3,000 Filings (Depth Over Breadth)

**Decision:** Ingest SEC filings for **30 companies** (targeting ~3,000 filings total) rather than attempting broad coverage across hundreds of tickers.

**Rationale:**

- **Depth enables validation.** A focused universe (10-Ks and 10-Qs for a fixed set of tickers over a defined time window) allows us to validate the full pipeline end-to-end—ingestion, parsing, and downstream analytics—without the operational and cost overhead of scaling prematurely.
- **Controlled variables.** With a fixed company set spanning Tech, Finance, Healthcare, Consumer, EV, and SaaS, we can compare parsing quality, schema stability, and performance across sectors while keeping cluster sizing and debugging tractable.
- **Incremental scaling path.** Once the medallion architecture (Bronze → Silver → Gold) is proven on 30 companies, expanding to more tickers is a configuration change rather than an architectural rewrite.

This choice prioritizes **reliable, repeatable pipelines** over raw volume in the early phase of the platform.

---

## 2. Parsing Strategy: Decoupled Ingestion vs. In-Process UDFs

**Decision:** Ingest **raw binary content** into the Bronze layer and perform document parsing in the **Silver layer** using Databricks’ native **`ai_parse_document`** (serverless), instead of parsing inside Spark via a PySpark UDF (e.g., PyMuPDF).

**Rationale:**

- **Avoid JVM–Python data transfer.** A PySpark UDF that receives large PDF (or other binary) payloads from the JVM and returns extracted text causes serialization and shuffling of big byte arrays across workers. This led to out-of-memory conditions on Python workers and `SparkRuntimeException: [UDF_PYSPARK_ERROR.OOM]` in our initial Week 2 pipeline.
- **Offload heavy work.** By persisting raw content in Bronze and invoking `ai_parse_document` in Silver, we move parsing to a serverless backend designed for document AI. Spark then orchestrates and writes results instead of hosting the parsing workload, reducing memory pressure and improving stability.
- **Clear separation of concerns.** Bronze = “what we received” (immutable raw); Silver = “what we understood” (parsed, typed). This aligns with the medallion model and makes reprocessing and schema evolution straightforward.

**Summary:** We chose a **decoupled parsing strategy** to eliminate in-worker memory explosion, leverage managed document AI, and keep the pipeline maintainable and scalable.

---

*Last updated: March 2025*
