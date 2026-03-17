# Challenges Log

A running record of significant technical challenges encountered during FinSage development, with problem description, root cause, and resolution. Used for post-mortems and to avoid repeating the same pitfalls.

---

## Challenge: Memory Explosion in PDF Parsing (Week 2)

**Status:** Resolved  
**Phase:** Silver-layer parsing (initial approach)

---

### Problem

The initial Week 2 pipeline failed during document parsing with a Spark runtime error. Jobs that ingested SEC filings and attempted to parse PDF content using a **PySpark UDF** (backed by **PyMuPDF**) consistently crashed with:

- **Exception type:** `SparkRuntimeException`
- **Error code:** `UDF_PYSPARK_ERROR.OOM`
- **SQLSTATE:** `39000`

Workers became unstable, and the driver reported that the Python side of the UDF had run out of memory. The failure was reproducible with moderate batch sizes and made it impossible to complete the Silver transformation in-process.

---

### Root Cause

1. **JVM–Python data transfer.** The UDF received large binary payloads (full PDF or document bytes) from the JVM for each row. Spark serializes these payloads, sends them to Python workers, and then serializes the UDF return values (e.g., extracted text) back to the JVM. For many rows with multi-megabyte documents, this created a large volume of data in flight and in Python heap.

2. **No offload of heavy work.** PyMuPDF ran inside the same Python process as the Spark executor. Parsing PDFs is memory- and CPU-intensive. Combining that with the copy of the document data in the executor’s memory led to rapid heap growth and OOM on the Python side.

3. **Amplification by parallelism.** With multiple tasks and partitions, several workers could hit the same pattern at once, making the failure systemic rather than limited to a single bad file.

The **SQLSTATE 39000** (external routine invocation exception) correctly indicated that the failure originated in the external Python UDF, not in Spark’s core JVM logic.

---

### Solution

We moved to a **decoupled parsing architecture**:

1. **Bronze:** Ingest and store only **raw binary content** (and metadata) in the Delta Lakehouse using Auto Loader. No parsing in this layer; we persist exactly what we received.

2. **Silver:** In a separate pipeline step, read from Bronze and call Databricks’ **`ai_parse_document`** to perform parsing. This runs on a serverless document-AI backend, so large binary data and CPU-heavy parsing are no longer executed inside Spark’s Python workers.

3. **No in-Spark PDF UDF.** We removed the PySpark UDF that used PyMuPDF. Spark’s role is limited to orchestration, passing references or small payloads where appropriate, and writing the parsed results to Silver tables.

This approach eliminates the JVM–Python bulk transfer and in-executor parsing that caused the OOM, and aligns with the medallion design: Bronze for raw, Silver for refined, structured content.

---

### Takeaways

- **UDFs and large binaries:** Be cautious with PySpark UDFs that accept or return large blobs; serialization and worker memory can dominate.
- **Use platform capabilities:** Offloading document parsing to a managed service (`ai_parse_document`) avoided reimplementing and tuning in-executor parsing.
- **Decoupling ingestion and parsing** improved both stability and observability: we can reprocess Silver from Bronze without re-downloading, and we can size/tune each stage independently.

---

*Last updated: March 2025*
