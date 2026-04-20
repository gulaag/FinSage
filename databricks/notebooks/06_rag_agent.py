# Databricks notebook source
# ==============================================================================
# FinSage | 06 — RAG Agent
#
# Builds a function-calling financial Q&A agent with two tools:
#   1. search_filings      — semantic search over filing_chunks_index (Vector Search)
#   2. get_company_metrics — structured lookup over company_metrics (Gold table)
#
# The agent is:
#   - Traced end-to-end with MLflow (every tool call + LLM call gets a span)
#   - Registered as a Unity Catalog model (main.finsage_gold.finsage_rag_agent)
#   - Deployed to Databricks Model Serving (finsage_agent_endpoint)
#
# LLM: databricks-meta-llama-3-3-70b-instruct (function-calling capable, READY)
# Framework: mlflow.pyfunc — no LangChain dependency
# ==============================================================================

# COMMAND ----------

# ── 1. Runtime Parameters ─────────────────────────────────────────────────────
dbutils.widgets.text("catalog",              "main",                                    "UC catalog")
dbutils.widgets.text("env",                  "dev",                                     "Environment")
dbutils.widgets.text("llm_endpoint",         "databricks-meta-llama-3-3-70b-instruct",  "LLM serving endpoint")
dbutils.widgets.text("vs_endpoint",          "finsage_vs_endpoint",                     "Vector Search endpoint")
dbutils.widgets.text("num_results",          "5",                                       "Top-k retrieval results")
dbutils.widgets.text("similarity_threshold", "0.6",                                     "Min similarity score (0-1)")

CATALOG              = dbutils.widgets.get("catalog")
ENV                  = dbutils.widgets.get("env")
LLM_ENDPOINT         = dbutils.widgets.get("llm_endpoint")
VS_ENDPOINT          = dbutils.widgets.get("vs_endpoint")
NUM_RESULTS          = int(dbutils.widgets.get("num_results"))
SIMILARITY_THRESHOLD = float(dbutils.widgets.get("similarity_threshold"))

VS_INDEX_NAME   = f"{CATALOG}.finsage_gold.filing_chunks_index"
METRICS_TABLE   = f"{CATALOG}.finsage_gold.company_metrics"
UC_MODEL_NAME   = f"{CATALOG}.finsage_gold.finsage_rag_agent"
AGENT_ENDPOINT  = "finsage_agent_endpoint"
MAX_ITERATIONS  = 5

print(f"[CONFIG] catalog={CATALOG} | env={ENV} | llm={LLM_ENDPOINT} | vs_index={VS_INDEX_NAME}")

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch mlflow databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ── 2. Imports + re-declare constants (required after restartPython wipes state) ──
import json
import logging
import mlflow
import mlflow.deployments
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
log = logging.getLogger("finsage-agent")

mlflow.set_registry_uri("databricks-uc")

# Re-read widgets — restartPython() clears all Python variables but widgets persist
CATALOG              = dbutils.widgets.get("catalog")
ENV                  = dbutils.widgets.get("env")
LLM_ENDPOINT         = dbutils.widgets.get("llm_endpoint")
VS_ENDPOINT          = dbutils.widgets.get("vs_endpoint")
NUM_RESULTS          = int(dbutils.widgets.get("num_results"))
SIMILARITY_THRESHOLD = float(dbutils.widgets.get("similarity_threshold"))

VS_INDEX_NAME  = f"{CATALOG}.finsage_gold.filing_chunks_index"
METRICS_TABLE  = f"{CATALOG}.finsage_gold.company_metrics"
UC_MODEL_NAME  = f"{CATALOG}.finsage_gold.finsage_rag_agent"
AGENT_ENDPOINT = "finsage_agent_endpoint"
MAX_ITERATIONS = 5

print(f"[CONFIG restored] catalog={CATALOG} | llm={LLM_ENDPOINT} | metrics={METRICS_TABLE}")

# COMMAND ----------

# ── 3. Pre-load Gold metrics into memory ──────────────────────────────────────
# company_metrics has only 180 rows — load once as a nested dict for zero-latency
# lookup inside the pyfunc serving container (no SQL warehouse needed at runtime).

def _load_metrics_cache(table: str) -> dict:
    df = spark.table(table).select(
        "ticker", "company_name", "fiscal_year",
        "revenue", "net_income", "gross_profit", "operating_income",
        "operating_cash_flow", "total_assets", "total_liabilities",
        "total_debt", "rd_expense", "gross_margin_pct",
        "revenue_yoy_growth_pct", "debt_to_equity", "data_quality_score",
    )
    cache = {}
    for row in df.collect():
        r = row.asDict()
        ticker = r.pop("ticker")
        fy = r.pop("fiscal_year")
        cache.setdefault(ticker.upper(), {})[fy] = r
    log.info("Metrics cache loaded: %d tickers", len(cache))
    return cache

METRICS_CACHE = _load_metrics_cache(METRICS_TABLE)
print(f"[CACHE] {len(METRICS_CACHE)} tickers loaded. Sample: {list(METRICS_CACHE.keys())[:5]}")

# COMMAND ----------

# ── 4. Tool: search_filings ───────────────────────────────────────────────────

@mlflow.trace(name="search_filings", span_type="TOOL")
def search_filings(
    query: str,
    ticker: str = None,
    section_name: str = None,
    fiscal_year: int = None,
    num_results: int = NUM_RESULTS,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> str:
    """
    Semantic search over SEC 10-K filing sections.
    Returns relevant passages with source metadata.
    """
    vsc = VectorSearchClient(disable_notice=True)
    index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX_NAME)

    filters = {}
    if ticker:
        filters["ticker"] = ticker.upper()
    if section_name and section_name in ("Business", "Risk Factors", "MD&A"):
        filters["section_name"] = section_name
    if fiscal_year:
        filters["fiscal_year"] = fiscal_year

    try:
        results = index.similarity_search(
            query_text=query,
            columns=["ticker", "fiscal_year", "section_name", "chunk_text"],
            filters=filters if filters else None,
            num_results=num_results,
            query_type="ANN",
        )
    except Exception as e:
        log.warning("Vector search failed: %s", e)
        return f"Search failed: {str(e)}"

    data = results.get("result", {}).get("data_array", [])
    if not data:
        return "No relevant passages found for this query."

    passages = []
    for row in data:
        ticker_val, fy, section, text = row[0], row[1], row[2], row[3]
        score = row[4] if len(row) > 4 else None
        if score is not None and score < similarity_threshold:
            continue
        passages.append(
            f"[Source: {ticker_val} | FY{int(fy)} | {section}]\n{text[:1200]}"
        )

    if not passages:
        return "No passages met the similarity threshold. Try a broader query."

    return "\n\n---\n\n".join(passages)


# Quick smoke test
_test = search_filings("supply chain risks manufacturing", ticker="AAPL", section_name="Risk Factors", num_results=2)
print("[search_filings test]", _test[:300])

# COMMAND ----------

# ── 5. Tool: get_company_metrics ──────────────────────────────────────────────

@mlflow.trace(name="get_company_metrics", span_type="TOOL")
def get_company_metrics(
    ticker: str,
    fiscal_year_start: int = None,
    fiscal_year_end: int = None,
    metrics_cache: dict = None,
) -> str:
    """
    Retrieves structured financial metrics for a company from the Gold table.
    Returns revenue, net income, margins, YoY growth, debt ratios.
    """
    cache = metrics_cache or METRICS_CACHE
    ticker_upper = ticker.upper()

    if ticker_upper not in cache:
        available = sorted(cache.keys())
        return f"No metrics found for ticker '{ticker_upper}'. Available tickers: {available}"

    ticker_data = cache[ticker_upper]
    years = sorted(ticker_data.keys())

    if fiscal_year_start:
        years = [y for y in years if y >= fiscal_year_start]
    if fiscal_year_end:
        years = [y for y in years if y <= fiscal_year_end]

    if not years:
        return f"No data for {ticker_upper} in the requested fiscal year range."

    def _fmt(v, pct=False):
        if v is None:
            return "N/A"
        if pct:
            return f"{v * 100:.1f}%"
        if abs(v) >= 1e9:
            return f"${v / 1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"${v / 1e6:.1f}M"
        return f"${v:,.0f}"

    lines = [f"Financial metrics for {ticker_upper} ({ticker_data[years[0]].get('company_name', ticker_upper)}):"]
    for fy in years:
        m = ticker_data[fy]
        lines.append(
            f"\nFY{int(fy)}:"
            f"\n  Revenue:              {_fmt(m.get('revenue'))}"
            f"\n  Net Income:           {_fmt(m.get('net_income'))}"
            f"\n  Gross Profit:         {_fmt(m.get('gross_profit'))}"
            f"\n  Operating Income:     {_fmt(m.get('operating_income'))}"
            f"\n  Operating Cash Flow:  {_fmt(m.get('operating_cash_flow'))}"
            f"\n  Total Assets:         {_fmt(m.get('total_assets'))}"
            f"\n  Total Debt:           {_fmt(m.get('total_debt'))}"
            f"\n  Gross Margin:         {_fmt(m.get('gross_margin_pct'), pct=True)}"
            f"\n  Revenue YoY Growth:   {_fmt(m.get('revenue_yoy_growth_pct'), pct=True)}"
            f"\n  Debt/Equity:          {str(round(m['debt_to_equity'], 2)) + 'x' if m.get('debt_to_equity') is not None else 'N/A'}"
            f"\n  Data Quality Score:   {m.get('data_quality_score', 0):.0%}"
        )
    return "\n".join(lines)


# Quick smoke test
_test2 = get_company_metrics("AAPL", fiscal_year_start=2022, fiscal_year_end=2023)
print("[get_company_metrics test]\n", _test2)

# COMMAND ----------

# ── 6. Tool schemas (OpenAI function-calling format) ──────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_filings",
            "description": (
                "Semantically search SEC 10-K filing text (Business, Risk Factors, MD&A sections). "
                "Use for qualitative questions about strategy, risks, products, competition, regulation, "
                "supply chain, or anything requiring direct quotes from annual reports."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query describing what to find in the filings."
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Optional stock ticker to restrict search (e.g. 'AAPL', 'MSFT'). Leave empty to search all companies."
                    },
                    "section_name": {
                        "type": "string",
                        "enum": ["Business", "Risk Factors", "MD&A"],
                        "description": "Optional section filter. Use 'Risk Factors' for risk/threat questions, 'MD&A' for management commentary, 'Business' for strategy/products."
                    },
                    "fiscal_year": {
                        "type": "integer",
                        "description": "Optional fiscal year to restrict search (e.g. 2024). Use when the question specifies 'most recent', 'latest', or a specific year. First call get_company_metrics to find the latest available year for that ticker, then pass it here."
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of passages to retrieve (default 5, max 10).",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_metrics",
            "description": (
                "Retrieve structured financial metrics for a company from SEC XBRL data. "
                "Use for numerical questions about revenue, profit, margins, debt, growth rates, "
                "or any quantitative financial comparison. Covers FY2020–FY2026 for 30 companies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. 'AAPL', 'NVDA', 'JPM')."
                    },
                    "fiscal_year_start": {
                        "type": "integer",
                        "description": "Optional start of fiscal year range (e.g. 2021)."
                    },
                    "fiscal_year_end": {
                        "type": "integer",
                        "description": "Optional end of fiscal year range (e.g. 2024)."
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]

TOOL_DISPATCH = {
    "search_filings":      lambda args: search_filings(**args),
    "get_company_metrics": lambda args: get_company_metrics(**args),
}

print(f"[TOOLS] Registered: {list(TOOL_DISPATCH.keys())}")

# COMMAND ----------

# ── 7. FinSageAgent — pyfunc model class ──────────────────────────────────────

SYSTEM_PROMPT = """\
You are FinSage, an expert financial analyst AI with access to SEC annual report filings (10-K) \
and structured financial metrics for 30 major public companies (2020–2026).

You have two tools:
1. search_filings — retrieves relevant passages from 10-K text sections (Risk Factors, MD&A, Business).
2. get_company_metrics — retrieves structured financial data: revenue, margins, growth rates, debt ratios.

Guidelines:
- Always use tools to ground your answer in actual data before responding.
- For numerical questions (revenue, growth %, margins), use get_company_metrics.
- For qualitative questions (risks, strategy, management commentary), use search_filings.
- For complex questions, use both tools.
- Always cite your sources: ticker, fiscal year, and section name.
- If data is unavailable, say so explicitly — never fabricate figures.
- Format numbers clearly ($B, %, bps). Be concise and precise.
- For "most recent" or "latest" filing questions: first call get_company_metrics to identify \
the latest fiscal year available for that ticker, then call search_filings with that specific \
fiscal_year to ensure all retrieved passages come from a single filing period.
- When citing text from filings: prefix direct quotes with [VERBATIM] and paraphrased content \
with [SUMMARY]. Never present a summary as a direct quote.
- When computing or presenting any financial ratio (margins, growth rates, leverage ratios), \
always state the formula explicitly on first use. \
Example: "Operating Margin (GAAP) = Operating Income ÷ Revenue"
"""


class FinSageAgent(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import json
        # Load the metrics cache artifact saved during logging
        with open(context.artifacts["metrics_cache"], "r") as f:
            raw = json.load(f)
        # Keys were stored as strings (JSON limitation) — restore int fiscal_year keys
        self._metrics_cache = {
            ticker: {int(fy): metrics for fy, metrics in years.items()}
            for ticker, years in raw.items()
        }
        self._llm_endpoint    = context.model_config.get("llm_endpoint", "databricks-meta-llama-3-3-70b-instruct")
        self._vs_endpoint     = context.model_config.get("vs_endpoint",  "finsage_vs_endpoint")
        self._vs_index        = context.model_config.get("vs_index",     "main.finsage_gold.filing_chunks_index")
        self._num_results     = int(context.model_config.get("num_results",     5))
        self._sim_threshold   = float(context.model_config.get("similarity_threshold", 0.6))

    @mlflow.trace(name="finsage_agent", span_type="AGENT")
    def predict(self, context, model_input, params=None):
        import mlflow.deployments, json

        # Accept both DataFrame input (Databricks serving) and dict input (notebook testing)
        if hasattr(model_input, "to_dict"):
            records = model_input.to_dict(orient="records")
            messages = records[0].get("messages", [])
        else:
            messages = model_input.get("messages", [])

        if not messages:
            return {"content": "No messages provided.", "messages": []}

        # Build working message list with system prompt prepended
        working_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)

        deploy_client = mlflow.deployments.get_deploy_client("databricks")

        for iteration in range(MAX_ITERATIONS):
            with mlflow.start_span(name=f"llm_call_iter_{iteration}", span_type="LLM") as span:
                span.set_inputs({"messages": working_messages, "iteration": iteration})
                response = deploy_client.predict(
                    endpoint=self._llm_endpoint,
                    inputs={
                        "messages":    working_messages,
                        "tools":       TOOL_SCHEMAS,
                        "tool_choice": "auto",
                        "temperature": 0.1,
                        "max_tokens":  2048,
                    },
                )
                span.set_outputs(response)

            choice      = response["choices"][0]
            finish      = choice.get("finish_reason", "")
            msg         = choice["message"]
            tool_calls  = msg.get("tool_calls") or []

            # No tool calls → final answer
            if not tool_calls or finish == "stop":
                final_content = msg.get("content", "")
                working_messages.append({"role": "assistant", "content": final_content})
                return {"content": final_content, "messages": working_messages[1:]}  # strip system prompt

            # Append assistant message with tool_calls to history
            working_messages.append(msg)

            # Execute each tool call
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                raw_args  = tc["function"].get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}

                with mlflow.start_span(name=f"tool_{tool_name}", span_type="TOOL") as tspan:
                    tspan.set_inputs(args)
                    if tool_name == "search_filings":
                        result = search_filings(
                            query=args.get("query", ""),
                            ticker=args.get("ticker"),
                            section_name=args.get("section_name"),
                            fiscal_year=args.get("fiscal_year"),
                            num_results=args.get("num_results", self._num_results),
                            similarity_threshold=self._sim_threshold,
                        )
                    elif tool_name == "get_company_metrics":
                        result = get_company_metrics(
                            ticker=args.get("ticker", ""),
                            fiscal_year_start=args.get("fiscal_year_start"),
                            fiscal_year_end=args.get("fiscal_year_end"),
                            metrics_cache=self._metrics_cache,
                        )
                    else:
                        result = f"Unknown tool: {tool_name}"
                    tspan.set_outputs({"result": result[:500]})

                working_messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.get("id", tool_name),
                    "content":      result,
                })

        # Max iterations reached — ask the LLM for a best-effort answer
        working_messages.append({
            "role":    "user",
            "content": "Based on the tool results above, provide your best answer now."
        })
        final_response = deploy_client.predict(
            endpoint=self._llm_endpoint,
            inputs={"messages": working_messages, "temperature": 0.1, "max_tokens": 1024},
        )
        final_content = final_response["choices"][0]["message"].get("content", "")
        return {"content": final_content, "messages": working_messages[1:]}


print("[FinSageAgent] Class defined.")

# COMMAND ----------

# ── 8. Local smoke tests ──────────────────────────────────────────────────────

agent = FinSageAgent()

# Simulate load_context manually for notebook testing
class _FakeContext:
    class artifacts:
        pass
    model_config = {
        "llm_endpoint": LLM_ENDPOINT,
        "vs_endpoint":  VS_ENDPOINT,
        "vs_index":     VS_INDEX_NAME,
        "num_results":  NUM_RESULTS,
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }

fake_ctx = _FakeContext()
fake_ctx.artifacts = {"metrics_cache": "/tmp/metrics_cache.json"}

# Save cache to temp file (mimics what MLflow will do)
import json, os
os.makedirs("/tmp", exist_ok=True)
with open("/tmp/metrics_cache.json", "w") as f:
    # Convert int keys to strings for JSON serialisation
    serialisable = {t: {str(fy): m for fy, m in yrs.items()} for t, yrs in METRICS_CACHE.items()}
    json.dump(serialisable, f)

agent.load_context(fake_ctx)

TEST_QUESTIONS = [
    "What was Apple's revenue and net income in fiscal year 2023?",
    "What supply chain risks did NVIDIA disclose in their most recent 10-K?",
    "Compare Microsoft and Alphabet's operating margins in 2023.",
]

for q in TEST_QUESTIONS:
    print(f"\n{'='*70}")
    print(f"Q: {q}")
    result = agent.predict(None, {"messages": [{"role": "user", "content": q}]})
    print(f"A: {result['content'][:800]}")

# COMMAND ----------

# ── 9. MLflow logging & Unity Catalog registration ────────────────────────────

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

input_schema  = Schema([ColSpec("string", "messages")])
output_schema = Schema([ColSpec("string", "content")])
signature     = ModelSignature(inputs=input_schema, outputs=output_schema)

input_example = {
    "messages": [
        {"role": "user", "content": "What was Apple's revenue growth in fiscal year 2023?"}
    ]
}

with mlflow.start_run(run_name=f"finsage_rag_agent_{ENV}") as run:
    mlflow.log_params({
        "llm_endpoint":          LLM_ENDPOINT,
        "vs_index":              VS_INDEX_NAME,
        "num_results":           NUM_RESULTS,
        "similarity_threshold":  SIMILARITY_THRESHOLD,
        "max_iterations":        MAX_ITERATIONS,
    })

    # Save metrics cache as a logged artifact so load_context can access it
    mlflow.log_artifact("/tmp/metrics_cache.json", artifact_path="artifacts")

    model_info = mlflow.pyfunc.log_model(
        artifact_path="finsage_rag_agent",
        python_model=agent,
        artifacts={"metrics_cache": "/tmp/metrics_cache.json"},
        model_config={
            "llm_endpoint":          LLM_ENDPOINT,
            "vs_endpoint":           VS_ENDPOINT,
            "vs_index":              VS_INDEX_NAME,
            "num_results":           NUM_RESULTS,
            "similarity_threshold":  SIMILARITY_THRESHOLD,
        },
        signature=signature,
        input_example=input_example,
        registered_model_name=UC_MODEL_NAME,
        pip_requirements=["databricks-vectorsearch", "mlflow"],
    )

    print(f"[MLflow] Run ID: {run.info.run_id}")
    print(f"[MLflow] Model URI: {model_info.model_uri}")
    print(f"[UC] Registered as: {UC_MODEL_NAME}")

# COMMAND ----------

# ── 10. Deploy to Databricks Model Serving ────────────────────────────────────

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from databricks.sdk.errors import ResourceDoesNotExist

w = WorkspaceClient()

# Get the latest registered model version
client        = mlflow.tracking.MlflowClient()
all_versions  = client.search_model_versions(f"name='{UC_MODEL_NAME}'")
model_version = max(all_versions, key=lambda v: int(v.version)).version
print(f"[DEPLOY] Deploying {UC_MODEL_NAME} version {model_version} → {AGENT_ENDPOINT}")

served_entity = ServedEntityInput(
    entity_name=UC_MODEL_NAME,
    entity_version=str(model_version),
    workload_size="Small",
    scale_to_zero_enabled=True,
)

endpoint_config = EndpointCoreConfigInput(name=AGENT_ENDPOINT, served_entities=[served_entity])

try:
    existing = w.serving_endpoints.get(AGENT_ENDPOINT)
    print(f"[DEPLOY] Endpoint exists (state={existing.state.ready}). Updating config...")
    w.serving_endpoints.update_config(name=AGENT_ENDPOINT, served_entities=[served_entity])
    print("[DEPLOY] Update submitted.")
except ResourceDoesNotExist:
    print("[DEPLOY] Creating new endpoint...")
    w.serving_endpoints.create(name=AGENT_ENDPOINT, config=endpoint_config)
    print("[DEPLOY] Creation submitted.")
    print("[DEPLOY] Creation submitted.")

print(f"[DEPLOY] Monitor at: https://dbc-f33010ed-00fc.cloud.databricks.com/ml/endpoints/{AGENT_ENDPOINT}")

# COMMAND ----------

# ── 11. Wait for endpoint + live test ─────────────────────────────────────────

import time

print(f"Waiting for endpoint '{AGENT_ENDPOINT}' to reach READY state...")
timeout, poll = 20 * 60, 20
start = time.time()

while True:
    if time.time() - start > timeout:
        print("Timeout waiting for endpoint. Check the Serving UI manually.")
        break
    try:
        ep    = w.serving_endpoints.get(AGENT_ENDPOINT)
        state = str(ep.state.ready) if ep.state else "UNKNOWN"
        print(f"  Endpoint state: {state}")
        if state == "EndpointStateReady.READY":
            print("Endpoint is READY.")
            break
        if "FAILED" in state.upper() or "NOT_READY" in state.upper() and time.time() - start < 30:
            pass  # NOT_READY is normal during startup — keep polling
        if "FAILED" in state.upper():
            print(f"Endpoint failed: {ep.state}")
            break
    except Exception as e:
        print(f"  Polling error: {e}")
    time.sleep(poll)

# Live test via SDK (avoids mlflow deploy client resolving wrong workspace URL)
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

live_test_questions = [
    "What was NVIDIA's revenue and net income in fiscal year 2024?",
    "What risks did Tesla disclose about autonomous driving in their 10-K?",
]

for q in live_test_questions:
    print(f"\n{'='*70}\nQ: {q}")
    try:
        resp = w.serving_endpoints.query(
            name=AGENT_ENDPOINT,
            messages=[ChatMessage(role=ChatMessageRole.USER, content=q)],
        )
        answer = resp.choices[0].message.content if resp.choices else str(resp)
        print(f"A: {answer[:800]}")
    except Exception as e:
        print(f"Live test error: {e}")
