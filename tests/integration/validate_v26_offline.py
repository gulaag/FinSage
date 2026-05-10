"""Offline validation for the v27 SYSTEM_PROMPT — runs BEFORE any deploy.

Industry-grade gate: the v27 prompt is only allowed to ship if it passes
this harness. The harness exercises the v22 pyfunc model loaded from UC,
swaps its SYSTEM_PROMPT in-process to the candidate v27 string, and runs a
hand-picked panel of questions covering each diagnosed v25 / v26 failure
mode plus a control set of v22-passing baselines.

v27 is single-LLM: there is NO synthesizer pass. The deployed predict()
returns the LLM's first response verbatim (after citation enforcement).
This harness mirrors that contract — no synthesizer is simulated here.

Acceptance criteria (all must hold):
  - Regression panel: every question that v22 passed and v25 failed must
    extract the correct ground-truth number within +/-1 % relative tolerance.
  - Baseline panel: the v22-passing controls must remain correct.
  - Refusal panel: refusal questions must produce a tight refusal whose
    response matches the expected reasoning tokens.
  - Verbose gate: every non-refusal answer must be >=200 words and
    >=3 paragraphs before the [Source: ...] line. v27's whole reason for
    existing is to make verbose answers come from the single LLM call —
    if this gate fails, single-LLM verbose is not achievable with
    Llama-3.3-70B function-calling and the user must decide.
  - No tool-call leakage: no answer may contain "<function=" or
    "Let me retrieve" / "let me check" stalling text.

If ANY criterion fails, the script exits non-zero and the v27 candidate
is rejected. We iterate the prompt design — we do NOT redeploy.

Run:
  DATABRICKS_CONFIG_PROFILE=DEFAULT \\
    .venv310/bin/python tests/integration/validate_v26_offline.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# v27 SYSTEM_PROMPT — sourced verbatim from the notebook so this harness
# never drifts from what we will deploy. We extract it programmatically.
# ─────────────────────────────────────────────────────────────────────────────

NOTEBOOK_PATH = Path(__file__).resolve().parents[2] / "databricks/notebooks/06_rag_agent.py"


def _extract_v27_prompt() -> str:
    text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    m = re.search(r'SYSTEM_PROMPT = """\\\n(.*?)\n"""\s*\n\n\nclass FinSageAgent', text, re.DOTALL)
    if not m:
        raise RuntimeError(f"Could not extract SYSTEM_PROMPT from {NOTEBOOK_PATH}")
    return m.group(1)


V27_PROMPT = _extract_v27_prompt()
print(f"[v27] SYSTEM_PROMPT loaded: {len(V27_PROMPT):,} chars")


# ─────────────────────────────────────────────────────────────────────────────
# Test panel — every question is grounded in the eval ground truth so the
# expected numbers are SEC-verified, not made up here.
# ─────────────────────────────────────────────────────────────────────────────

REGRESSION_PANEL = [
    # Hallucinated context (v25 invented $43.79B for KO FY22; truth is $43.00B)
    dict(
        qid="A004",
        question="What was Coca-Cola's revenue in fiscal year 2022?",
        expected_value=43_004_000_000.0,
        kind="numerical",
    ),
    # False refusal (v25 said "NET not in corpus"; NET IS in corpus)
    dict(
        qid="A015",
        question="What were Cloudflare's total assets in fiscal year 2022?",
        expected_value=2_587_908_000.0,
        kind="numerical",
    ),
    # Tool-call leakage (v25 wrote <function=...> as text)
    dict(
        qid="D002",
        question="Which had higher total assets in fiscal year 2025: JPMorgan Chase or Bank of America?",
        expected_value=4_424_900_000_000.0,  # JPM, the higher
        kind="numerical",
    ),
    dict(
        qid="D006",
        question="Which had higher total liabilities in fiscal year 2021: Microsoft or Alphabet?",
        expected_value=191_791_000_000.0,  # MSFT, the higher
        kind="numerical",
    ),
]

BASELINE_PANEL = [
    # v22 already passed these — v27 must not regress
    dict(
        qid="BASE-AAPL-FY24",
        question="What was Apple's revenue in fiscal year 2024?",
        expected_value=391_035_000_000.0,
        kind="numerical",
    ),
    dict(
        qid="BASE-MSFT-FY24",
        question="What was Microsoft's revenue in fiscal year 2024?",
        expected_value=245_122_000_000.0,
        kind="numerical",
    ),
    dict(
        qid="BASE-NVDA-FY24",
        question="What was NVIDIA's net income in fiscal year 2024?",
        expected_value=29_760_000_000.0,
        kind="numerical",
    ),
]

REFUSAL_PANEL = [
    # Future period
    dict(
        qid="REF-AAPL-FY30",
        question="What was Apple's revenue in fiscal year 2030?",
        expected_tokens=("future", "hasn't", "has not", "not yet"),
        kind="refusal",
    ),
    # Out-of-corpus ticker
    dict(
        qid="REF-IBM",
        question="What was IBM's revenue in fiscal year 2024?",
        expected_tokens=("not in", "corpus", "30 ", "thirty", "not tracked"),
        kind="refusal",
    ),
]

ALL_PANELS = [
    ("REGRESSION", REGRESSION_PANEL),
    ("BASELINE",   BASELINE_PANEL),
    ("REFUSAL",    REFUSAL_PANEL),
]


# ─────────────────────────────────────────────────────────────────────────────
# Number extraction — copy of scorers.py logic, kept inline so this harness
# is a self-contained validation gate that does not depend on the eval suite.
# ─────────────────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9])([-+]?\$?\s*\d[\d,]*(?:\.\d+)?)\s*(billion|million|thousand|trillion|bn|mn|tn|k|b|m|t|%|x)?",
    re.IGNORECASE,
)
_UNIT = {
    "trillion": 1e12, "tn": 1e12, "t": 1e12,
    "billion":  1e9,  "bn": 1e9,  "b": 1e9,
    "million":  1e6,  "mn": 1e6,  "m": 1e6,
    "thousand": 1e3,  "k":  1e3,
}


def extract_numbers(text: str) -> list[float]:
    out: list[float] = []
    for m in _NUM_RE.finditer(text or ""):
        token = m.group(1).replace("$", "").replace(",", "").strip()
        try:
            v = float(token)
        except ValueError:
            continue
        u = (m.group(2) or "").lower()
        if u in _UNIT:
            v *= _UNIT[u]
        out.append(v)
    return out


def best_match(candidates: list[float], target: float) -> Optional[float]:
    if not candidates:
        return None
    filtered = [v for v in candidates if not (1900 <= abs(v) <= 2100)]
    pool = filtered if filtered else candidates
    if target == 0:
        return pool[0]
    return min(pool, key=lambda x: abs(x - target) / abs(target))


# ─────────────────────────────────────────────────────────────────────────────
# Tool-call leakage / stalling detector
# ─────────────────────────────────────────────────────────────────────────────

LEAKAGE_PATTERNS = [
    re.compile(r"<function\s*=", re.IGNORECASE),
    re.compile(r"\blet me (?:retrieve|check|look|fetch|pull|see)\b", re.IGNORECASE),
    re.compile(r"\bplease wait\b", re.IGNORECASE),
    re.compile(r"\bI'll (?:now|first) (?:check|retrieve|fetch|look)\b", re.IGNORECASE),
]


def has_leakage(text: str) -> Optional[str]:
    for p in LEAKAGE_PATTERNS:
        m = p.search(text or "")
        if m:
            return m.group(0)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Load the deployed agent and patch its SYSTEM_PROMPT in-process
# ─────────────────────────────────────────────────────────────────────────────

def load_agent_with_v27_prompt():
    """Load v22 from UC and inject the v27 SYSTEM_PROMPT.

    Cloudpickle restores the FinSageAgent class into the loader's __main__
    module. The predict() method body looks up `SYSTEM_PROMPT` via that
    module's globals, so we set it on __main__ directly. We also walk every
    candidate module that exposes a `predict_module_globals` reference, in
    case the cloudpickle encoding routed globals elsewhere.
    """
    import mlflow

    mlflow.set_registry_uri("databricks-uc")
    print("[load] Loading main.finsage_gold.finsage_rag_agent/22 ...")
    t0 = time.time()
    pyfunc_model = mlflow.pyfunc.load_model("models:/main.finsage_gold.finsage_rag_agent/22")
    agent = pyfunc_model.unwrap_python_model()
    print(f"[load] v22 agent loaded in {time.time() - t0:.1f}s")

    # The function's __globals__ is the authoritative namespace it reads
    # SYSTEM_PROMPT from. Patch THAT — survives whatever module routing
    # cloudpickle did.
    predict_globals = agent.predict.__func__.__globals__
    existing = predict_globals.get("SYSTEM_PROMPT")
    if existing:
        print(f"[load] Patching SYSTEM_PROMPT in predict.__globals__ "
              f"(was {len(existing):,} chars -> now {len(V27_PROMPT):,})")
    else:
        print(f"[load] Injecting SYSTEM_PROMPT into predict.__globals__ "
              f"({len(V27_PROMPT):,} chars; no prior binding)")
    predict_globals["SYSTEM_PROMPT"] = V27_PROMPT

    # Also mirror onto __main__ so any indirect references resolve too.
    import __main__ as _main
    _main.SYSTEM_PROMPT = V27_PROMPT

    # v27 also bumps the LLM call temperature (0.1 -> 0.3) and max_tokens
    # (1024 -> 2000) inside predict(). Those values are baked into the v22
    # cloudpickle bytecode, so to test the FULL v27 contract we monkey-patch
    # the deploy client to override those two params on the tool-routing
    # call (identified by the presence of the `tools` field in inputs).
    # Other Databricks deploy_client.predict() calls in the agent path
    # (e.g. embedding lookups) are passed through unchanged.
    import mlflow.deployments as _dep

    _real_get = _dep.get_deploy_client

    def _patched_get_deploy_client(*args, **kwargs):
        client = _real_get(*args, **kwargs)
        _orig_predict = client.predict

        # The verbose-elaboration reminder injected right before the LLM
        # generates its final response. Same content as the notebook's
        # predict() turn-injection so this harness mirrors deployed behavior.
        VERBOSE_REMINDER = (
            "Now write the comprehensive 4-6 paragraph senior "
            "equity-analyst briefing using ONLY the values that "
            "appear in the tool output above. Use markdown "
            "section headers and bold key figures. Lead with the "
            "headline figure in the first sentence, then expand "
            "into prior-year comparison, profitability, cash "
            "generation, balance sheet, and a closing implication "
            "paragraph (only those that the tool output supports). "
            "Do NOT introduce numbers that are not in the tool "
            "output. End with the [Source: ...] citation line(s)."
        )

        def _wrapped_predict(*p_args, **p_kwargs):
            # Caller (FinSageAgent.predict) invokes as
            #   deploy_client.predict(endpoint=..., inputs=...)
            # but the underlying DatabricksDeploymentClient.predict signature
            # is (deployment_name=None, inputs=None, endpoint=None), so we
            # MUST forward by keyword to preserve the endpoint binding.
            inputs = p_kwargs.get("inputs")
            if inputs is not None:
                patched = dict(inputs)
                if "tools" in patched:
                    patched["temperature"] = 0.3
                    patched["max_tokens"] = 2000
                # Mirror the notebook's turn-injection: when the messages
                # array ends with a tool-role message, append the verbose
                # briefing reminder so it lands as the most recent message
                # the LLM sees. v22's pickled predict() does NOT do this
                # injection, so the harness must inject it to test what we
                # will actually deploy.
                msgs = patched.get("messages") or []
                if msgs and isinstance(msgs[-1], dict) and msgs[-1].get("role") == "tool":
                    msgs = list(msgs) + [{"role": "user", "content": VERBOSE_REMINDER}]
                    patched["messages"] = msgs
                p_kwargs["inputs"] = patched
            return _orig_predict(*p_args, **p_kwargs)

        client.predict = _wrapped_predict
        return client

    _dep.get_deploy_client = _patched_get_deploy_client
    print("[load] Monkey-patched deploy_client.predict to inject v27's "
          "temperature=0.3, max_tokens=2000 on tool-routing calls.")

    # v27 swaps the final-LLM endpoint from databricks-meta-llama-3-3-70b-instruct
    # to databricks-claude-sonnet-4-6. Llama in tool-calling mode is RLHF-tuned
    # toward terse final responses; prompt-only verbose attempts (v24, v26
    # attempt 1) confirmed empirically that no prompt overrides this bias.
    # Claude Sonnet 4.6 follows length instructions reliably while preserving
    # the OpenAI-compatible tool-calling schema we already use. Sec-parser
    # analog: replace the structurally-wrong component, do not patch around it.
    new_llm = "databricks-claude-sonnet-4-6"
    print(f"[load] Re-pointing agent._llm_endpoint -> {new_llm} (was "
          f"{getattr(agent, '_llm_endpoint', '<unset>')})")
    agent._llm_endpoint = new_llm
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Main validation loop — single LLM, no synthesizer pass
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(agent, q: dict) -> dict:
    """Run one question through the patched v22+v27-prompt agent and
    measure the response directly. v27 has no synthesizer pass — what
    the LLM returns is what ships."""
    t0 = time.time()
    try:
        result = agent.predict(None, {"messages": [{"role": "user", "content": q["question"]}]})
    except Exception as exc:
        import traceback
        return dict(
            qid=q["qid"], elapsed=time.time() - t0,
            error=f"{type(exc).__name__}: {exc}",
            response="", passed=False,
            rationale=f"agent.predict raised: {type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}",
        )

    response = result.get("content", "") if isinstance(result, dict) else str(result)
    elapsed = time.time() - t0

    leakage = has_leakage(response)
    if leakage:
        return dict(
            qid=q["qid"], elapsed=elapsed, response=response[:400],
            passed=False, rationale=f"tool-call leakage detected: {leakage!r}",
        )

    if q["kind"] == "numerical":
        cands = extract_numbers(response)
        pred = best_match(cands, q["expected_value"])
        if pred is None or q["expected_value"] == 0:
            return dict(qid=q["qid"], elapsed=elapsed, response=response[:400],
                        passed=False, rationale="no numeric value extractable")
        rel = abs(pred - q["expected_value"]) / abs(q["expected_value"])
        numerical_ok = rel <= 0.01
        # Verbose gate — non-refusal answers MUST be >=200 words and
        # >=3 paragraphs before the [Source: ...] line. v27's reason for
        # existing is to make verbose come from the single LLM call;
        # this gate is the empirical check on whether that worked.
        body = response.split("[Source:")[0].strip()
        word_count = len(body.split())
        n_paragraphs = len([p for p in body.split("\n\n") if p.strip()])
        verbose_ok = word_count >= 200 and n_paragraphs >= 3
        passed = numerical_ok and verbose_ok
        rationale = (
            f"pred={pred:.4g} target={q['expected_value']:.4g} rel_err={rel*100:.2f}% "
            f"| words={word_count} paragraphs={n_paragraphs} "
            f"verbose_ok={verbose_ok}"
        )
        return dict(qid=q["qid"], elapsed=elapsed, response=response[:400],
                    passed=passed, rationale=rationale)

    if q["kind"] == "refusal":
        low = response.lower()
        refusal_phrases = (
            "cannot", "can't", "couldn't", "unable", "decline",
            "isn't", "is not", "hasn't", "has not", "not in",
            "future", "out of scope", "not currently",
        )
        refused = any(p in low for p in refusal_phrases)
        cited = any(t.lower() in low for t in q["expected_tokens"])
        passed = refused and cited
        return dict(
            qid=q["qid"], elapsed=elapsed, response=response[:400], passed=passed,
            rationale=f"refused={refused} cited_reason={cited} (tokens={q['expected_tokens']})",
        )

    return dict(qid=q["qid"], elapsed=elapsed, response=response[:400],
                passed=False, rationale=f"unknown kind={q['kind']!r}")


def main() -> int:
    agent = load_agent_with_v27_prompt()

    all_results: list[dict] = []
    for panel_name, panel in ALL_PANELS:
        print(f"\n{'=' * 72}\n  PANEL: {panel_name}  ({len(panel)} questions)\n{'=' * 72}")
        for q in panel:
            print(f"\n| {q['qid']}: {q['question']}")
            r = evaluate(agent, q)
            r["panel"] = panel_name
            all_results.append(r)
            tag = "PASS" if r["passed"] else "FAIL"
            print(f"|   [{tag}]  ({r['elapsed']:.1f}s)  {r['rationale']}")
            if not r["passed"]:
                print(f"|   response: {r['response']}")

    # Acceptance gate — math correctness vs verbose presentation are tracked
    # separately so a verbose-only failure can be reported without falsely
    # condemning the v27 prompt's math (which it inherits from v22).
    print(f"\n{'=' * 72}\n  ACCEPTANCE GATE\n{'=' * 72}")

    # Per-panel pass/fail
    panel_ok: dict[str, bool] = {}
    for panel_name, _ in ALL_PANELS:
        rows = [r for r in all_results if r["panel"] == panel_name]
        passed = sum(1 for r in rows if r["passed"])
        total = len(rows)
        ok = passed == total
        panel_ok[panel_name] = ok
        symbol = "OK" if ok else "FAIL"
        print(f"  [{symbol}] {panel_name:<10s}  {passed}/{total}")

    # Math vs verbose breakdown for non-refusal rows (so we can tell the
    # user whether a failure is due to wrong numbers or due to a terse LLM).
    numeric_rows = [r for r in all_results if r["panel"] in ("REGRESSION", "BASELINE")]
    math_pass = sum(1 for r in numeric_rows if "rel_err" in r.get("rationale", "")
                    and "verbose_ok=" in r["rationale"]
                    and float(r["rationale"].split("rel_err=")[1].split("%")[0]) <= 1.0)
    verbose_pass = sum(1 for r in numeric_rows if "verbose_ok=True" in r.get("rationale", ""))
    total_numeric = len(numeric_rows)
    print(f"\n  math:    {math_pass}/{total_numeric} numeric rows within +/-1%")
    print(f"  verbose: {verbose_pass}/{total_numeric} non-refusal rows >=200 words and >=3 paragraphs")

    overall = all(panel_ok.values())
    if overall:
        print("\n[GATE] PASSED - v27 candidate is approved for deploy.\n")
        return 0

    if math_pass == total_numeric and verbose_pass < total_numeric:
        print("\n[GATE] FAILED - math is correct but verbose-via-prompt did not")
        print("hold. Single-LLM verbose appears unachievable with Llama-3.3-70B")
        print("function-calling on this prompt formulation. Report to user")
        print("before iterating - do NOT add a synthesizer pass.\n")
        return 2

    print("\n[GATE] FAILED - v27 candidate is rejected. Diagnose specific")
    print("failing rows before iterating the prompt design.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
