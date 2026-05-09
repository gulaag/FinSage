"""Offline validation for the v26 SYSTEM_PROMPT — runs BEFORE any deploy.

Industry-grade gate: the v26 prompt is only allowed to ship if it passes
this harness. The harness exercises the v22 pyfunc model loaded from UC,
swaps its SYSTEM_PROMPT in-process to the candidate v26 string, and runs a
hand-picked panel of questions covering each diagnosed v25 failure mode
plus a control set of v22-passing baselines.

Acceptance criteria (all must hold):
  • Regression panel: every question that v22 passed and v25 failed must
    extract the correct ground-truth number within ±1 % relative tolerance.
  • Baseline panel: the v22-passing controls must remain correct.
  • Refusal panel: refusal questions must produce a tight refusal whose
    response matches the expected reasoning tokens.
  • No tool-call leakage: no answer may contain "<function=" or
    "Let me retrieve" / "let me check" stalling text.

If ANY criterion fails, the script exits non-zero and the v26 candidate
is rejected. We iterate the prompt design — we do NOT redeploy.

Run:
  DATABRICKS_CONFIG_PROFILE=DEFAULT \\
    .venv/bin/python tests/integration/validate_v26_offline.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# v26 SYSTEM_PROMPT — sourced verbatim from the notebook so this harness
# never drifts from what we will deploy. We extract it programmatically.
# ─────────────────────────────────────────────────────────────────────────────

NOTEBOOK_PATH = Path(__file__).resolve().parents[2] / "databricks/notebooks/06_rag_agent.py"


def _extract_v26_prompt() -> str:
    text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    m = re.search(r'SYSTEM_PROMPT = """\\\n(.*?)\n"""\s*\n\n\nclass FinSageAgent', text, re.DOTALL)
    if not m:
        raise RuntimeError(f"Could not extract SYSTEM_PROMPT from {NOTEBOOK_PATH}")
    return m.group(1)


V26_PROMPT = _extract_v26_prompt()
print(f"[v26] SYSTEM_PROMPT loaded: {len(V26_PROMPT):,} chars")


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
    # v22 already passed these — v26 must not regress
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

def load_agent_with_v26_prompt():
    """Load v22 from UC and inject the v26 SYSTEM_PROMPT.

    Cloudpickle restores the FinSageAgent class into the loader's __main__
    module. The predict() method body looks up `SYSTEM_PROMPT` via that
    module's globals, so we set it on __main__ directly. We also walk every
    candidate module that exposes a `predict_module_globals` reference, in
    case the cloudpickle encoding routed globals elsewhere.
    """
    import mlflow

    mlflow.set_registry_uri("databricks-uc")
    print("[load] Loading main.finsage_gold.finsage_rag_agent/22 …")
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
              f"(was {len(existing):,} chars → now {len(V26_PROMPT):,})")
    else:
        print(f"[load] Injecting SYSTEM_PROMPT into predict.__globals__ "
              f"({len(V26_PROMPT):,} chars; no prior binding)")
    predict_globals["SYSTEM_PROMPT"] = V26_PROMPT

    # Also mirror onto __main__ so any indirect references resolve too.
    import __main__ as _main
    _main.SYSTEM_PROMPT = V26_PROMPT
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Main validation loop
# ─────────────────────────────────────────────────────────────────────────────

def _run_synthesizer(working_messages: list, terse_content: str, llm_endpoint: str) -> str:
    """Reproduce v26's synthesizer pass: take the terse first-pass content
    and ask the LLM to expand it into a 4-6 paragraph briefing without
    introducing new numbers. Mirrors the exact prompt added to predict()
    in notebook 06 so the validation harness exercises the SAME contract
    the deployed agent will."""
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    msgs = list(working_messages) + [
        {"role": "assistant", "content": terse_content},
        {
            "role": "user",
            "content": (
                "Now rewrite that as a comprehensive senior-analyst briefing "
                "of 4 to 6 short paragraphs (minimum 250 words before the "
                "[Source: ...] line). Use ONLY the numerical values that "
                "appear in the tool outputs above and in your previous "
                "answer — do NOT introduce any new numbers, prior-year "
                "figures, segment data, or facts that the tools did not "
                "supply. Keep every [Source: ...] line intact at the end. "
                "Follow the EXACT-FIRST formatting rule for any dollar "
                "figure on first mention. If the previous answer was a "
                "refusal, do NOT expand it — return it unchanged."
            ),
        },
    ]
    resp = deploy_client.predict(
        endpoint=llm_endpoint,
        inputs={"messages": msgs, "temperature": 0.4, "max_tokens": 1500},
    )
    return resp["choices"][0]["message"].get("content", "") or terse_content


def evaluate(agent, q: dict) -> dict:
    """Run one question through the patched v22+v26-prompt agent, then
    apply the v26 synthesizer pass to the result so we test the EXACT
    v26 deployment behavior end-to-end."""
    t0 = time.time()
    try:
        result = agent.predict(None, {"messages": [{"role": "user", "content": q["question"]}]})
    except Exception as exc:
        return dict(
            qid=q["qid"], elapsed=time.time() - t0,
            error=f"{type(exc).__name__}: {exc}",
            response="", passed=False, rationale="agent.predict raised",
        )

    terse = result.get("content", "") if isinstance(result, dict) else str(result)
    messages_history = result.get("messages", []) if isinstance(result, dict) else []

    # Run the synthesizer pass for non-refusal answers. Refusals stay tight.
    is_refusal = q["kind"] == "refusal"
    used_a_tool = any(
        isinstance(m, dict) and m.get("role") == "tool"
        for m in messages_history
    )
    if used_a_tool and not is_refusal:
        try:
            response = _run_synthesizer(
                working_messages=messages_history,
                terse_content=terse,
                llm_endpoint=getattr(agent, "_llm_endpoint", "databricks-meta-llama-3-3-70b-instruct"),
            )
        except Exception as exc:
            return dict(
                qid=q["qid"], elapsed=time.time() - t0,
                error=f"synthesizer raised: {type(exc).__name__}: {exc}",
                response=terse[:400], passed=False,
                rationale="synthesizer pass failed",
            )
    else:
        response = terse
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
        # Word-count gate — non-refusal answers MUST be verbose (≥200 words
        # before the [Source: ...] line). Catches the regression we hit
        # with the v26-without-synthesizer attempt where the LLM answered
        # the math correctly but in a single sentence.
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
    agent = load_agent_with_v26_prompt()

    all_results: list[dict] = []
    for panel_name, panel in ALL_PANELS:
        print(f"\n{'═' * 72}\n  PANEL: {panel_name}  ({len(panel)} questions)\n{'═' * 72}")
        for q in panel:
            print(f"\n┃ {q['qid']}: {q['question']}")
            r = evaluate(agent, q)
            r["panel"] = panel_name
            all_results.append(r)
            tag = "PASS" if r["passed"] else "FAIL"
            print(f"┃   [{tag}]  ({r['elapsed']:.1f}s)  {r['rationale']}")
            if not r["passed"]:
                print(f"┃   response: {r['response']}")

    # Acceptance gate
    print(f"\n{'═' * 72}\n  ACCEPTANCE GATE\n{'═' * 72}")
    overall_pass = True
    for panel_name, _ in ALL_PANELS:
        rows = [r for r in all_results if r["panel"] == panel_name]
        passed = sum(1 for r in rows if r["passed"])
        total = len(rows)
        ok = passed == total
        overall_pass = overall_pass and ok
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {panel_name:<10s}  {passed}/{total}")

    if overall_pass:
        print("\n[GATE] PASSED — v26 candidate is approved for deploy.\n")
        return 0
    else:
        print("\n[GATE] FAILED — v26 candidate is rejected. Iterate the prompt design.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
