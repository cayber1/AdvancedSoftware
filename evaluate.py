"""
ContextGuard — Experimental Evaluation (§6 Experimental Study)

Proposal §5 Datasets:
  - HotpotQA (distractor setting) — Hugging Face veya built-in fallback
  - Natural Questions (NQ-open)   — Hugging Face veya built-in fallback

Proposal §6 Conditions:
  [A] Baseline RAG pipeline       — no ranking, no validation
  [B] Single-agent LLM            — no context at all
  [C] Multi-agent without MCP     — retrieval + reasoning, no governance
  [D] Full ContextGuard system    — MCP + multi-agent + validation

Metrics: HR, CIS, CUE, RS, Overall Score
"""

import json
import os
import argparse

from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from contextguard.mcp_governance import MCPGovernance
from contextguard.context_store import ContextStore
from contextguard.agents import (
    ContextRetrievalAgent,
    ReasoningAgent,
    GroundingValidatorAgent,
    AdversarialTesterAgent,
)
from contextguard.metrics import evaluate_response
from contextguard.data_loader import load_dataset, describe, DatasetSource

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"
client       = Groq(api_key=GROQ_API_KEY)


# ── Helper: Groq chat ─────────────────────────────────────────────────────────
def _chat(system: str, user: str, temperature: float = 0.3) -> str:
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()


# ── Condition A: Baseline RAG ─────────────────────────────────────────────────
def baseline_rag(query: str, docs: list[str]) -> str:
    context = "\n".join(docs)
    return _chat(
        "Answer the question using the provided context.",
        f"Context:\n{context}\n\nQuestion: {query}",
    )


# ── Condition B: Single-agent LLM ─────────────────────────────────────────────
def single_agent_llm(query: str) -> str:
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": query}],
        temperature=0.5,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()


# ── Condition C: Multi-agent without MCP ──────────────────────────────────────
def multiagent_no_mcp(query: str, docs: list[str]) -> str:
    vectorizer = TfidfVectorizer()
    matrix     = vectorizer.fit_transform([query] + docs)
    scores     = cosine_similarity(matrix[0:1], matrix[1:])[0]
    top_docs   = [docs[i] for i in scores.argsort()[::-1][:3]]
    context    = "\n".join(top_docs)
    return _chat(
        "Answer based strictly on the context. Do not use external knowledge.",
        f"Context:\n{context}\n\nQuestion: {query}",
    )


# ── Condition D: Full ContextGuard ────────────────────────────────────────────
def full_contextguard(query: str, docs: list[str]) -> dict:
    mcp         = MCPGovernance()
    store       = ContextStore()
    retrieval   = ContextRetrievalAgent(mcp, store)
    reasoning   = ReasoningAgent(mcp, store)
    validator   = GroundingValidatorAgent(mcp, store)
    adversarial = AdversarialTesterAgent(mcp, store)

    context, vid = retrieval.retrieve_and_rank(query, docs)
    if not context:
        return {"answer": "", "metrics": {}, "mcp_denied": mcp.get_denied_events()}

    answer     = reasoning.generate_answer(query, context, version_id=vid)
    validation = validator.validate(query, answer, context)
    robustness = adversarial.test_robustness(query, context, reasoning, num_trials=2)
    metrics    = evaluate_response(answer, context, validation, robustness)

    return {
        "answer":      answer,
        "metrics":     metrics,
        "validation":  validation,
        "robustness":  robustness,
        "mcp_log":     mcp.get_execution_log(),
        "mcp_denied":  mcp.get_denied_events(),
        "store":       store.history(),
    }


# ── Metric helpers ────────────────────────────────────────────────────────────
def keyword_accuracy(answer: str, keywords: list[str]) -> float:
    a = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in a)
    return round(hits / len(keywords), 4) if keywords else 0.0


# ── Main Benchmark ────────────────────────────────────────────────────────────
def run_benchmark(
    source: DatasetSource = "both",
    max_samples: int = 10,
    prefer_real: bool = True,
    output_file: str = "benchmark_results.json",
):
    print("\n" + "═" * 72)
    print("  ContextGuard Benchmark — 4-Condition Experimental Evaluation")
    print("  Proposal §5: HotpotQA + Natural Questions")
    print("  Proposal §6: Baseline RAG / Single-Agent / Multi-Agent / ContextGuard")
    print("═" * 72)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("\n  Loading datasets…")
    eval_items = load_dataset(source=source, max_samples=max_samples, prefer_real=prefer_real)
    print(describe(eval_items))
    print()

    results = []

    for idx, item in enumerate(eval_items, 1):
        query    = item["query"]
        docs     = item["docs"]
        keywords = item["keywords"]
        src_tag  = item["source"].upper()

        print(f"\n[{idx}/{len(eval_items)}] [{src_tag}] {query[:75]}")
        print("─" * 72)

        # A — Baseline RAG
        ans_a = baseline_rag(query, docs)
        acc_a = keyword_accuracy(ans_a, keywords)
        print(f"  [A] Baseline RAG          acc={acc_a:.0%}")

        # B — Single-agent LLM
        ans_b = single_agent_llm(query)
        acc_b = keyword_accuracy(ans_b, keywords)
        print(f"  [B] Single-Agent LLM      acc={acc_b:.0%}")

        # C — Multi-agent, no MCP
        ans_c = multiagent_no_mcp(query, docs)
        acc_c = keyword_accuracy(ans_c, keywords)
        print(f"  [C] Multi-Agent / no MCP  acc={acc_c:.0%}")

        # D — Full ContextGuard
        res_d  = full_contextguard(query, docs)
        ans_d  = res_d.get("answer", "")
        acc_d  = keyword_accuracy(ans_d, keywords)
        met_d  = res_d.get("metrics", {})
        denied = len(res_d.get("mcp_denied", []))

        print(
            f"  [D] ContextGuard (full)   acc={acc_d:.0%}  "
            f"HR={met_d.get('hallucination_rate', 0):.0%}  "
            f"CIS={met_d.get('cis_score', 0):.0%}  "
            f"CUE={met_d.get('cue_score', 0):.0%}  "
            f"RS={met_d.get('robustness_score', 0):.0%}  "
            f"Overall={met_d.get('overall_score', 0):.0%}  "
            f"MCP_denied={denied}"
        )

        for v in met_d.get("constraint_violations", []):
            print(f"    ⚠ {v}")

        results.append({
            "idx":    idx,
            "source": item["source"],
            "query":  query,
            "answer_gt": item["answer"],
            "A_baseline_rag":      {"answer": ans_a, "accuracy": acc_a},
            "B_single_agent":      {"answer": ans_b, "accuracy": acc_b},
            "C_multiagent_no_mcp": {"answer": ans_c, "accuracy": acc_c},
            "D_contextguard": {
                "answer":   ans_d,
                "accuracy": acc_d,
                **met_d,
                "mcp_denied_events": denied,
            },
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  SUMMARY — Averages across all queries")
    print("─" * 72)

    conditions = [
        ("A_baseline_rag",      "Baseline RAG"),
        ("B_single_agent",      "Single-Agent LLM"),
        ("C_multiagent_no_mcp", "Multi-Agent / no MCP"),
        ("D_contextguard",      "ContextGuard (full)"),
    ]
    for key, label in conditions:
        avg_acc = sum(r[key]["accuracy"] for r in results) / len(results)
        row = f"  {label:<26} acc={avg_acc:.1%}"
        if key == "D_contextguard":
            for metric in ["hallucination_rate", "cis_score", "cue_score", "robustness_score", "overall_score"]:
                avg = sum(r[key].get(metric, 0) for r in results) / len(results)
                short = {"hallucination_rate": "HR", "cis_score": "CIS",
                         "cue_score": "CUE", "robustness_score": "RS",
                         "overall_score": "OV"}[metric]
                row += f"  {short}={avg:.1%}"
        print(row)

    # ── Per-source breakdown ──────────────────────────────────────────────────
    sources = list({r["source"] for r in results})
    if len(sources) > 1:
        print("\n  Per-source breakdown:")
        for src in sources:
            src_items = [r for r in results if r["source"] == src]
            avg_d = sum(r["D_contextguard"]["accuracy"] for r in src_items) / len(src_items)
            avg_hr = sum(r["D_contextguard"].get("hallucination_rate", 0) for r in src_items) / len(src_items)
            print(f"    [{src.upper():<10}] n={len(src_items)}  acc={avg_d:.1%}  HR={avg_hr:.1%}")

    # ── Hypothesis ────────────────────────────────────────────────────────────
    avg_hr_d = sum(r["D_contextguard"].get("hallucination_rate", 1) for r in results) / len(results)
    print("\n  Hypothesis (proposal §6):")
    if avg_hr_d < 0.30:
        print(f"    ✓ SUPPORTED — ContextGuard avg HR = {avg_hr_d:.1%} < τ (30%)")
    else:
        print(f"    ✗ NOT MET   — ContextGuard avg HR = {avg_hr_d:.1%} ≥ τ (30%)")

    print("═" * 72)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Results saved to {output_file}\n")

    return results


# ── EVAL_DATASET alias (for app.py SSE benchmark endpoint) ───────────────────
# app.py'nin /api/benchmark endpoint'i bu listeyi import ediyor.
# Gerçek dataset yükleme olmadan hızlı SSE akışı için fallback listesi.
from contextguard.data_loader import _FALLBACK_HOTPOTQA, _FALLBACK_NQ

EVAL_DATASET = (_FALLBACK_HOTPOTQA + _FALLBACK_NQ)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ContextGuard Benchmark")
    parser.add_argument("--source",      default="both",  choices=["hotpotqa", "nq", "both"],
                        help="Dataset source (default: both)")
    parser.add_argument("--samples",     default=10, type=int,
                        help="Total number of samples (default: 10)")
    parser.add_argument("--no-real",     action="store_true",
                        help="Skip Hugging Face download, use built-in fallback only")
    parser.add_argument("--output",      default="benchmark_results.json",
                        help="Output JSON file")
    args = parser.parse_args()

    run_benchmark(
        source=args.source,
        max_samples=args.samples,
        prefer_real=not args.no_real,
        output_file=args.output,
    )
