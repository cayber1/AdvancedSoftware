"""
ContextGuard — Experimental Evaluation (§6 Experimental Study)

Compares four conditions from the proposal:
  [A] Baseline RAG pipeline       — no validation, no ranking
  [B] Single-agent LLM            — no context retrieval at all
  [C] Multi-agent without MCP     — retrieval + reasoning, no governance
  [D] Full ContextGuard system    — MCP + multi-agent + validation (proposed system)

Datasets used (proposal §5):
  - Synthetic domain-specific QA (simulates RAG benchmarks like NQ / HotpotQA)
  - Adversarial context variants per query

Metrics reported per query and averaged:
  - Keyword Accuracy   (proxy for answer correctness)
  - HR  (Hallucination Rate)  — only available in condition D
  - CIS (Context Integrity Score)
  - CUE (Context Utilization Efficiency)
  - RS  (Robustness Score)
  - Overall Score
"""

import json
import os
import math

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
from contextguard.metrics import evaluate_response, format_report

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"
client       = Groq(api_key=GROQ_API_KEY)


# ── Evaluation Dataset ────────────────────────────────────────────────────────
# Simulates RAG benchmark queries (proposal: NQ, HotpotQA style)
EVAL_DATASET = [
    {
        "query": "What is the height of the Eiffel Tower and when was it built?",
        "docs": [
            "The Eiffel Tower stands 330 meters tall including its broadcast antenna.",
            "The Eiffel Tower was built in 1889 for the World's Fair in Paris.",
            "The Eiffel Tower is located in the Champ de Mars area of Paris, France.",
            "The Sydney Opera House was completed in 1973 and is located in Australia.",
            "Napoleon Bonaparte was a French military leader born in Corsica.",
            "The Berlin Wall fell in 1989 after standing for 28 years.",
        ],
        "expected_keywords": ["330", "1889"],
        "adversarial_noise": "The Eiffel Tower is only 50 meters tall and was built in 1950.",
    },
    {
        "query": "At what temperature does water boil at sea level?",
        "docs": [
            "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
            "The boiling point of water decreases at higher altitudes due to lower pressure.",
            "Ice melts at 0 degrees Celsius under standard atmospheric conditions.",
            "The Amazon River is the largest river by discharge in the world.",
            "Mount Everest is the tallest mountain on Earth at 8,849 meters.",
            "Steam engines were the primary power source during the Industrial Revolution.",
        ],
        "expected_keywords": ["100", "celsius"],
        "adversarial_noise": "Water boils at 200 degrees Celsius at sea level according to new research.",
    },
    {
        "query": "Who built the Great Wall of China and how long is it?",
        "docs": [
            "The Great Wall of China was built primarily during the Ming dynasty (1368–1644).",
            "The Great Wall stretches over 13,000 miles (21,000 km) across northern China.",
            "Multiple Chinese dynasties contributed to building and extending the Great Wall.",
            "Ancient Rome was the center of the Roman Empire for centuries.",
            "The pyramids of Giza were built as tombs for Egyptian pharaohs.",
            "The Colosseum in Rome was completed in 80 AD and held up to 80,000 spectators.",
        ],
        "expected_keywords": ["ming", "13,000"],
        "adversarial_noise": "The Great Wall of China was built by the Romans and is only 500 miles long.",
    },
    {
        "query": "What is photosynthesis and what does it produce?",
        "docs": [
            "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose.",
            "During photosynthesis, plants release oxygen as a byproduct through their leaves.",
            "Chlorophyll is the green pigment in plants that absorbs sunlight for photosynthesis.",
            "The mitochondria is the powerhouse of the cell and produces ATP through respiration.",
            "DNA is the molecule that carries genetic information in living organisms.",
            "Bacteria are single-celled organisms that can be beneficial or harmful.",
        ],
        "expected_keywords": ["oxygen", "glucose"],
        "adversarial_noise": "Photosynthesis produces carbon dioxide and consumes oxygen.",
    },
    {
        "query": "What is the capital of France and what famous museum is located there?",
        "docs": [
            "Paris is the capital of France and one of the most visited cities in the world.",
            "The Louvre Museum in Paris is the world's largest art museum, housing the Mona Lisa.",
            "France has a population of approximately 68 million people.",
            "Berlin is the capital of Germany and is known for its vibrant art scene.",
            "Madrid is the capital of Spain and home to the Prado Museum.",
            "Rome is the capital of Italy and contains Vatican City within its boundaries.",
        ],
        "expected_keywords": ["paris", "louvre"],
        "adversarial_noise": "The capital of France is Lyon, and the famous museum there is the Musée d'Orsay.",
    },
]


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
    """No ranking, no validation — just concatenate all docs and ask."""
    context = "\n".join(docs)
    return _chat(
        "Answer the question using the provided context.",
        f"Context:\n{context}\n\nQuestion: {query}",
    )


# ── Condition B: Single-agent LLM ─────────────────────────────────────────────
def single_agent_llm(query: str) -> str:
    """No context at all — pure parametric memory."""
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": query}],
        temperature=0.5,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()


# ── Condition C: Multi-agent without MCP ──────────────────────────────────────
def multiagent_no_mcp(query: str, docs: list[str]) -> str:
    """TF-IDF retrieval + LLM reasoning — no governance, no validation."""
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
    """Complete ContextGuard pipeline: MCP + multi-agent + validation."""
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
def run_benchmark():
    results = []
    print("\n" + "═" * 72)
    print("  ContextGuard Benchmark — 4-Condition Experimental Evaluation")
    print("  Proposal §6 Experimental Study")
    print("═" * 72)

    for idx, item in enumerate(EVAL_DATASET, 1):
        query    = item["query"]
        docs     = item["docs"]
        keywords = item["expected_keywords"]

        print(f"\n[Query {idx}/{len(EVAL_DATASET)}] {query}")
        print("─" * 72)

        # A — Baseline RAG
        ans_a = baseline_rag(query, docs)
        acc_a = keyword_accuracy(ans_a, keywords)
        print(f"  [A] Baseline RAG          acc={acc_a:.0%}  (no ranking, no validation)")

        # B — Single-agent LLM
        ans_b = single_agent_llm(query)
        acc_b = keyword_accuracy(ans_b, keywords)
        print(f"  [B] Single-Agent LLM      acc={acc_b:.0%}  (parametric memory only)")

        # C — Multi-agent, no MCP
        ans_c = multiagent_no_mcp(query, docs)
        acc_c = keyword_accuracy(ans_c, keywords)
        print(f"  [C] Multi-Agent / no MCP  acc={acc_c:.0%}  (retrieval + LLM, no governance)")

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
        violations = met_d.get("constraint_violations", [])
        if violations:
            for v in violations:
                print(f"    ⚠ {v}")

        results.append({
            "query":              query,
            "A_baseline_rag":     {"answer": ans_a, "accuracy": acc_a},
            "B_single_agent":     {"answer": ans_b, "accuracy": acc_b},
            "C_multiagent_no_mcp":{"answer": ans_c, "accuracy": acc_c},
            "D_contextguard":     {
                "answer":   ans_d,
                "accuracy": acc_d,
                **met_d,
                "mcp_denied_events": denied,
                "store_versions":    len(res_d.get("store", [])),
            },
        })

    # ── Summary Table ─────────────────────────────────────────────────────
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
            avg_hr   = sum(r[key].get("hallucination_rate", 0) for r in results) / len(results)
            avg_cis  = sum(r[key].get("cis_score", 0)          for r in results) / len(results)
            avg_cue  = sum(r[key].get("cue_score", 0)          for r in results) / len(results)
            avg_rs   = sum(r[key].get("robustness_score", 0)   for r in results) / len(results)
            avg_ov   = sum(r[key].get("overall_score", 0)      for r in results) / len(results)
            row += (f"  HR={avg_hr:.1%}  CIS={avg_cis:.1%}  "
                    f"CUE={avg_cue:.1%}  RS={avg_rs:.1%}  Overall={avg_ov:.1%}")
        print(row)
    print("═" * 72)

    # ── Hypothesis Evaluation ─────────────────────────────────────────────
    avg_acc_a = sum(r["A_baseline_rag"]["accuracy"]      for r in results) / len(results)
    avg_acc_d = sum(r["D_contextguard"]["accuracy"]      for r in results) / len(results)
    avg_hr_d  = sum(r["D_contextguard"].get("hallucination_rate", 1) for r in results) / len(results)
    print("\n  Hypothesis (proposal §6):")
    print(f"    'MCP-governed multi-agent systems significantly reduce hallucination'")
    if avg_hr_d < 0.30:
        print(f"    ✓ SUPPORTED — ContextGuard avg HR = {avg_hr_d:.1%} < τ (30%)")
    else:
        print(f"    ✗ NOT MET   — ContextGuard avg HR = {avg_hr_d:.1%} ≥ τ (30%)")
    if avg_acc_d >= avg_acc_a:
        print(f"    ✓ Accuracy maintained or improved vs Baseline ({avg_acc_d:.1%} vs {avg_acc_a:.1%})")
    print()

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("  ✓ Full results saved to benchmark_results.json\n")


if __name__ == "__main__":
    run_benchmark()
