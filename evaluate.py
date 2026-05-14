"""
ContextGuard Experimental Evaluation (Groq backend)
Compares 4 conditions from the proposal:
  A) Baseline RAG pipeline (no validation)
  B) Single-agent LLM (no retrieval ranking)
  C) Multi-agent without MCP constraints
  D) Full system (MCP + multi-agent validation)  <- ContextGuard
"""

import json
from groq import Groq
from contextguard.mcp_governance import MCPGovernance
from contextguard.agents import (
    ContextRetrievalAgent,
    ReasoningAgent,
    GroundingValidatorAgent,
    AdversarialTesterAgent,
)
from contextguard.metrics import evaluate_response

GROQ_API_KEY = "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"
GROQ_MODEL   = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY)

EVAL_DATASET = [
    {
        "query": "What is the height of the Eiffel Tower?",
        "docs": [
            "The Eiffel Tower stands 330 meters tall including its broadcast antenna.",
            "The Eiffel Tower is located in Paris, France.",
            "The Sydney Opera House is a famous landmark in Australia.",
            "Napoleon Bonaparte was a French military leader.",
        ],
        "expected_keywords": ["330", "meters"],
    },
    {
        "query": "At what temperature does water boil?",
        "docs": [
            "Water boils at 100 degrees Celsius at sea level under standard pressure.",
            "The Amazon River is the longest river in South America.",
            "Mount Everest is the tallest mountain on Earth at 8,849 meters.",
            "Ice melts at 0 degrees Celsius.",
        ],
        "expected_keywords": ["100", "celsius"],
    },
    {
        "query": "Who built the Great Wall of China?",
        "docs": [
            "The Great Wall of China was built primarily during the Ming dynasty.",
            "China is the most populous country in the world.",
            "The Great Wall stretches over 13,000 miles.",
            "Ancient Rome was the center of the Roman Empire.",
        ],
        "expected_keywords": ["ming", "dynasty"],
    },
]


def _chat(system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.3,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()


def baseline_rag(query: str, docs: list[str]) -> str:
    context = "\n".join(docs)
    return _chat(
        "Answer the question using the provided context.",
        f"Context:\n{context}\n\nQuestion: {query}",
    )


def single_agent_llm(query: str) -> str:
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": query}],
        temperature=0.5,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()


def multiagent_no_mcp(query: str, docs: list[str]) -> str:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec    = TfidfVectorizer().fit_transform([query] + docs)
    scores = cosine_similarity(vec[0:1], vec[1:])[0]
    top    = [docs[i] for i in scores.argsort()[::-1][:3]]
    return _chat(
        "Answer based strictly on the context.",
        f"Context:\n{chr(10).join(top)}\n\nQuestion: {query}",
    )


def full_contextguard(query: str, docs: list[str]) -> dict:
    mcp         = MCPGovernance()
    retrieval   = ContextRetrievalAgent(mcp)
    reasoning   = ReasoningAgent(mcp)
    validator   = GroundingValidatorAgent(mcp)
    adversarial = AdversarialTesterAgent(mcp)
    context     = retrieval.retrieve_and_rank(query, docs)
    answer      = reasoning.generate_answer(query, context)
    validation  = validator.validate(query, answer, context)
    robustness  = adversarial.test_robustness(query, context, reasoning, num_trials=2)
    metrics     = evaluate_response(answer, context, validation, robustness)
    return {"answer": answer, "metrics": metrics}


def keyword_accuracy(answer: str, keywords: list[str]) -> float:
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords) if keywords else 0.0


def run_benchmark():
    results = []
    print("\n" + "=" * 70)
    print("ContextGuard Benchmark — 4-Condition Experimental Evaluation")
    print("=" * 70)

    for item in EVAL_DATASET:
        query    = item["query"]
        docs     = item["docs"]
        keywords = item["expected_keywords"]
        print(f"\nQuery: {query}")

        ans_a = baseline_rag(query, docs);      acc_a = keyword_accuracy(ans_a, keywords)
        ans_b = single_agent_llm(query);        acc_b = keyword_accuracy(ans_b, keywords)
        ans_c = multiagent_no_mcp(query, docs); acc_c = keyword_accuracy(ans_c, keywords)
        res_d = full_contextguard(query, docs)
        ans_d = res_d["answer"];                acc_d = keyword_accuracy(ans_d, keywords)
        met_d = res_d["metrics"]

        print(f"  [A] Baseline RAG       accuracy={acc_a:.0%}")
        print(f"  [B] Single-Agent LLM   accuracy={acc_b:.0%}")
        print(f"  [C] Multi-Agent/noMCP  accuracy={acc_c:.0%}")
        print(f"  [D] ContextGuard       accuracy={acc_d:.0%}  CIS={met_d['cis_score']:.0%}  HR={met_d['hallucination_rate']:.0%}  RS={met_d['robustness_score']:.0%}")

        results.append({
            "query": query,
            "A_baseline_rag":      {"accuracy": acc_a},
            "B_single_agent":      {"accuracy": acc_b},
            "C_multiagent_no_mcp": {"accuracy": acc_c},
            "D_contextguard":      {"accuracy": acc_d, **met_d},
        })

    print("\n" + "=" * 70)
    print("Summary — averages across all queries:")
    for cond in ["A_baseline_rag", "B_single_agent", "C_multiagent_no_mcp", "D_contextguard"]:
        avg = sum(r[cond]["accuracy"] for r in results) / len(results)
        print(f"  {cond.replace('_',' ').title():30s}: avg accuracy = {avg:.1%}")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmark_results.json")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
