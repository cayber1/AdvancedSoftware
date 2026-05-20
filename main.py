"""
ContextGuard — Main Pipeline
Group 2: Diyar Buyuksahin, Etem Tolga Erten, Süleyman Kılıç, Andrew Mabuto

Full pipeline:
  [1] Context Retrieval Agent   — ranks & filters docs, commits to ContextStore
  [2] Reasoning Agent           — generates context-grounded answer
  [3] Grounding Validator Agent — verifies grounding (3-layer check)
  [4] Adversarial Tester Agent  — LLM-crafted injections → robustness score

All steps are governed by MCPGovernance (RBAC + schema validation + logging).
"""

import json
import os
import sys

from contextguard.mcp_governance import MCPGovernance
from contextguard.context_store import ContextStore
from contextguard.agents import (
    ContextRetrievalAgent,
    ReasoningAgent,
    GroundingValidatorAgent,
    AdversarialTesterAgent,
)
from contextguard.metrics import evaluate_response, format_report


def run_contextguard_pipeline(
    query: str,
    context_documents: list[str],
    num_adversarial_trials: int = 3,
    verbose_mcp: bool = False,
) -> dict:
    """
    Execute the full ContextGuard multi-agent pipeline.

    Parameters
    ----------
    query                  : The user question
    context_documents      : Candidate document pool
    num_adversarial_trials : How many adversarial injections to test
    verbose_mcp            : If True, print MCP log details

    Returns
    -------
    Comprehensive result dict including answer, all metrics, MCP log,
    ContextStore history, and formal risk model output.
    """

    mcp   = MCPGovernance()
    store = ContextStore()

    print("\n" + "═" * 62)
    print("  ContextGuard — Multi-Agent MCP Pipeline")
    print("═" * 62)
    print(f"  Query: {query}\n")

    # ── Step 1: Context Retrieval ─────────────────────────────────────────
    print("[Agent 1] ContextRetrievalAgent — ranking documents …")
    retrieval_agent  = ContextRetrievalAgent(mcp, store)
    retrieved, vid   = retrieval_agent.retrieve_and_rank(query, context_documents)
    print(f"          Retrieved {len(retrieved)} docs  |  version_id={vid}\n")

    if not retrieved:
        print("  ⚠ No relevant context found. Aborting pipeline.")
        return {}

    # ── Step 2: Reasoning ─────────────────────────────────────────────────
    print("[Agent 2] ReasoningAgent — generating grounded answer …")
    reasoning_agent = ReasoningAgent(mcp, store)
    answer          = reasoning_agent.generate_answer(query, retrieved, version_id=vid)
    print(f"          Answer: {answer[:120]}{'…' if len(answer) > 120 else ''}\n")

    # ── Step 3: Grounding Validation ──────────────────────────────────────
    print("[Agent 3] GroundingValidatorAgent — verifying grounding …")
    validator_agent   = GroundingValidatorAgent(mcp, store)
    validation_result = validator_agent.validate(query, answer, retrieved)
    print(f"          Verdict          : {validation_result.get('verdict', '?')}")
    print(f"          CIS Score        : {validation_result.get('cis_score', 0):.2%}")
    print(f"          Hallucination HR : {validation_result.get('hallucination_risk', 0):.2%}")
    attr = validation_result.get("attribution", {})
    if attr:
        print(f"          Attribution Score: {attr.get('attribution_score', 0):.2%}")
    if validation_result.get("unsupported_claims"):
        print(f"          Unsupported claims: {validation_result['unsupported_claims']}")
    print()

    # ── Step 4: Adversarial Testing ───────────────────────────────────────
    print(f"[Agent 4] AdversarialTesterAgent — {num_adversarial_trials} LLM-crafted injections …")
    adversarial_agent  = AdversarialTesterAgent(mcp, store)
    robustness_result  = adversarial_agent.test_robustness(
        query, retrieved, reasoning_agent, num_trials=num_adversarial_trials
    )
    print(f"          Robustness Score : {robustness_result.get('robustness_score', 0):.2%}")
    for t in robustness_result.get("trials", []):
        print(f"            Trial {t['trial']}: injected='{t['injected_doc'][:60]}…'  "
              f"RS={t['robustness']:.2%}")
    print()

    # ── Final Metrics ─────────────────────────────────────────────────────
    metrics = evaluate_response(
        answer=answer,
        context=retrieved,
        validation=validation_result,
        robustness=robustness_result,
    )

    print(format_report(metrics, query))

    # ── MCP Log ───────────────────────────────────────────────────────────
    mcp.print_log(verbose=verbose_mcp)

    # ── ContextStore History ──────────────────────────────────────────────
    store_history = store.history()

    result = {
        "query":              query,
        "answer":             answer,
        "retrieved_context":  retrieved,
        "context_version_id": vid,
        "validation":         validation_result,
        "robustness":         robustness_result,
        "metrics":            metrics,
        "mcp_log":            mcp.get_execution_log(),
        "context_store":      store_history,
    }

    return result


if __name__ == "__main__":
    # ── Demo scenario ────────────────────────────────────────────────────
    sample_docs = [
        "The Eiffel Tower is located in Paris, France. It was built in 1889 by Gustave Eiffel.",
        "The Great Wall of China stretches over 13,000 miles and was built during the Ming dynasty.",
        "Water boils at 100 degrees Celsius at sea level under standard atmospheric pressure.",
        "The Eiffel Tower stands 330 meters tall including its broadcast antenna.",
        "Photosynthesis is the process by which plants convert sunlight into food using chlorophyll.",
        "Paris is the capital of France and has a population of over 2 million people.",
        "The Louvre Museum in Paris houses over 35,000 works of art including the Mona Lisa.",
        "Mount Everest is the highest mountain on Earth at 8,849 meters above sea level.",
    ]

    query = "How tall is the Eiffel Tower and where is it located?"

    result = run_contextguard_pipeline(
        query=query,
        context_documents=sample_docs,
        num_adversarial_trials=3,
        verbose_mcp=False,
    )

    # Save full result
    with open("contextguard_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\n✓ Full result saved to contextguard_result.json")
