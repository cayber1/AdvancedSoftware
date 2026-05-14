"""
ContextGuard: Multi-Agent MCP Framework for Context Integrity Verification
Group 2 - Diyar Buyuksahin, Etem Tolga Erten, Süleyman Kılıç, Andrew Mabuto
"""

import os
import json
import time
from contextguard.mcp_governance import MCPGovernance
from contextguard.agents import (
    ContextRetrievalAgent,
    ReasoningAgent,
    GroundingValidatorAgent,
    AdversarialTesterAgent,
)
from contextguard.metrics import evaluate_response


def run_contextguard_pipeline(query: str, context_documents: list[str]) -> dict:
    """
    Full ContextGuard pipeline:
    1. Context Retrieval Agent ranks & filters docs
    2. Reasoning Agent generates answer
    3. Grounding Validator verifies grounding
    4. Adversarial Tester checks robustness
    """

    mcp = MCPGovernance()

    print("\n" + "=" * 60)
    print("ContextGuard Pipeline Starting")
    print("=" * 60)
    print(f"Query: {query}\n")

    # Step 1: Context Retrieval
    print("[Agent 1] Context Retrieval Agent...")
    retrieval_agent = ContextRetrievalAgent(mcp)
    retrieved_context = retrieval_agent.retrieve_and_rank(query, context_documents)
    print(f"  Retrieved {len(retrieved_context)} relevant context segments.\n")

    # Step 2: Reasoning Agent
    print("[Agent 2] Reasoning Agent...")
    reasoning_agent = ReasoningAgent(mcp)
    raw_answer = reasoning_agent.generate_answer(query, retrieved_context)
    print(f"  Answer generated ({len(raw_answer)} chars).\n")

    # Step 3: Grounding Validator
    print("[Agent 3] Grounding Validator Agent...")
    validator_agent = GroundingValidatorAgent(mcp)
    validation_result = validator_agent.validate(query, raw_answer, retrieved_context)
    print(f"  Hallucination Risk: {validation_result['hallucination_risk']:.2%}")
    print(f"  CIS Score: {validation_result['cis_score']:.2%}\n")

    # Step 4: Adversarial Tester
    print("[Agent 4] Adversarial Tester Agent...")
    adversarial_agent = AdversarialTesterAgent(mcp)
    robustness_result = adversarial_agent.test_robustness(
        query, retrieved_context, reasoning_agent
    )
    print(f"  Robustness Score: {robustness_result['robustness_score']:.2%}\n")

    # Final Metrics
    metrics = evaluate_response(
        answer=raw_answer,
        context=retrieved_context,
        validation=validation_result,
        robustness=robustness_result,
    )

    # MCP Execution Log
    mcp_log = mcp.get_execution_log()

    result = {
        "query": query,
        "answer": raw_answer,
        "retrieved_context": retrieved_context,
        "validation": validation_result,
        "robustness": robustness_result,
        "metrics": metrics,
        "mcp_log": mcp_log,
    }

    print("=" * 60)
    print("FINAL ANSWER:")
    print("-" * 60)
    print(raw_answer)
    print("-" * 60)
    print(f"Hallucination Rate (HR):        {metrics['hallucination_rate']:.2%}")
    print(f"Context Integrity Score (CIS):  {metrics['cis_score']:.2%}")
    print(f"Context Utilization Eff. (CUE): {metrics['cue_score']:.2%}")
    print(f"Robustness Score (RS):          {metrics['robustness_score']:.2%}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    # --- Demo: Simple RAG-like scenario ---
    sample_docs = [
        "The Eiffel Tower is located in Paris, France. It was built in 1889 by Gustave Eiffel.",
        "The Great Wall of China stretches over 13,000 miles and was built during the Ming dynasty.",
        "Water boils at 100 degrees Celsius at sea level under standard atmospheric pressure.",
        "The Eiffel Tower stands 330 meters tall and attracts millions of tourists each year.",
        "Photosynthesis is the process by which plants convert sunlight into food using chlorophyll.",
        "Paris is the capital of France and has a population of over 2 million people in the city proper.",
    ]

    query = "How tall is the Eiffel Tower and where is it located?"

    result = run_contextguard_pipeline(query, sample_docs)

    # Save full result as JSON
    with open("contextguard_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nFull result saved to contextguard_result.json")
