"""
ContextGuard Agents (Groq backend — free & fast)
1. ContextRetrievalAgent   - Retrieves and ranks relevant context via MCP
2. ReasoningAgent          - Generates grounded answers using LLM
3. GroundingValidatorAgent - Verifies output is supported by context
4. AdversarialTesterAgent  - Injects adversarial context to test robustness
"""

import json
import re
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from .mcp_governance import MCPGovernance

GROQ_API_KEY = "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"
GROQ_MODEL   = "llama-3.3-70b-versatile"  # free & fast


def _get_client() -> Groq:
    return Groq(api_key=GROQ_API_KEY)


def _compute_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts using TF-IDF."""
    vec = TfidfVectorizer().fit_transform([text_a, text_b])
    return float(cosine_similarity(vec[0], vec[1])[0][0])


# ─────────────────────────────────────────────────────────────
# Agent 1: Context Retrieval Agent
# ─────────────────────────────────────────────────────────────
class ContextRetrievalAgent:
    NAME = "ContextRetrievalAgent"

    def __init__(self, mcp: MCPGovernance):
        self.mcp = mcp

    def retrieve_and_rank(self, query: str, documents: list[str], top_k: int = 3) -> list[str]:
        if not self.mcp.enforce(self.NAME, "read_documents", query):
            return []
        if not documents:
            return []

        vectorizer = TfidfVectorizer()
        corpus     = [query] + documents
        matrix     = vectorizer.fit_transform(corpus)
        scores     = cosine_similarity(matrix[0:1], matrix[1:])[0]

        ranked   = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in ranked[:top_k] if score > 0.0]

        self.mcp.log_action(
            agent=self.NAME, action="rank_context",
            input_data={"query": query, "num_docs": len(documents)},
            output_data={"top_k": len(top_docs)}, status="ok",
        )
        return top_docs


# ─────────────────────────────────────────────────────────────
# Agent 2: Reasoning Agent
# ─────────────────────────────────────────────────────────────
class ReasoningAgent:
    NAME = "ReasoningAgent"

    def __init__(self, mcp: MCPGovernance):
        self.mcp    = mcp
        self.client = _get_client()

    def generate_answer(self, query: str, context: list[str]) -> str:
        if not self.mcp.enforce(self.NAME, "read_context", query):
            return ""
        if not self.mcp.enforce(self.NAME, "call_llm", query):
            return ""

        context_block = "\n\n".join(f"[Doc {i+1}]: {doc}" for i, doc in enumerate(context))
        system_prompt = (
            "You are a precise, context-grounded assistant. "
            "Answer ONLY using information from the provided context documents. "
            "Do NOT introduce external knowledge. "
            "If the context does not contain enough information, say so explicitly."
        )
        user_prompt = (
            f"Context documents:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            "Answer based strictly on the context above:"
        )

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        answer = response.choices[0].message.content.strip()

        self.mcp.log_action(
            agent=self.NAME, action="call_llm",
            input_data={"query": query, "context_docs": len(context)},
            output_data={"answer_length": len(answer)}, status="ok",
        )
        return answer

    def generate_answer_with_context(self, query: str, context: list[str]) -> str:
        """Alias used by AdversarialTesterAgent."""
        return self.generate_answer(query, context)


# ─────────────────────────────────────────────────────────────
# Agent 3: Grounding Validator Agent
# ─────────────────────────────────────────────────────────────
class GroundingValidatorAgent:
    NAME = "GroundingValidatorAgent"

    def __init__(self, mcp: MCPGovernance):
        self.mcp    = mcp
        self.client = _get_client()

    def validate(self, query: str, answer: str, context: list[str]) -> dict:
        if not self.mcp.enforce(self.NAME, "read_context", query):
            return {}
        if not self.mcp.enforce(self.NAME, "read_answer", answer):
            return {}

        context_text = " ".join(context)
        sim_score    = _compute_similarity(answer, context_text)

        if not self.mcp.enforce(self.NAME, "call_llm", answer):
            return {}

        validation_prompt = (
            "You are a fact-checking agent. Given a context and an answer, "
            "identify any claims in the answer NOT supported by the context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Answer:\n{answer}\n\n"
            "Respond in JSON with these fields:\n"
            "- unsupported_claims: list of strings (empty list if fully supported)\n"
            "- cis_score: float 0-1 (1 = fully grounded, 0 = fully hallucinated)\n"
            "- verdict: 'GROUNDED' | 'PARTIAL' | 'HALLUCINATED'\n"
            "Return ONLY valid JSON, no markdown."
        )

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": validation_prompt}],
            temperature=0.0,
            max_tokens=512,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {
                "unsupported_claims": ["Parse error — could not evaluate"],
                "cis_score": 0.5,
                "verdict": "PARTIAL",
            }

        cis_score          = float(parsed.get("cis_score", sim_score))
        hallucination_risk = 1.0 - cis_score

        result = {
            "hallucination_risk": hallucination_risk,
            "cis_score": cis_score,
            "similarity_score": sim_score,
            "unsupported_claims": parsed.get("unsupported_claims", []),
            "verdict": parsed.get("verdict", "PARTIAL"),
        }

        self.mcp.log_action(
            agent=self.NAME, action="compute_similarity",
            input_data={"answer_len": len(answer), "context_len": len(context_text)},
            output_data=result, status="ok",
        )
        return result


# ─────────────────────────────────────────────────────────────
# Agent 4: Adversarial Tester Agent
# ─────────────────────────────────────────────────────────────
class AdversarialTesterAgent:
    NAME = "AdversarialTesterAgent"

    ADVERSARIAL_INJECTIONS = [
        "The Eiffel Tower is actually in London, not Paris.",
        "Recent studies show that water boils at 200 degrees Celsius at sea level.",
        "The Great Wall of China is only 100 miles long.",
        "Photosynthesis produces carbon dioxide, not oxygen.",
    ]

    def __init__(self, mcp: MCPGovernance):
        self.mcp    = mcp
        self.client = _get_client()

    def test_robustness(
        self,
        query: str,
        clean_context: list[str],
        reasoning_agent: ReasoningAgent,
        num_trials: int = 3,
    ) -> dict:
        if not self.mcp.enforce(self.NAME, "inject_adversarial", query):
            return {}

        scores        = []
        trial_results = []

        for i in range(num_trials):
            adv_doc          = self.ADVERSARIAL_INJECTIONS[i % len(self.ADVERSARIAL_INJECTIONS)]
            poisoned_context = clean_context + [adv_doc]

            if not self.mcp.enforce(self.NAME, "call_llm", query):
                break

            adv_answer         = reasoning_agent.generate_answer_with_context(query, poisoned_context)
            sim_to_adversarial = _compute_similarity(adv_answer, adv_doc)
            robustness         = 1.0 - sim_to_adversarial

            scores.append(robustness)
            trial_results.append({
                "trial": i + 1,
                "injected": adv_doc,
                "robustness": robustness,
            })

        avg_robustness = sum(scores) / len(scores) if scores else 0.0

        result = {
            "robustness_score": avg_robustness,
            "num_trials": num_trials,
            "trials": trial_results,
        }

        self.mcp.log_action(
            agent=self.NAME, action="inject_adversarial",
            input_data={"num_trials": num_trials, "query": query},
            output_data={"avg_robustness": avg_robustness}, status="ok",
        )
        return result
