"""
ContextGuard Evaluation Metrics
- HR  (Hallucination Rate)
- CIS (Context Integrity Score)
- CUE (Context Utilization Efficiency)
- RS  (Robustness Score)
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _coverage(answer: str, context_docs: list[str], threshold: float = 0.1) -> float:
    """
    CUE: fraction of context docs that meaningfully contributed to the answer.
    A doc 'contributed' if its TF-IDF similarity to the answer > threshold.
    """
    if not context_docs:
        return 0.0

    used = 0
    for doc in context_docs:
        vec = TfidfVectorizer().fit_transform([answer, doc])
        sim = cosine_similarity(vec[0], vec[1])[0][0]
        if sim > threshold:
            used += 1

    return used / len(context_docs)


def evaluate_response(
    answer: str,
    context: list[str],
    validation: dict,
    robustness: dict,
) -> dict:
    """
    Aggregates all four ContextGuard metrics into a single evaluation dict.
    """

    # HR: derived from grounding validator's hallucination_risk
    hallucination_rate = validation.get("hallucination_risk", 0.5)

    # CIS: context integrity score from validator
    cis_score = validation.get("cis_score", 0.5)

    # CUE: computed from coverage analysis
    cue_score = _coverage(answer, context)

    # RS: average robustness from adversarial tester
    robustness_score = robustness.get("robustness_score", 0.5)

    # Overall system score (simple average — tunable weights)
    overall = (
        (1 - hallucination_rate) * 0.3
        + cis_score * 0.3
        + cue_score * 0.2
        + robustness_score * 0.2
    )

    return {
        "hallucination_rate": hallucination_rate,
        "cis_score": cis_score,
        "cue_score": cue_score,
        "robustness_score": robustness_score,
        "overall_score": overall,
    }
