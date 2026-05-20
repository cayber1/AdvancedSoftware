"""
ContextGuard — Attribution Scoring
Implements context-to-output mapping (proposal §Detection Methods).

For each sentence in the answer we compute:
  - which source document supports it most
  - the support score (TF-IDF cosine similarity)
  - whether it crosses the grounding threshold

This produces a fine-grained attribution map and an overall Attribution Score (AS).
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


GROUNDING_THRESHOLD = 0.08   # minimum similarity to count as "supported"


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


def compute_attribution(answer: str, context_docs: list[str]) -> dict:
    """
    For every sentence in the answer, find the best-matching context doc
    and its similarity score.

    Returns:
        {
            "attribution_map": [
                {"sentence": str, "best_doc_idx": int, "score": float, "supported": bool},
                ...
            ],
            "attribution_score": float,   # fraction of sentences that are supported
            "unsupported_sentences": [str],
        }
    """
    sentences = _split_sentences(answer)
    if not sentences or not context_docs:
        return {
            "attribution_map": [],
            "attribution_score": 0.0,
            "unsupported_sentences": [],
        }

    attribution_map = []
    unsupported = []

    for sent in sentences:
        corpus = [sent] + context_docs
        try:
            vec = TfidfVectorizer().fit_transform(corpus)
            sims = cosine_similarity(vec[0:1], vec[1:])[0]
        except ValueError:
            sims = [0.0] * len(context_docs)

        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])
        supported = best_score >= GROUNDING_THRESHOLD

        attribution_map.append({
            "sentence": sent,
            "best_doc_idx": best_idx,
            "score": round(best_score, 4),
            "supported": supported,
        })

        if not supported:
            unsupported.append(sent)

    supported_count = sum(1 for a in attribution_map if a["supported"])
    attribution_score = supported_count / len(sentences) if sentences else 0.0

    return {
        "attribution_map": attribution_map,
        "attribution_score": round(attribution_score, 4),
        "unsupported_sentences": unsupported,
    }
