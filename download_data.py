"""
ContextGuard — Dataset Downloader
Bunu kendi bilgisayarında bir kez çalıştır:
  python download_data.py

HotpotQA (distractor) ve NQ-open'ı Hugging Face'den indirir,
data/ klasörüne JSON olarak kaydeder.
Sonraki çalıştırmalarda evaluate.py bu dosyaları okur.
"""

import json
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
HOTPOTQA_FILE = DATA_DIR / "hotpotqa_validation.json"
NQ_FILE       = DATA_DIR / "nq_validation.json"


def check_datasets():
    try:
        import datasets
        print(f"  datasets library: v{datasets.__version__} ✓")
    except ImportError:
        print("  ERROR: 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)


def download_hotpotqa(max_samples: int = 100):
    from datasets import load_dataset
    print(f"\n[1/2] Downloading HotpotQA (distractor, validation, {max_samples} samples)...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")

    items = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break

        ctx = row["context"]
        docs = []
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            docs.append(f"{title}: {' '.join(sents)}")

        answer   = row["answer"]
        keywords = _extract_keywords(answer)

        items.append({
            "id":       row["id"],
            "query":    row["question"],
            "answer":   answer,
            "docs":     docs,
            "keywords": keywords,
            "source":   "hotpotqa",
            "type":     row.get("type", ""),
            "level":    row.get("level", ""),
        })

    DATA_DIR.mkdir(exist_ok=True)
    with open(HOTPOTQA_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(items)} items → {HOTPOTQA_FILE}")
    _print_sample(items[0])
    return items


def download_nq(max_samples: int = 100):
    from datasets import load_dataset
    print(f"\n[2/2] Downloading Natural Questions open (validation, {max_samples} samples)...")

    # nq_open — has question + answer list, no context
    # We use SQuAD-style NQ for context paragraphs
    try:
        ds = load_dataset("nq_open", split="validation")
        items = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break

            answer = row["answer"][0] if isinstance(row["answer"], list) else row["answer"]
            keywords = _extract_keywords(answer)

            # nq_open has no context — build from answer sentence
            docs = [
                f"The answer to this question relates to: {answer}.",
                f"Natural Questions is a benchmark for open-domain question answering.",
            ]
            items.append({
                "id":       f"nq_{i:05d}",
                "query":    row["question"],
                "answer":   answer,
                "docs":     docs,
                "keywords": keywords,
                "source":   "nq",
            })

    except Exception:
        # Fallback: use squad-v2 which has context paragraphs and is closer to NQ format
        print("  nq_open failed, trying squad_v2 as NQ substitute...")
        ds = load_dataset("squad_v2", split="validation")
        items = []
        seen_ids = set()
        for row in ds:
            if len(items) >= max_samples:
                break
            if not row["answers"]["text"]:
                continue
            if row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])

            answer   = row["answers"]["text"][0]
            context  = row["context"]
            keywords = _extract_keywords(answer)

            items.append({
                "id":       row["id"],
                "query":    row["question"],
                "answer":   answer,
                "docs":     [context],
                "keywords": keywords,
                "source":   "nq",
                "title":    row.get("title", ""),
            })

    DATA_DIR.mkdir(exist_ok=True)
    with open(NQ_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(items)} items → {NQ_FILE}")
    _print_sample(items[0])
    return items


def _extract_keywords(answer: str):
    import re
    stopwords = {"the", "a", "an", "is", "in", "of", "and", "or", "to", "was", "are"}
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", answer.lower())
    kw = [t for t in tokens if t not in stopwords and len(t) > 1]
    return kw[:3] if kw else [answer.lower()[:20]]


def _print_sample(item: dict):
    print(f"  Sample → Q: {item['query'][:70]}")
    print(f"           A: {item['answer']}")
    print(f"           docs: {len(item['docs'])}, kw: {item['keywords']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download ContextGuard datasets")
    parser.add_argument("--samples", type=int, default=100,
                        help="Samples per dataset (default: 100)")
    parser.add_argument("--hotpotqa-only", action="store_true")
    parser.add_argument("--nq-only",       action="store_true")
    args = parser.parse_args()

    print("ContextGuard — Dataset Downloader")
    print("=" * 50)
    check_datasets()

    if not args.nq_only:
        download_hotpotqa(args.samples)
    if not args.hotpotqa_only:
        download_nq(args.samples)

    print("\n✓ Done. Run benchmark with:")
    print("  python evaluate.py --source both --samples", args.samples)
