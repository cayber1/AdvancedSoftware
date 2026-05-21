"""
ContextGuard — Dataset Loader
Proposal §5: "RAG benchmarks (Natural Questions, HotpotQA)"

3 katmanlı strateji (öncelik sırasıyla):
  1. data/ klasöründeki JSON (download_data.py ile indirilmiş gerçek dataset)
  2. Hugging Face datasets kütüphanesi (canlı internet bağlantısı)
  3. Built-in fallback örnekler (offline)

Kullanım:
  from contextguard.data_loader import load_dataset
  items = load_dataset("hotpotqa", max_samples=50)
  items = load_dataset("nq",       max_samples=50)
  items = load_dataset("both",     max_samples=100)
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Literal

DatasetSource = Literal["hotpotqa", "nq", "both"]

DATA_DIR      = Path(__file__).parent.parent / "data"
HOTPOTQA_FILE = DATA_DIR / "hotpotqa_validation.json"
NQ_FILE       = DATA_DIR / "nq_validation.json"


# ─────────────────────────────────────────────────────────────────────────────
# Built-in fallback (offline)
# ─────────────────────────────────────────────────────────────────────────────
_FALLBACK_HOTPOTQA = [
    {
        "id": "hpqa_f001",
        "query": "Which magazine was started first, Arthur's Magazine or First for Women?",
        "answer": "Arthur's Magazine",
        "docs": [
            "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.",
            "First for Women is a woman's magazine published by Bauer Media Group in the USA.",
            "The magazine was founded in 1989 and is based in Englewood Cliffs, New Jersey.",
            "Arthur's Magazine was founded in 1844, predating many modern publications by over a century.",
            "Bauer Media Group is one of the largest privately owned publishing groups in the world.",
            "Philadelphia has been a center of American publishing since the colonial era.",
            "Literary periodicals flourished in the mid-19th century United States.",
            "First for Women focuses on health, food, and lifestyle content for American women.",
            "The 1840s saw a proliferation of literary magazines in major American cities.",
            "Bauer Media publishes over 600 magazines globally across 15 countries.",
        ],
        "keywords": ["arthur", "1844"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f002",
        "query": "What nationality was the director of Baahubali: The Beginning?",
        "answer": "Indian",
        "docs": [
            "Baahubali: The Beginning is a 2015 Indian epic action film directed by S. S. Rajamouli.",
            "S. S. Rajamouli is an Indian film director known for his epic productions.",
            "The film was produced by Shobu Yarlagadda and Prasad Devineni.",
            "Rajamouli was born on October 10, 1973, in Raichur, Karnataka, India.",
            "Baahubali: The Beginning grossed over $250 million worldwide.",
            "The film stars Prabhas, Rana Daggubati, Anushka Shetty and Tamannaah.",
            "Telugu cinema, also known as Tollywood, is based in Hyderabad, India.",
            "The Baahubali franchise is one of the most successful Indian film series.",
            "S. S. Rajamouli won the National Film Award for Best Direction for RRR.",
            "Indian cinema produces more films annually than any other country.",
        ],
        "keywords": ["indian"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f003",
        "query": "The Oberoi family is part of a hotel company headquartered in what city?",
        "answer": "Delhi",
        "docs": [
            "The Oberoi Group is a hotel company with its headquarters in Delhi, India.",
            "The Oberoi family founded the Oberoi Group, one of Asia's leading luxury hotel chains.",
            "EIH Limited, the flagship company of The Oberoi Group, is listed on the Bombay Stock Exchange.",
            "The Oberoi Group operates 31 hotels and two river cruise ships in six countries.",
            "New Delhi serves as the political and commercial capital of India.",
            "Luxury hotel chains in India have expanded significantly in the 21st century.",
            "The Trident Hotels are also part of The Oberoi Group portfolio.",
            "Prithviraj Singh Oberoi led the Oberoi Group for several decades as chairman.",
            "Delhi's hospitality industry attracts millions of business and leisure travelers annually.",
            "The Oberoi New Delhi is one of the most prestigious hotels in the Indian capital.",
        ],
        "keywords": ["delhi"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f004",
        "query": "What is the capital of the country with the largest proven oil reserves?",
        "answer": "Caracas",
        "docs": [
            "Venezuela has the largest proven oil reserves in the world, surpassing Saudi Arabia.",
            "Caracas is the capital and largest city of Venezuela.",
            "Venezuela's proven oil reserves are estimated at around 300 billion barrels.",
            "Saudi Arabia's capital is Riyadh, and it holds the second-largest proven oil reserves.",
            "The Orinoco Belt in Venezuela contains vast heavy crude oil deposits.",
            "Venezuela joined OPEC in 1960 as one of its founding members.",
            "Caracas is located in a mountainous valley in northern Venezuela.",
            "Oil revenues have historically constituted the majority of Venezuela's export earnings.",
            "The Venezuelan bolívar is the official currency of Venezuela.",
            "OPEC was founded in Baghdad, Iraq in September 1960.",
        ],
        "keywords": ["caracas"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f005",
        "query": "AKMU, whose debut album is 2014 S/S, consists of how many members?",
        "answer": "two",
        "docs": [
            "2014 S/S is the debut album of AKMU, a South Korean music duo.",
            "AKMU consists of siblings Lee Chan-hyuk and Lee Su-hyun, making it a two-member group.",
            "Akdong Musician (AKMU) debuted under YG Entertainment in 2014.",
            "The album 2014 S/S was released on May 7, 2014 in South Korea.",
            "South Korean K-pop groups vary in size from soloists to groups of over ten members.",
            "YG Entertainment is one of the 'Big Four' South Korean entertainment companies.",
            "Lee Chan-hyuk wrote most of the songs on the debut album.",
            "AKMU won multiple awards at major Korean music award shows in 2014.",
            "The Mnet Asian Music Awards recognized AKMU as Best New Artist in 2014.",
            "K-pop duos are less common than larger idol groups in the South Korean music industry.",
        ],
        "keywords": ["two", "2"],
        "source": "hotpotqa",
    },
]

_FALLBACK_NQ = [
    {
        "id": "nq_f001",
        "query": "when did the us enter world war ii",
        "answer": "December 8, 1941",
        "docs": [
            "The United States entered World War II on December 8, 1941, one day after the Japanese attack on Pearl Harbor.",
            "President Franklin D. Roosevelt addressed Congress on December 8, 1941, calling December 7 'a date which will live in infamy.'",
            "Congress declared war on Japan on December 8, 1941, with only one dissenting vote.",
            "Germany and Italy declared war on the United States on December 11, 1941.",
            "The attack on Pearl Harbor on December 7, 1941, killed over 2,400 Americans.",
            "Prior to Pearl Harbor, the US had been providing support to the Allies through the Lend-Lease Act.",
            "World War II began in Europe on September 1, 1939, when Germany invaded Poland.",
            "The US had maintained a policy of official neutrality until the Pearl Harbor attack.",
        ],
        "keywords": ["december", "1941"],
        "source": "nq",
    },
    {
        "id": "nq_f002",
        "query": "who wrote the book to kill a mockingbird",
        "answer": "Harper Lee",
        "docs": [
            "To Kill a Mockingbird is a novel by Harper Lee published on July 11, 1960.",
            "Harper Lee was an American novelist best known for To Kill a Mockingbird.",
            "The novel won the Pulitzer Prize for Fiction in 1961.",
            "Harper Lee was born on April 28, 1926, in Monroeville, Alabama.",
            "To Kill a Mockingbird was adapted into an Academy Award-winning film in 1962.",
            "Lee's second novel, Go Set a Watchman, was published in 2015.",
            "The story is set in the fictional town of Maycomb, Alabama, during the 1930s.",
            "Harper Lee received the Presidential Medal of Freedom in 2007.",
        ],
        "keywords": ["harper", "lee"],
        "source": "nq",
    },
    {
        "id": "nq_f003",
        "query": "what is the largest ocean on earth",
        "answer": "Pacific Ocean",
        "docs": [
            "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 165 million square kilometers.",
            "The Pacific Ocean spans from the Arctic in the north to the Antarctic in the south.",
            "The Mariana Trench, located in the Pacific Ocean, is the deepest point on Earth at 11,034 meters.",
            "The Atlantic Ocean is the second-largest ocean, covering about 106 million square kilometers.",
            "The Pacific Ocean contains more than half of the world's oceanic water.",
            "Ferdinand Magellan was the first European to cross the Pacific Ocean in 1521.",
            "The Indian Ocean is the third-largest ocean, covering approximately 70 million square kilometers.",
            "Pacific Ocean temperatures vary from freezing near the poles to about 30°C near the equator.",
        ],
        "keywords": ["pacific"],
        "source": "nq",
    },
    {
        "id": "nq_f004",
        "query": "who invented the telephone",
        "answer": "Alexander Graham Bell",
        "docs": [
            "Alexander Graham Bell is widely credited with inventing the telephone, awarded the first patent in 1876.",
            "Bell was born on March 3, 1847, in Edinburgh, Scotland.",
            "On March 10, 1876, Bell made the first successful telephone call, speaking to his assistant Thomas Watson.",
            "Elisha Gray also developed a telephone device around the same time, leading to a famous patent dispute.",
            "The first telephone exchange was established in New Haven, Connecticut, in 1878.",
            "Bell's patent number 174,465 is often called the most valuable patent in history.",
            "Bell also founded what would eventually become AT&T.",
            "Thomas Edison improved on Bell's design by developing a better microphone.",
        ],
        "keywords": ["bell", "alexander"],
        "source": "nq",
    },
    {
        "id": "nq_f005",
        "query": "how many bones are in the human body",
        "answer": "206",
        "docs": [
            "The adult human body has 206 bones, while a newborn baby has around 270 to 300 bones.",
            "As children grow, many bones fuse together, reducing the total count to 206 by early adulthood.",
            "The femur, or thigh bone, is the longest and strongest bone in the human body.",
            "The smallest bones in the human body are the ossicles in the middle ear.",
            "The human skeleton provides structure, protects organs, enables movement, and produces blood cells.",
            "Bone marrow produces red blood cells, white blood cells, and platelets.",
            "The skull consists of 22 bones, including 8 cranial bones and 14 facial bones.",
            "Osteoporosis causes bones to become weak and brittle, increasing fracture risk.",
        ],
        "keywords": ["206"],
        "source": "nq",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _extract_keywords(answer: str) -> list[str]:
    stopwords = {"the", "a", "an", "is", "in", "of", "and", "or", "to", "was", "are", "it"}
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", answer.lower())
    kw = [t for t in tokens if t not in stopwords and len(t) > 1]
    return kw[:3] if kw else [answer.lower()[:20]]


def _load_from_file(path: Path, max_samples: int) -> list[dict] | None:
    """Load from local JSON file (downloaded by download_data.py)."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    print(f"  [data] Loaded {min(len(items), max_samples)}/{len(items)} items from {path.name}")
    return items[:max_samples]


def _load_hotpotqa_hf(max_samples: int) -> list[dict]:
    """Load HotpotQA from Hugging Face (requires internet)."""
    from datasets import load_dataset as hf_load
    ds = hf_load("hotpot_qa", "distractor", split="validation")
    items = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        ctx  = row["context"]
        docs = [f"{t}: {' '.join(s)}" for t, s in zip(ctx["title"], ctx["sentences"])]
        items.append({
            "id":       row["id"],
            "query":    row["question"],
            "answer":   row["answer"],
            "docs":     docs,
            "keywords": _extract_keywords(row["answer"]),
            "source":   "hotpotqa",
            "type":     row.get("type", ""),
            "level":    row.get("level", ""),
        })
    print(f"  [data] HotpotQA: {len(items)} samples from Hugging Face")
    return items


def _load_nq_hf(max_samples: int) -> list[dict]:
    """Load NQ from Hugging Face (requires internet)."""
    from datasets import load_dataset as hf_load
    ds = hf_load("nq_open", split="validation")
    items = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        answer = row["answer"][0] if isinstance(row["answer"], list) else row["answer"]
        items.append({
            "id":       f"nq_{i:05d}",
            "query":    row["question"],
            "answer":   answer,
            "docs":     [f"This question has the answer: {answer}."],
            "keywords": _extract_keywords(answer),
            "source":   "nq",
        })
    print(f"  [data] NQ: {len(items)} samples from Hugging Face")
    return items


def _load_source(source_name: str, file_path: Path, hf_loader, fallback: list[dict],
                 max_samples: int, prefer_real: bool) -> list[dict]:
    """Try file → HuggingFace → fallback."""
    # 1. Local file (best — real data, no internet needed after first download)
    if prefer_real:
        items = _load_from_file(file_path, max_samples)
        if items is not None:
            return items

    # 2. Hugging Face live
    if prefer_real:
        try:
            return hf_loader(max_samples)
        except Exception as e:
            print(f"  [data] {source_name} HuggingFace failed ({type(e).__name__}) — using fallback")

    # 3. Built-in fallback
    print(f"  [data] {source_name}: using built-in fallback ({min(len(fallback), max_samples)} items)")
    return fallback[:max_samples]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    source: DatasetSource = "both",
    max_samples: int = 10,
    prefer_real: bool = True,
) -> list[dict]:
    """
    Load evaluation items.

    Priority:
      1. data/hotpotqa_validation.json / data/nq_validation.json  (after running download_data.py)
      2. Hugging Face live download
      3. Built-in fallback

    Parameters
    ----------
    source      : "hotpotqa" | "nq" | "both"
    max_samples : Total items (split evenly for "both")
    prefer_real : If False, skip to built-in fallback immediately
    """
    half = max(1, max_samples // 2)
    results = []

    if source in ("hotpotqa", "both"):
        n = half if source == "both" else max_samples
        items = _load_source("HotpotQA", HOTPOTQA_FILE, _load_hotpotqa_hf,
                             _FALLBACK_HOTPOTQA, n, prefer_real)
        results.extend(items)

    if source in ("nq", "both"):
        n = half if source == "both" else max_samples
        items = _load_source("NQ", NQ_FILE, _load_nq_hf,
                             _FALLBACK_NQ, n, prefer_real)
        results.extend(items)

    return results[:max_samples]


def describe(items: list[dict]) -> str:
    from collections import Counter
    counts  = Counter(it["source"] for it in items)
    sources = [f"{src}: {cnt}" for src, cnt in counts.items()]
    return f"  Dataset: {len(items)} items  ({', '.join(sources)})"


# Alias for evaluate.py SSE endpoint (quick benchmark without download)
_EVAL_DATASET_FAST = _FALLBACK_HOTPOTQA[:3] + _FALLBACK_NQ[:3]
