# ContextGuard — Multi-Agent MCP Framework for Context Integrity Verification

**Group 2** — Diyar Buyuksahin · Etem Tolga Erten · Süleyman Kılıç · Andrew Mabuto

> A hallucination-resistant LLM reasoning framework using multi-agent validation and MCP governance.

---

## Proje Yapısı

```
AdvancedSoftware/
├── main.py                        # Ana pipeline (demo çalıştırma)
├── evaluate.py                    # 4-koşul benchmark (§6 Experimental Study)
├── requirements.txt
├── README.md
└── contextguard/
    ├── __init__.py
    ├── agents.py                  # 4 ajan implementasyonu
    ├── mcp_governance.py          # MCP governance (RBAC + schema + loglama)
    ├── context_store.py           # Context versioning & traceability  ← YENİ
    ├── attribution.py             # Sentence-level attribution scoring ← YENİ
    └── metrics.py                 # HR, CIS, CUE, RS metrikleri
```

---

## Kurulum

```powershell
# 1. Klasöre git
cd "C:\...\AdvancedSoftware"

# 2. Bağımlılıkları kur
pip install -r requirements.txt

# 3. GROQ API key ayarla (ücretsiz: https://console.groq.com)
$env:GROQ_API_KEY = "gsk_..."   # PowerShell
# veya
set GROQ_API_KEY=gsk_...        # CMD
```

---

## Çalıştırma

### Demo (tek sorgu, tam pipeline)
```powershell
python main.py
```

### Benchmark (4-koşul karşılaştırma, §6)
```powershell
python evaluate.py
```

---

## Mimari

```
Query + Documents
      │
      ▼
[Agent 1] ContextRetrievalAgent
  - TF-IDF cosine ranking
  - Context Dilution prevention (min_score threshold)
  - ContextStore commit (versioned snapshot)
      │
      ▼  version_id
[Agent 2] ReasoningAgent
  - Groq LLM (llama-3.3-70b-versatile)
  - Strict context-grounded prompt
  - ContextStore checkout (consistency guarantee)
      │
      ▼
[Agent 3] GroundingValidatorAgent
  ├── Layer 1: TF-IDF cosine similarity
  ├── Layer 2: LLM-based claim extraction & verdict
  └── Layer 3: Sentence-level attribution scoring
      │
      ▼
[Agent 4] AdversarialTesterAgent
  - LLM-generated (not hardcoded) adversarial injections
  - Poisoned context → ContextStore commit
  - Robustness = 1 - sim(answer, adversarial_doc)
      │
      ▼
[Metrics] HR · CIS · CUE · RS · Overall
[MCP Log] All events logged with RBAC enforcement
```

---

## Metrikler (Proposal §5)

| Metrik | Formül | Hedef |
|--------|--------|-------|
| **HR** (Hallucination Rate)        | `1 - CIS`                                  | < τ = 30%  |
| **CIS** (Context Integrity Score)  | `0.7 × LLM_CIS + 0.3 × Attribution`       | > 70%      |
| **CUE** (Context Util. Efficiency) | `# used_docs / # retrieved_docs`           | > δ = 50%  |
| **RS** (Robustness Score)          | `1 - avg(sim_to_adversarial)`              | > 70%      |
| **Overall**                        | `0.3×(1-HR) + 0.3×CIS + 0.2×CUE + 0.2×RS`| > 70%      |

**Formal Risk Model (Proposal §3):**
```
P(H) = 1 - ∏ P(context supports output)
Constraints: P(H) < τ  AND  CUE > δ
```

---

## MCP Governance (Proposal §4)

Her ajan yalnızca tanımlı rollerine ait aksiyonları gerçekleştirebilir:

| Agent                   | İzin Verilen Aksiyonlar |
|-------------------------|------------------------|
| ContextRetrievalAgent   | read_documents, rank_context, commit_context |
| ReasoningAgent          | read_context, call_llm, checkout_context |
| GroundingValidatorAgent | read_context, read_answer, call_llm, compute_similarity, compute_attribution |
| AdversarialTesterAgent  | inject_adversarial, read_context, call_llm, commit_context |

---

## 4-Koşul Deneysel Karşılaştırma (Proposal §6)

| Koşul | Sistem | Özellikler |
|-------|--------|------------|
| A | Baseline RAG | Sıralama yok, doğrulama yok |
| B | Single-Agent LLM | Context yok, parametrik bellek |
| C | Multi-Agent / no MCP | Retrieval + LLM, governance yok |
| **D** | **ContextGuard** | **MCP + multi-agent + 3-katmanlı doğrulama** |

---

## Yeni Özellikler (Tamamlanan)

- **`context_store.py`** — SHA-256 hash ile context versioning, tamper detection
- **`attribution.py`** — Sentence-level attribution map (hangi cümle hangi kaynaktan)
- **LLM-generated adversarial docs** — Query-specific, hardcoded değil
- **Schema validation** — MCP input type checking
- **Formal risk model** — P(H) hesaplama ve constraint violation raporlama
- **`format_report()`** — Güzel konsol raporu
- **Hypothesis testing** — Benchmark'ta otomatik sonuç değerlendirmesi
