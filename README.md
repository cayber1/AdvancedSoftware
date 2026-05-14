# ContextGuard — AdvancedSoftware

**Group 2** — Diyar Buyuksahin · Etem Tolga Erten · Süleyman Kılıç · Andrew Mabuto

---

## Klasör Yapısı

```
AdvancedSoftware\
├── main.py
├── evaluate.py
├── requirements.txt
├── README.md
└── contextguard\
    ├── __init__.py
    ├── agents.py
    ├── mcp_governance.py
    └── metrics.py
```

---

## Kurulum ve Çalıştırma (Windows PowerShell)

### 1. Klasöre git
```powershell
cd "C:\Users\Admin\OneDrive\Masaüstü\AdvancedSoftware"
```

### 2. Kütüphaneleri yükle
```powershell
pip install -r requirements.txt
```

### 3. Demo çalıştır
```powershell
python main.py
```

### 4. Benchmark çalıştır
```powershell
python evaluate.py
```

> API key gerekmez — Groq key kodun içinde tanımlı.

---

## Metrikler

| Metrik | Açıklama |
|--------|----------|
| **HR**  | Hallucination Rate |
| **CIS** | Context Integrity Score |
| **CUE** | Context Utilization Efficiency |
| **RS**  | Robustness Score |
