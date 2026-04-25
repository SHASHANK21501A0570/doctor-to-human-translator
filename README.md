# Results

Automated evaluation results for **Doctor-to-Human Translator** — a medical text simplification system built on Llama 3.1 8B with QLoRA fine-tuning.

All experiments are evaluated on four test domains: **SELLS**, **MedLane**, **Cochrane**, and **PLABA (OOD)**.  
Core metrics: **SARI**, **BERTScore F1**, **Entity F1**.

\---

## Folder structure

```
results/
├── exp0/                          # Copy-source baseline (lower bound)
├── exp1/                          # Zero-shot Llama 3.1 8B
├── exp2/
│   ├── v1/                        # SELLS-only fine-tune (original run)
│   │   └── predictions/           # Per-domain raw outputs
│   └── v2/                        # SELLS-only fine-tune (canonical, response-only masking)
│       └── predictions/
├── exp3/
│   ├── exp3a/                     # Mixed fine-tune, max\_len=1024
│   │   └── predictions/
│   └── exp3b/                     # Mixed fine-tune, max\_len=1536
│       └── predictions/
└── exp4/
    ├── exp4a/                     # BioBART zero-shot
    │   └── predictions/
    └── exp4c/                     # Flan-T5 zero-shot
        └── predictions/
```

\---

## Experiment reference

|Exp|Model|Training data|Key detail|
|-|-|-|-|
|Exp 0|—|—|Copy-source SARI floor (lower bound per domain)|
|Exp 1|Llama 3.1 8B Instruct|None|Zero-shot baseline — no fine-tuning|
|Exp 2 v1|Llama 3.1 8B + QLoRA|SELLS only|Original run — full-sequence loss (superseded)|
|Exp 2 v2|Llama 3.1 8B + QLoRA|SELLS only|Response-only masking — use this for comparisons|
|Exp 3a|Llama 3.1 8B + QLoRA|SELLS + MedLane + Cochrane|max\_len=1024, T=2 sampling — **best model**|
|Exp 3b|Llama 3.1 8B + QLoRA|SELLS + MedLane + Cochrane|max\_len=1536 — marginal difference vs 3a|
|Exp 4a|BioBART|None|Zero-shot domain-pretrained baseline|
|Exp 4c|Flan-T5 Large|None|Zero-shot instruction-tuned baseline|

> \*\*Exp 2 v1 vs v2:\*\* Both use SELLS-only QLoRA fine-tuning. The difference is the training loss computation — v1 computes loss over the full sequence including prompt tokens; v2 applies response-only masking so the model only learns from the assistant output. v2 is the correct setup for instruction fine-tuning and is the canonical result.

> \*\*Exp 3a vs 3b:\*\* Identical training data and config except max sequence length (1024 vs 1536). Differences across all domains are negligible — SARI delta < 0.5 on every domain — confirming 1024 is sufficient and no significant information is lost by truncating Cochrane examples.

> \*\*Exp 4b and 4d\*\* (fine-tuned BioBART and Flan-T5) were planned but not completed due to compute constraints.

\---

## Headline results

|Domain|Exp 1 (ZS)|Exp 2v2 (SELLS FT)|Exp 3a (Mixed FT)|Human ref|
|-|-|-|-|-|
|**SELLS** SARI|36.5|\~37.7|\~37.7|—|
|**MedLane** SARI|16.6|14.6|**55.53**|—|
|**Cochrane** SARI|34.1|lower|42.31|—|
|**PLABA** SARI|32.25|—|30.25|—|
|**MedLane** Entity F1|0.160|lower|**0.627**|0.674|
|**Cochrane** Entity F1|0.205|0.107|0.326|—|

Key findings:

* Mixed fine-tuning (Exp 3a) yields a **+38.9 SARI** gain on MedLane over zero-shot
* SELLS-only fine-tuning (Exp 2) collapses cross-domain Entity F1, especially on Cochrane (0.205 → 0.107) — dataset overfitting
* PLABA (OOD, unseen domain): Exp 3a slightly underperforms zero-shot (30.25 vs 32.25), indicating fine-tuning on domain-specific data slightly hurts generalization
* Exp 3a MedLane Entity F1 (0.627) reaches within 0.047 of human reference quality (0.674)
* Flan-T5 (Exp 4c) Entity F1 of 0.991 on SELLS is anomalous — it copies the source rather than simplifying (SARI = 7.60, just above the 7.05 copy-source floor)

\---

## File types

**Aggregate metric files** (`exp\*\_\*.json`) — domain-level averaged scores for SARI, BERTScore F1, Entity F1. Used for the main results tables in the paper.

**Per-example files** (`\*\_per\_example.json`) — raw score arrays per test example. Used for statistical testing (Wilcoxon signed-rank with Bonferroni correction) and confidence interval computation.

**Prediction files** (`predictions/`) — raw model output strings per test example, one file per domain (`preds\_cochrane.json`, `preds\_medlane.json`, `preds\_plaba.json`, `preds\_sells.json`). Used for qualitative error analysis.

\---

## Notes

* MIMIC-IV discharge summaries (used for human evaluation) are credentialed data via PhysioNet — raw text is **not committed**
* Copy-source floor (Exp 0) is the SARI score when the model outputs the input verbatim — any experiment scoring below this is actively degrading readability

