# Doctor-to-Human Translator

**Simplifying clinical notes into patient-friendly language using fine-tuned LLMs.**

NLP-based medical text simplification system that translates clinical notes into easy-to-understand patient language using LLMs and deep learning models.
A research project from Northeastern University's CS 6120 (Natural Language Processing) course. We fine-tune Llama 3.1 8B with QLoRA on a mixed medical simplification corpus and evaluate against zero-shot and domain-pretrained baselines across four test domains, including out-of-distribution clinical discharge summaries from MIMIC-IV.

---

## Research question

> *How effectively can modern instruction-tuned LLMs simplify complex clinical notes into patient-friendly language while preserving medical meaning and avoiding hallucinations?*

---



## Model & approach

**Primary model:** Llama 3.1 8B Instruct + QLoRA (4-bit quantization, LoRA adapters)  
**Training data:** SELLS + MedLane + Cochrane, mixed using temperature sampling at T=2  
**Sampling ratio:** SELLS ~59%, MedLane ~27%, Cochrane ~14%  
**Baselines:** Zero-shot Llama 3.1 8B, BioBART-Large (zero-shot), Flan-T5-Large (zero-shot)  
**Evaluation domains:** SELLS, MedLane, Cochrane, PLABA (OOD)  
**Human evaluation:** 50 MIMIC-IV discharge summaries, rated across Simplicity, Accuracy, Completeness, Fluency  

---

## Repository structure

```
doctor-to-human-translator/
├── data/                          # Data pipeline (data branch)
│   ├── notebooks/
│   │   ├── EDA_notebook.ipynb         # Exploratory data analysis
│   │   ├── mimic_calibration.ipynb    # MIMIC-IV discharge summary preprocessing
│   │   └── Exp_00.ipynb               # Reference quality audit + copy-source baseline
│   └── EDA_visuals/                   # Cochrane token lengths, MedLane readability, etc.
│
├── baseline-models/               # Baseline experiments (baseline-models branch)
│   ├── Exp_01.ipynb                   # Zero-shot Llama 3.1 8B baseline
│   ├── Exp_02_2.ipynb                 # SELLS-only QLoRA fine-tune (v2, canonical)
│   ├── exp_4a.ipynb                   # BioBART-Large zero-shot
│   └── exp_4c.ipynb                   # Flan-T5-Large zero-shot
│
├── primary-model/                 # Primary model experiments (primary-model branch)
│   ├── Exp_3a.ipynb                   # Mixed fine-tune, max_len=1024 — best model
│   └── Exp_3b.ipynb                   # Mixed fine-tune, max_len=1536
│
└── results/                       # All evaluation outputs (results branch)
    ├── exp0/                          # Copy-source SARI floor
    ├── exp1/                          # Zero-shot results
    ├── exp2/
    │   ├── v1/                        # SELLS-only (original run, superseded)
    │   │   └── predictions/
    │   └── v2/                        # SELLS-only (response-only masking, canonical)
    │       └── predictions/
    ├── exp3/
    │   ├── exp3a/                     # Mixed 1024 — best model
    │   │   └── predictions/
    │   └── exp3b/                     # Mixed 1536
    │       └── predictions/
    ├── exp4/
    │   ├── exp4a/                     # BioBART zero-shot
    │   │   └── predictions/
    │   └── exp4c/                     # Flan-T5 zero-shot
    │       └── predictions/
    └── human_eval/                    # Annotation files, IAA scores
```

---

## Experiments

| Exp | Model | Training data | Notes |
|-----|-------|---------------|-------|
| Exp 0 | — | — | Copy-source SARI floor — lower bound per domain |
| Exp 1 | Llama 3.1 8B Instruct | None | Zero-shot — primary baseline |
| Exp 2 v1 | Llama 3.1 8B + QLoRA | SELLS only | Full-sequence loss — superseded by v2 |
| Exp 2 v2 | Llama 3.1 8B + QLoRA | SELLS only | Response-only masking — canonical SELLS-only result |
| Exp 3a | Llama 3.1 8B + QLoRA | SELLS + MedLane + Cochrane | max_len=1024, T=2 — **best model** |
| Exp 3b | Llama 3.1 8B + QLoRA | SELLS + MedLane + Cochrane | max_len=1536 — marginal diff vs 3a |
| Exp 4a | BioBART-Large | None | Zero-shot domain-pretrained baseline |
| Exp 4c | Flan-T5-Large | None | Zero-shot instruction-tuned baseline |

> **Exp 2 v1 vs v2:** Same SELLS data and QLoRA setup. The difference is loss masking — v1 computes loss over the full sequence including prompt tokens; v2 uses response-only masking so the model only learns from the assistant output. v2 is the correct setup and the canonical result.

> **Exp 3a vs 3b:** Identical setup except max sequence length. SARI delta is < 0.5 across all domains — validates that 1024 is sufficient and no meaningful information is lost truncating Cochrane examples.

> **Exp 4b and 4d** (fine-tuned BioBART and Flan-T5 respectively) were planned but not completed due to compute constraints — exhausted free GPU units on Colab/Kaggle running Exp 3b on A100.

---

## Key results

| Domain | Exp 1 (Zero-shot) | Exp 2v2 (SELLS FT) | Exp 3a (Mixed FT) |
|--------|------------------|--------------------|-------------------|
| **SELLS** SARI | 36.5 | ~37.7 | ~37.7 |
| **MedLane** SARI | 16.6 | 14.6 | **55.53** |
| **Cochrane** SARI | 34.1 | lower | 42.31 |
| **PLABA** SARI (OOD) | 32.25 | — | 30.25 |
| **MedLane** Entity F1 | 0.160 | lower | **0.627** (ref: 0.674) |
| **Cochrane** Entity F1 | 0.205 | 0.107 | 0.326 |

**Key findings:**
- Exp 3a achieves a **+38.9 SARI** gain on MedLane over zero-shot
- SELLS-only fine-tuning causes cross-domain Entity F1 collapse (Cochrane: 0.205 → 0.107) — dataset overfitting
- Exp 3a MedLane Entity F1 (0.627) reaches within 0.047 of human reference quality (0.674)
- OOD generalization (PLABA): Exp 3a marginally underperforms zero-shot (30.25 vs 32.25)
- Exp 3a vs 3b differences are negligible — max_len=1024 is sufficient

---

## Datasets

| Dataset | Type | Size | Use |
|---------|------|------|-----|
| SELLS | Sentence-level biomedical simplification | ~168K pairs | Primary training data |
| MedLane | Clinical note simplification | ~34K pairs | Training + evaluation |
| Cochrane | Review-style medical simplification | ~4.7K pairs | Training + evaluation |
| PLABA | Biomedical abstract simplification | — | OOD test only |
| MIMIC-IV | Real EHR discharge summaries | 50 samples | Human evaluation only |

> **MIMIC-IV is credentialed data via PhysioNet** — raw discharge summaries are not committed to this repository. Access requires completion of CITI training and a signed data use agreement at [physionet.org](https://physionet.org).

---

## Setup

```bash
git clone https://github.com/SHASHANK21501A0570/doctor-to-human-translator.git
```

**Key dependencies:** `transformers`, `peft`, `bitsandbytes`, `trl`, `datasets`, `evaluate`, `bert_score`, `nltk`

**Hardware:** Tested on T4, V100, and A100 GPUs (Colab/Kaggle).

---

## Branches

| Branch | Contents |
|--------|----------|
| `main` | Root README, LICENSE, .gitignore |
| `data` | EDA notebook, MIMIC calibration, Exp_00, EDA visuals |
| `baseline-models` | Exp_01, Exp_02_2, exp_4a, exp_4c notebooks |
| `primary-model` | Exp_3a, Exp_3b notebooks |
| `results` | All evaluation metric files, predictions, human eval |

---

## Citation

If you use this work, please cite:

```
@misc{doctor-to-human-2026,
  title={Doctor-to-Human Translator: Simplifying Clinical Notes with Instruction-Tuned LLMs},
  author={Ning-Hsuan Tseng, Theresa Coleman, Kishan Prajapati, Shashank Kadiyala, Siddharth Chalasani},
  year={2026},
  institution={Northeastern University, Khoury College of Computer Sciences}
}
```

---

## License

MIT License — see `LICENSE` for details.

> **Note on data:** MIMIC-IV data is subject to its own PhysioNet credentialed access agreement and is not covered by this license.
