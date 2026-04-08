# COMP41840 — AI for Health: Multi-Modal Breast Cancer Classification

**Team:** Liban · Thomas · Sergio  
**Module:** COMP41840 AI for Health — Dr. Aonghus Lawlor  
**Deadline:** 24 April 2026 @ 17:00  

---

## Project Overview

This project builds a multi-modal breast cancer classification system using the [Multi-Modal Breast Cancer Dataset](https://www.kaggle.com/datasets/ajithdari/multi-modal-breast-cancer-dataset) (780 patients, 3 classes: benign / malignant / normal).

The system combines ultrasound image classification (MONAI CNN) with clinical/tabular data classification (XGBoost) into a late-fusion model, with full explainability via Grad-CAM and SHAP.

---

## Dataset Setup

> **Do not commit raw data to this repo.**

1. Download from Kaggle:
   ```bash
   kaggle datasets download -d ajithdari/multi-modal-breast-cancer-dataset
   unzip multi-modal-breast-cancer-dataset.zip -d data/raw/
   ```
2. Expected structure after extraction:
   ```
   data/raw/
   ├── benign/
   │   ├── images/
   │   └── masks/
   ├── malignant/
   │   ├── images/
   │   └── masks/
   └── normal/
       ├── images/
       └── masks/
   ```
3. A `data/clinical.csv` tabular file should also be present in the archive.

---

## Repo Structure

```
comp41840-breast-cancer/
├── README.md
├── requirements.txt
├── data/
│   └── README.md              # Data download instructions (no raw files)
├── notebooks/
│   ├── 01_eda.ipynb            # Task 1 — EDA (Sergio)
│   ├── 02_imaging_model.ipynb  # Task 2 — CNN imaging model (Thomas)
│   ├── 03_tabular_model.ipynb  # Task 3 — XGBoost tabular model (Liban)
│   ├── 04_fusion_model.ipynb   # Task 4 — Late fusion (Liban + Thomas)
│   └── 05_explainability.ipynb # Task 5 — Grad-CAM + SHAP (Sergio)
├── src/
│   ├── dataset.py              # Shared data loaders
│   ├── imaging.py              # CNN model definition & training
│   ├── tabular.py              # XGBoost pipeline
│   ├── fusion.py               # Late fusion logic
│   └── explain.py              # Grad-CAM + SHAP utilities
├── results/
│   ├── figures/                # Plots, confusion matrices, Grad-CAM overlays
│   └── metrics.json            # Stored evaluation metrics per model
└── report/
    └── report.pdf              # Final group report (not tracked in git)
```

---

## Task Summary

| # | Task | Owner | Status |
|---|------|-------|--------|
| 1 | Dataset Understanding & EDA | Sergio | ⬜ |
| 2 | Imaging Model (MONAI CNN) | Thomas | ⬜ |
| 3 | Tabular Model (XGBoost) | Liban | ⬜ |
| 4 | Fusion Model (Late Fusion) | Liban + Thomas | ⬜ |
| 5 | Explainability (Grad-CAM + SHAP) | Sergio | ⬜ |
| 6 | Research Questions | All | ⬜ |
| 7 | Optional: MedGemma comparison | Stretch | ⬜ |

---

## Milestones

| Date | Target |
|------|--------|
| 11 Apr | Repo set up · Data downloaded · EDA drafted |
| 15 Apr | Imaging model trained & evaluated |
| 17 Apr | Tabular model done · Models compared |
| 19 Apr | Fusion model · Explainability complete |
| 21 Apr | Report draft circulated for review |
| 23 Apr | Final report + individual statements ready |
| **24 Apr 17:00** | **Submission** |

---

## Installation

```bash
git clone <repo-url>
cd comp41840-breast-cancer
pip install -r requirements.txt
```

Requires Python 3.10+. GPU recommended for Task 2 (imaging model).

---

## GitHub Workflow

- One branch per task: `task/01-eda`, `task/02-imaging`, `task/03-tabular`, etc.
- PRs require **at least one reviewer** before merging to `main`
- Use **GitHub Issues** to track blockers and decisions
- Commit regularly — the lecturer checks commit history, PRs, and issue tracking

---

## Research Questions (Task 6)

1. Can ultrasound images alone predict benign vs malignant breast cancer?
2. How well do clinical/genomic variables perform on their own?
3. Does combining image and tabular data in a fusion model improve predictive performance?
4. Do Grad-CAM and SHAP provide clinically plausible explanations?

---

## Team

| Name | GitHub | Role |
|------|--------|------|
| Liban | @liban | Tabular model, Fusion model, Report |
| Thomas | @thomas | Imaging model, Fusion model, Report |
| Sergio | @sergio | EDA, Explainability, Report |
