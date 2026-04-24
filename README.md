# COMP41840 — AI for Health: Multi-Modal Breast Cancer Classification

**Team:** Liban · Thomas · Sergio Cordero *(add full surnames for Thomas and Sergio on the title page and anywhere official names are required)*  
**Module:** COMP41840 AI for Health — Dr. Aonghus Lawlor  
**Deadline:** 24 April 2026 @ 17:00  
**Repository:** https://github.com/libanmohamud-spec/comp41840-breast-cancer 

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
2. Expected structure after extraction (Kaggle zip often nests one extra `dataset/` folder):
   ```
   data/raw/
   ├── dataset/
   │   ├── dataset1/…/benign|malignant|normal/images/
   │   ├── dataset2/patient_history_dataset.csv
   │   └── dataset3/molecular_biomarker_dataset.csv
   └── …
   ```
3. **Tabular data:** When `dataset2` + `dataset3` are present, Tasks 3–4 use the merged Kaggle tables and the **same patient-level train/val/test split** as Task 2 (see `src/patient_split.py`). Otherwise place a standalone `data/clinical.csv` for the legacy path in `03_tabular_model.ipynb`.

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
│   └── patient_split.py        # Shared patient alignment + stratified split (Tasks 2–4)
├── results/
│   ├── figures/                # Plots, confusion matrices, Grad-CAM overlays
│   ├── patient_split.json      # Patient IDs per split (source of truth for cohort sizes)
│   └── metrics.json            # Stored evaluation metrics per model
├── outputs/                    # Optional: place final COMP41840_group_report.pdf / .docx here (see outputs/README.md)
└── report/
    └── report.pdf              # Alternative path for a local PDF build (ignored in git if under report/*.pdf)
```

---

## Task Summary

| # | Task | Owner | Status |
|---|------|-------|--------|
| 1 | Dataset Understanding & EDA | Sergio | ✅ Complete |
| 2 | Imaging Model (MONAI CNN) | Thomas | ✅ Complete |
| 3 | Tabular Model (XGBoost) | Liban | ✅ Complete |
| 4 | Fusion Model (Late Fusion) | Liban + Thomas | ✅ Complete |
| 5 | Explainability (Grad-CAM + SHAP) | Sergio | ✅ Complete |
| 6 | Research Questions | All | ✅ Complete |
| 7 | Optional: MedGemma comparison | Stretch | ⬜ |

---

## Milestones

| Date | Target | Status |
|------|--------|--------|
| 21 Apr (Tue) | Sprint start · Kaggle data extract · shared split (`patient_split.py`) · Task 3 underway | ✅ |
| 22 Apr (Wed) | Task 3 complete · metrics + figures + `tabular_xgb_model.json` / preprocess for fusion & SHAP | ✅ |
| 23 Apr (Thu) | Task 2 imaging · Task 4 fusion · `model_comparison.csv` · aligned test exports | ✅ |
| 23–24 Apr | Task 1 EDA · Task 5 explainability (Grad-CAM + SHAP) · Task 6 research answers in README | ✅ |
| 24 Apr (Fri, before 17:00) | Final `report/report.pdf` · per-member statements · repo tidy · optional Task 7 only if time | ⬜ |
| **24 Apr 17:00** | **Brightspace / module submission** | ⬜ |

---

## Submission checklist (group report + module brief)

- **PDF:** Dr. Lawlor’s brief asks for a PDF submission; treat the Word file as an editable source only.
- **Patient split sizes (for Table 1 / methods text):** from `results/patient_split.json` on the aligned cohort: **train 452 · validation 97 · test 98** (647 patients with both tabular rows and ultrasound). Replace any rounded **453 / 96** wording with these counts.
- **Limitations (esp. §9):** if you keep outcome/treatment columns in tabular features, the report should acknowledge possible leakage; if you re-run after dropping `LEAKAGE_COLS` in `03_tabular_model.ipynb`, tighten that section and refresh metrics from `results/metrics.json`.
- **Optional robustness pass (~1 h):** re-run `03_tabular_model.ipynb` → `02_imaging_model.ipynb` → `04_fusion_model.ipynb` after leakage drops; tune the fusion weight on **validation** probabilities (save val probs from notebooks 02 and 03 if you add that), then update the report with find-and-replace on metrics.
- **Per-member deliverable:** individual contribution statements (**2–5 pages each**) are separate from the group report; each author writes their own (technical work, notebooks, report role, challenges, team process).

Place final **`COMP41840_group_report.pdf`** (and optional `.docx`) under **`outputs/`** if you want a single hand-in folder alongside the repo; see `outputs/README.md`.

---

## Installation

```bash
git clone https://github.com/libanmohamud-spec/comp41840-breast-cancer.git
cd comp41840-breast-cancer
pip install -r requirements.txt
```

Requires Python 3.10+. GPU recommended for Task 2 (imaging model).

---

## Run Task 3 (Tabular)

1. **Preferred (fusion-aligned):** extract Kaggle data so `data/raw/dataset/dataset2/patient_history_dataset.csv` and `dataset3/molecular_biomarker_dataset.csv` exist.  
   **Fallback:** ensure `data/clinical.csv` exists (standalone survival-style CSV).
2. Open and run `notebooks/03_tabular_model.ipynb` end-to-end.
3. Expected outputs:
   - `results/metrics.json` with `tabular` metrics
   - `results/patient_split.json` (when using Kaggle multi-modal layout)
   - `results/tabular_classification_report.txt`
   - `results/tabular_test_probs.npy`
   - `results/tabular_test_labels.npy`
   - `results/tabular_xgb_model.json` and `results/tabular_preprocess.joblib` (for Task 5 SHAP)
   - `results/test_patient_ids.npy` (aligned mode; same order as Task 2 test exports)
   - `results/figures/tabular_confusion_matrix.png`
   - `results/figures/tabular_roc.png`
   - `results/figures/tabular_feature_importance.png`

## Run Task 2 (Imaging) and Task 4 (Fusion)

1. With the same Kaggle extraction as above, run `notebooks/02_imaging_model.ipynb` (uses `src/patient_split.py` for the **same stratified split** as Task 3 when multi-modal files are present).
2. Run `notebooks/04_fusion_model.ipynb` to combine `imaging_test_probs.npy` and `tabular_test_probs.npy`, update `results/metrics.json` with `fusion`, and write `results/model_comparison.csv`.

## Run Task 1 (EDA)

Run `notebooks/01_eda.ipynb` after extracting data. It writes figures under `results/figures/` (class distribution, samples, mask overlay when masks exist, tabular summaries).

## Run Task 5 (Explainability)

1. Run Tasks **2** and **3** first so `results/best_imaging_model.pt`, `tabular_xgb_model.json`, and `tabular_preprocess.joblib` exist.
2. Run `notebooks/05_explainability.ipynb`.  
   On some Windows setups, SHAP may pull optional Hugging Face TensorFlow stubs; if `TreeExplainer` fails, install `tf-keras` (`pip install tf-keras`) and retry.
3. Outputs include `gradcam_malignant.png`, `gradcam_benign.png`, `shap_beeswarm.png`, `shap_bar.png`, `shap_waterfall.png`, `shap_dependence.png`.

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

### Findings (aligned test set; see `results/metrics.json` and `results/model_comparison.csv`)

Numbers below come from the last full pipeline run saved in the repo. **Re-run notebooks 02–04** after longer imaging training or GPU runs to refresh metrics; the qualitative conclusions should be checked the same way.

**1. Ultrasound alone**  
Yes, to a useful but limited extent. The MONAI DenseNet121 imaging model reached **test AUC 0.77** and **F1 0.63** on the shared patient-level test split, so images carry real signal but are weaker than tabular scores in this setup (short CPU training in our default notebook profile will underestimate imaging performance).

**2. Clinical / genomic tabular alone**  
Stronger than imaging alone on the same split: **test AUC 0.97** and **F1 0.85** for XGBoost on merged patient history + molecular features. Tabular data is highly predictive for benign vs malignant in this cohort.

**3. Fusion vs single modality**  
Late fusion **improves ranking (AUC)** over imaging alone and slightly over tabular alone in our saved run: best fusion (**weighted**, AUC **0.971** vs tabular **0.966**, imaging **0.766**). **F1** is similar between tabular and fusion here (**0.866**), so the gain is clearest in discrimination (AUC) rather than balanced F1. Interpretation: combining modalities helps most when image quality or label noise limits the CNN; always validate on held-out data and calibration for deployment.

**4. Grad-CAM and SHAP plausibility**  
They give **auditable, feature-level rationales**, not a clinical diagnosis. **Grad-CAM** (`results/figures/gradcam_*.png`) highlights where the CNN focuses; reviewers should check whether saliency aligns with lesion margins / texture versus artefacts (shadowing, gain settings). **SHAP** (`results/figures/shap_*.png`) ranks which engineered variables drive the XGBoost malignant score; plausibility means comparing top features to oncologic intuition and known data leakage (e.g. surgery type, stage proxies). Formal clinical plausibility would require radiologist/pathologist review and external validation.

---

## Team

| Name | GitHub | Role |
|------|--------|------|
| Liban | @liban | Tabular model, Fusion model, Report |
| Thomas | @thomas | Imaging model, Fusion model, Report |
| Sergio | @sergio | EDA, Explainability, Report |
