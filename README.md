# COMP41840 — AI for Health: Multi-Modal Breast Cancer Classification

**Team:** Liban Mohamud · Thomas Gordon · Sergio Cordero 
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
3. **Tabular data:** Tasks 3–4 use the merged Kaggle tables (`dataset2` + `dataset3`) with the **same patient-level train/val/test split** as Task 2 (see `src/patient_split.py`).

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

## Installation

```bash
git clone https://github.com/libanmohamud-spec/comp41840-breast-cancer.git
cd comp41840-breast-cancer
pip install -r requirements.txt
```

Requires Python 3.10+. GPU recommended for Task 2 (imaging model).

---

## Run Task 3 (Tabular)

1. Extract Kaggle data so `data/raw/dataset/dataset2/patient_history_dataset.csv` and `dataset3/molecular_biomarker_dataset.csv` exist.
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

---

## Research Questions (Task 6)

1. Can ultrasound images alone predict benign vs malignant breast cancer?
2. How well do clinical/genomic variables perform on their own?
3. Does combining image and tabular data in a fusion model improve predictive performance?
4. Do Grad-CAM and SHAP provide clinically plausible explanations?

### Findings (aligned test set; see `results/metrics.json` and `results/model_comparison.csv`)

The final reported numbers come from the corrected leakage-controlled pipeline saved in `results/metrics.json`. Earlier exploratory runs produced much higher tabular performance, but those results were contaminated by post-diagnostic and cohort-confounded variables. The final analysis removes those variables before training.

**1. Ultrasound alone**  
Yes. The MONAI DenseNet121 ultrasound model reached **test AUC 0.924** and **F1 0.794** on the shared patient-level held-out test split. This makes imaging the strongest single modality in the corrected pipeline and suggests that the ultrasound images contain substantial benign-versus-malignant diagnostic signal.

**2. Clinical / genomic tabular data alone**  
The corrected XGBoost tabular model reached **test AUC 0.723** and **F1 0.523** after removing post-diagnostic treatment/outcome variables and the cohort confound. This is much lower than the earlier leakage-contaminated tabular result, but it is more clinically defensible because the model no longer relies on variables unavailable at diagnosis time.

**3. Fusion versus single modality**  
Late fusion produced the best overall test performance, with the weighted fusion variant reaching **AUC 0.939** and **F1 0.824**. This improves modestly over imaging alone, but the gain is small and should be interpreted cautiously because the test set contains only 98 patients. Any fusion weight should ideally be tuned on validation predictions rather than selected using the test set.

**4. Grad-CAM and SHAP plausibility**  
Grad-CAM and SHAP are used as auditing tools rather than proof of clinical correctness. Grad-CAM checks whether the CNN focuses on lesion regions or image artefacts such as probe markers and calipers. SHAP shows that, after leakage removal, the tabular model relies on more plausible tumour and molecular features rather than survival, treatment or cohort variables.

Final corrected metrics:

| Model | AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Imaging DenseNet121 | 0.924 | 0.794 | 0.750 | 0.844 |
| Tabular XGBoost | 0.723 | 0.523 | 0.515 | 0.531 |
| Fusion Average | 0.936 | 0.776 | 0.743 | 0.813 |
| Fusion Weighted | 0.939 | 0.824 | 0.778 | 0.875 |
| Fusion Stacking | 0.936 | 0.730 | 0.742 | 0.719 |

---

## Team

| Name | GitHub | Role |
|------|--------|------|
| Liban | @liban | Tabular model, Fusion model, Report |
| Thomas | @thomas | Imaging model, Fusion model, Report |
| Sergio | @sergio | EDA, Explainability, Report |
