"""Shared patient-level alignment and train/val/test split for imaging + tabular fusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _history_path(data_root: Path) -> Path | None:
    p = data_root / "dataset" / "dataset2" / "patient_history_dataset.csv"
    return p if p.exists() else None


def _molecular_path(data_root: Path) -> Path | None:
    p = data_root / "dataset" / "dataset3" / "molecular_biomarker_dataset.csv"
    return p if p.exists() else None


def kaggle_multimodal_available(data_root: Path) -> bool:
    return _history_path(data_root) is not None and _molecular_path(data_root) is not None


def build_aligned_manifest(data_root: Path) -> pd.DataFrame:
    """
    One row per patient with tabular features, label, and ultrasound image path.
    Requires extracted Kaggle layout under data_root/dataset/...
    """
    hp = _history_path(data_root)
    mp = _molecular_path(data_root)
    if hp is None or mp is None:
        raise FileNotFoundError(
            "Kaggle multi-modal clinical files not found. Expected "
            f"{data_root}/dataset/dataset2/patient_history_dataset.csv and "
            f"{data_root}/dataset/dataset3/molecular_biomarker_dataset.csv"
        )

    hist = pd.read_csv(hp)
    mol = pd.read_csv(mp)
    hist["patient_id"] = hist["Patient ID"].astype(str).str.strip()
    mol["patient_id"] = mol["Patient ID"].astype(str).str.strip()
    mol = mol.drop(columns=["Patient ID"], errors="ignore")

    tab = hist.merge(mol, on="patient_id", how="inner")
    tab = tab[tab["class"].astype(str).str.lower().isin(["benign", "malignant"])].copy()
    tab["class"] = tab["class"].astype(str).str.lower()
    tab["label_enc"] = (tab["class"] == "malignant").astype(int)

    by_pid: dict[str, tuple[str, int]] = {}
    for cls, lbl in [("benign", 0), ("malignant", 1)]:
        paths: list[Path] = []
        direct = data_root / cls / "images"
        if direct.exists():
            paths.extend(sorted(direct.glob("*.png")))
        for img_dir in sorted((data_root / "dataset").glob(f"dataset*/{cls}/images")):
            paths.extend(sorted(img_dir.glob("*.png")))
        for f in paths:
            pid = f.stem
            rec = (str(f.resolve()), lbl)
            if pid in by_pid and by_pid[pid][1] != lbl:
                raise ValueError(f"Conflicting image labels for patient {pid}")
            by_pid[pid] = rec

    if not by_pid:
        raise FileNotFoundError(f"No ultrasound PNGs found under {data_root}.")

    img = pd.DataFrame(
        [{"patient_id": k, "image_path": v[0], "img_folder_label": v[1]} for k, v in sorted(by_pid.items())]
    )

    man = tab.merge(img, on="patient_id", how="inner")
    if (man["label_enc"] != man["img_folder_label"]).any():
        bad = man.loc[man["label_enc"] != man["img_folder_label"], "patient_id"].head(5).tolist()
        raise ValueError(f"Image folder label disagrees with clinical class for patients: {bad}")

    man = man.drop(columns=["img_folder_label"], errors="ignore")
    return man.reset_index(drop=True)


def assign_split_manifest(manifest: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Stratified 70/15/15 patient split; adds string column `split` in {train,val,test}."""
    ids = manifest["patient_id"].values
    y = manifest["label_enc"].values

    id_train, id_temp, y_train, y_temp = train_test_split(
        ids, y, test_size=0.3, stratify=y, random_state=seed
    )
    id_val, id_test, _, _ = train_test_split(
        id_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )

    split_map = {pid: "train" for pid in id_train}
    split_map.update({pid: "val" for pid in id_val})
    split_map.update({pid: "test" for pid in id_test})

    out = manifest.copy()
    out["split"] = out["patient_id"].map(split_map)
    if out["split"].isna().any():
        raise RuntimeError("Split assignment failed for some rows.")
    return out


def save_patient_split(manifest: pd.DataFrame, results_dir: Path) -> None:
    """Persist patient IDs per split for debugging and downstream checks."""
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": sorted(manifest.loc[manifest["split"] == "train", "patient_id"].tolist()),
        "val": sorted(manifest.loc[manifest["split"] == "val", "patient_id"].tolist()),
        "test": sorted(manifest.loc[manifest["split"] == "test", "patient_id"].tolist()),
    }
    (results_dir / "patient_split.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sort_test_by_patient(patient_ids, probs, preds, labels) -> tuple:
    """Order parallel test arrays by patient_id so imaging and tabular exports align."""
    pid = np.asarray(patient_ids)
    order = np.argsort(pid)
    return (
        np.asarray(probs, dtype=np.float64)[order],
        np.asarray(preds)[order],
        np.asarray(labels)[order],
        pid[order],
    )
