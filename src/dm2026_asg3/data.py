from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LABEL_COLUMNS = ("Label", "label", "activity", "Activity", "target", "Target", "y")
ID_COLUMNS = ("Id", "id", "file_id", "FileId", "fileId", "file")
REQUIRED_SIGNAL_COLUMNS = ("mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z")


@dataclass(frozen=True)
class WindowRecord:
    path: Path
    file_id: int | str
    user_id: str
    label: int | None
    split: str


def normalize_id(value) -> int | str:
    if pd.isna(value):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return int(text[:-2])
    if text.isdigit():
        return int(text)
    return text


def find_csv_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


def infer_user_id(path: Path, split_root: Path) -> str:
    try:
        rel = path.relative_to(split_root)
    except ValueError:
        return "unknown_user"
    if len(rel.parts) > 1:
        return rel.parts[0]
    return "unknown_user"


def _first_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = set(columns)
    for candidate in candidates:
        if candidate in cols:
            return candidate
    lower_to_original = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        match = lower_to_original.get(candidate.lower())
        if match is not None:
            return match
    return None


def infer_file_id(df: pd.DataFrame, path: Path) -> int | str:
    id_col = _first_existing_column(df.columns, ID_COLUMNS)
    if id_col is not None:
        values = df[id_col].dropna().unique()
        if len(values) == 1:
            value = values[0]
            return normalize_id(value)
    stem = path.stem
    return normalize_id(stem)


def infer_embedded_label(df: pd.DataFrame) -> int | None:
    label_col = _first_existing_column(df.columns, LABEL_COLUMNS)
    if label_col is None:
        return None
    values = df[label_col].dropna().unique()
    if len(values) == 0:
        return None
    if len(values) > 1:
        raise ValueError(f"Expected one window label, found {values.tolist()}")
    return int(values[0])


def load_label_table(data_dir: Path) -> dict[int | str, int]:
    candidates = (
        "train_labels.csv",
        "labels.csv",
        "train_label.csv",
        "y_train.csv",
        "metadata.csv",
    )
    for name in candidates:
        path = data_dir / name
        if not path.exists():
            continue
        labels = pd.read_csv(path)
        id_col = _first_existing_column(labels.columns, ID_COLUMNS)
        label_col = _first_existing_column(labels.columns, LABEL_COLUMNS)
        if id_col is None or label_col is None:
            continue
        return dict(zip(labels[id_col].map(normalize_id), labels[label_col].astype(int)))
    return {}


def load_window_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_SIGNAL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if "index" in df.columns:
        df = df.sort_values("index", kind="mergesort").reset_index(drop=True)
    return df


def discover_records(data_dir: Path, split: str) -> list[WindowRecord]:
    split_root = data_dir / split
    label_table = load_label_table(data_dir) if split == "train" else {}
    records: list[WindowRecord] = []
    for path in find_csv_files(split_root):
        df = load_window_csv(path)
        file_id = infer_file_id(df, path)
        label = infer_embedded_label(df)
        if label is None and file_id in label_table:
            label = int(label_table[file_id])
        records.append(
            WindowRecord(
                path=path,
                file_id=file_id,
                user_id=infer_user_id(path, split_root),
                label=label,
                split=split,
            )
        )
    return records


def load_sample_submission(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "sample_submission.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download the Kaggle data and place it under {data_dir}."
        )
    sub = pd.read_csv(path)
    if "Id" not in sub.columns or "Label" not in sub.columns:
        raise ValueError("sample_submission.csv must contain Id and Label columns")
    return sub


def validate_training_records(records: list[WindowRecord]) -> None:
    if not records:
        raise FileNotFoundError("No training CSV files found under data/raw/train")
    missing = [r.path for r in records if r.label is None]
    if missing:
        preview = "\n".join(str(p) for p in missing[:5])
        raise ValueError(
            "Training labels were not found for some windows. Add train_labels.csv "
            "or include a Label column in each train CSV. Examples:\n"
            f"{preview}"
        )
