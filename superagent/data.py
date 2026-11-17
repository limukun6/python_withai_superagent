"""Safe development-data readers and schema-based task detection."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class TaskFormatError(ValueError):
    pass


@dataclass
class DevelopmentTask:
    kind: str
    source: str
    X_train: Any
    y_train: np.ndarray
    X_val: Any
    y_val: np.ndarray
    spec: dict[str, Any]


def _member_by_suffix(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.replace("\\", "/").endswith(suffix)]
    if len(matches) != 1:
        raise TaskFormatError(f"Expected exactly one archive member ending in {suffix!r}; found {matches}")
    return matches[0]


def _read_csv(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    content = archive.read(member).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(content)))


def _instructions(archive: zipfile.ZipFile, names: list[str]) -> dict[str, str]:
    """Read task instructions as evidence; never read test members here."""
    readmes = [n for n in names if n.rsplit("/", 1)[-1].lower() == "readme.md"]
    if len(readmes) != 1:
        return {"instructions": "", "instruction_source": "unavailable"}
    member = readmes[0]
    return {"instructions": archive.read(member).decode("utf-8-sig", errors="replace"),
            "instruction_source": member}


def _load_archive(path: Path) -> DevelopmentTask:
    if not zipfile.is_zipfile(path):
        raise TaskFormatError(f"Only zip task packages are supported for now: {path}")

    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.startswith("__MACOSX/")]
        lower = {name.lower() for name in names}
        if any(name.endswith("c_train.npy") for name in lower):
            return _load_categorical_tabular(path, archive, names)
        if any(name.endswith("train.csv") for name in lower):
            return _load_pair_text(path, archive, names)
    raise TaskFormatError("No supported train/validation schema found in task archive.")


def _load_categorical_tabular(path: Path, archive: zipfile.ZipFile, names: list[str]) -> DevelopmentTask:
    required = {f"C_{split}.npy" for split in ("train", "val")} | {f"y_{split}.npy" for split in ("train", "val")}
    members = {suffix: _member_by_suffix(names, suffix) for suffix in required}
    arrays = {
        suffix: np.load(io.BytesIO(archive.read(member)), allow_pickle=True)
        for suffix, member in members.items()
    }
    info_members = [name for name in names if name.endswith("info.json")]
    info = json.loads(archive.read(info_members[0]).decode("utf-8")) if len(info_members) == 1 else {}
    x_train, x_val = arrays["C_train.npy"], arrays["C_val.npy"]
    y_train, y_val = arrays["y_train.npy"], arrays["y_val.npy"]
    if x_train.ndim != 2 or len(x_train) != len(y_train) or len(x_val) != len(y_val):
        raise TaskFormatError("Tabular feature/label dimensions do not agree.")
    spec = {
        "task_kind": "categorical_tabular_binary",
        "metric": "accuracy",
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "feature_count": int(x_train.shape[1]),
        "label_values": sorted(map(str, np.unique(y_train))),
        "source_metadata": info,
        **_instructions(archive, names),
        "test_accessed": False,
    }
    return DevelopmentTask("categorical_tabular", str(path), x_train, y_train, x_val, y_val, spec)


def _load_pair_text(path: Path, archive: zipfile.ZipFile, names: list[str]) -> DevelopmentTask:
    train = _read_csv(archive, _member_by_suffix(names, "train.csv"))
    val = _read_csv(archive, _member_by_suffix(names, "val.csv"))
    required = {"headline_A", "headline_B", "label_pairwise"}
    if not train or not val or not required.issubset(train[0]) or not required.issubset(val[0]):
        raise TaskFormatError("CSV task must contain headline_A, headline_B and label_pairwise.")
    x_train = [(row["headline_A"], row["headline_B"]) for row in train]
    x_val = [(row["headline_A"], row["headline_B"]) for row in val]
    y_train = np.asarray([int(row["label_pairwise"]) for row in train])
    y_val = np.asarray([int(row["label_pairwise"]) for row in val])
    spec = {
        "task_kind": "pair_text_binary",
        "metric": "accuracy",
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "input_fields": ["headline_A", "headline_B"],
        **_instructions(archive, names),
        "label_values": sorted(map(str, np.unique(y_train))),
        "test_accessed": False,
    }
    return DevelopmentTask("pair_text", str(path), x_train, y_train, x_val, y_val, spec)


def load_development_task(path: str | Path) -> DevelopmentTask:
    """Load only development splits.  Test members are never opened."""
    return _load_archive(Path(path).expanduser().resolve())


def load_prediction_features(path: str | Path, kind: str) -> Any:
    """Load test inputs only after development has frozen a selected model.

    This function never opens ``y_test.npy`` and, for CSV data, extracts only
    the declared input columns.  It deliberately has no evaluation function.
    """
    archive_path = Path(path).expanduser().resolve()
    if not zipfile.is_zipfile(archive_path):
        raise TaskFormatError(f"Only zip task packages are supported for now: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if not name.startswith("__MACOSX/")]
        if kind == "categorical_tabular":
            member = _member_by_suffix(names, "C_test.npy")
            return np.load(io.BytesIO(archive.read(member)), allow_pickle=True)
        if kind == "pair_text":
            member = _member_by_suffix(names, "test.csv")
            content = archive.read(member).decode("utf-8-sig")
            reader = csv.reader(io.StringIO(content))
            header = next(reader, None)
            if header is None or "headline_A" not in header or "headline_B" not in header:
                raise TaskFormatError("Pair-text test CSV must contain headline_A and headline_B.")
            left, right = header.index("headline_A"), header.index("headline_B")
            return [(row[left], row[right]) for row in reader]
    raise TaskFormatError(f"Prediction input does not match selected task kind: {kind}")

