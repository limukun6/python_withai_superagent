"""Safe development-data readers and schema-based task detection."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from .task_files import open_package


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


@dataclass
class TrainingTask:
    kind: str
    X_train: Any
    y_train: np.ndarray
    spec: dict[str, Any] = field(default_factory=dict)


def load_training_task(path: str | Path, *, read_declarations=False) -> TrainingTask:
    """Train-only capability for selection studies; never opens val or test."""
    with open_package(path,'train',shared=read_declarations) as archive:
        from .schema import read_manifest,load_manifest_split
        manifest=read_manifest(archive)
        if manifest is None:
            from .native_schema import infer_manifest
            manifest=infer_manifest(archive,read_declarations=read_declarations)
        if manifest is not None:
            kind,x,y,spec=load_manifest_split(archive,manifest,'train')
            if getattr(archive,'shared_table',None):
                spec['shared_table']=archive.shared_table
            if len(np.unique(y))<2:raise TaskFormatError('Training requires at least two classes')
            return TrainingTask(kind,x,y,spec)
        names = [n for n in archive.namelist() if not n.startswith("__MACOSX/")]
        if any(n.endswith("C_train.npy") for n in names):
            x = np.load(io.BytesIO(archive.read(_member_by_suffix(names,"C_train.npy"))), allow_pickle=True)
            y = np.load(io.BytesIO(archive.read(_member_by_suffix(names,"y_train.npy"))), allow_pickle=True)
            if x.ndim != 2 or len(x) != len(y):
                raise TaskFormatError("Training feature/label dimensions do not agree")
            from .task_readme import legacy_declarations
            spec=legacy_declarations(archive,'categorical_tabular',{})
            if read_declarations:
                from .native_schema import description
                from .table_input import declared_metric
                metric=declared_metric(description(archive))
                if metric:
                    spec['metric']=metric
                    spec.setdefault('goal_sources',{})['primary_metric']='declared_task_readme'
            return TrainingTask("categorical_tabular", x, y, spec)
        rows = _read_csv(archive, _member_by_suffix(names,"train.csv"))
        if not rows or not {"headline_A","headline_B","label_pairwise"}.issubset(rows[0]):
            raise TaskFormatError("Unsupported training CSV schema")
        from .task_readme import legacy_declarations
        return TrainingTask("pair_text", [(r["headline_A"],r["headline_B"]) for r in rows],
                            np.asarray([int(r["label_pairwise"]) for r in rows]),legacy_declarations(archive,'pair_text',{}))


def _member_by_suffix(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.replace("\\", "/").endswith(suffix)]
    if len(matches) != 1:
        raise TaskFormatError(f"Expected exactly one archive member ending in {suffix!r}; found {matches}")
    return matches[0]


def _read_csv(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    content = archive.read(member).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(content)))


def _load_archive(path: Path) -> DevelopmentTask:
    if not zipfile.is_zipfile(path):
        raise TaskFormatError(f"Only zip task packages are supported for now: {path}")

    with zipfile.ZipFile(path) as archive:
        from .schema import read_manifest,load_manifest_split
        manifest=read_manifest(archive)
        if manifest is None:
            from .native_schema import infer_manifest,add_native_split
            manifest=infer_manifest(archive)
            if manifest is not None:add_native_split(archive,manifest,'validation')
        if manifest is not None:
            kind,x,y,spec=load_manifest_split(archive,manifest,'train')
            x_val=y_val=None
            if 'validation' in manifest.get('splits',{}):
                val_kind,x_val,y_val,val_spec=load_manifest_split(archive,manifest,'validation')
                if kind!=val_kind:raise TaskFormatError('Train and validation schemas differ')
                if spec.get('feature_shape')!=val_spec.get('feature_shape'):
                    raise TaskFormatError('Train and validation feature dimensions differ')
            if len(np.unique(y))<2:raise TaskFormatError('Training requires at least two classes')
            spec.update(train_rows=len(y),validation_rows=len(y_val) if y_val is not None else 0,
                        provided_validation_available=y_val is not None,class_count=len(np.unique(y)),
                        task_kind=kind+'_classification',model_kind=kind)
            return DevelopmentTask(kind,str(path),x,y,x_val,y_val,spec)
        names = [name for name in archive.namelist() if not name.startswith("__MACOSX/")]
        lower = {name.lower() for name in names}
        if any(name.endswith("c_train.npy") for name in lower):
            return _load_categorical_tabular(path, archive, names)
        if any(name.endswith("train.csv") for name in lower):
            return _load_pair_text(path, archive, names)
    raise TaskFormatError("No supported train/validation schema found in task archive.")


def _load_categorical_tabular(path: Path, archive: zipfile.ZipFile, names: list[str]) -> DevelopmentTask:
    required={'C_train.npy','y_train.npy'}
    present=[any(n.endswith(suffix) for n in names) for suffix in ('C_val.npy','y_val.npy')]
    if any(present) and not all(present):raise TaskFormatError('Validation features and labels must both be supplied')
    if all(present):required.update({'C_val.npy','y_val.npy'})
    members = {suffix: _member_by_suffix(names, suffix) for suffix in required}
    arrays = {
        suffix: np.load(io.BytesIO(archive.read(member)), allow_pickle=True)
        for suffix, member in members.items()
    }
    info_members = [name for name in names if name.endswith("info.json")]
    info = json.loads(archive.read(info_members[0]).decode("utf-8")) if len(info_members) == 1 else {}
    x_train, x_val = arrays["C_train.npy"], arrays.get("C_val.npy")
    y_train, y_val = arrays["y_train.npy"], arrays.get("y_val.npy")
    if x_train.ndim != 2 or len(x_train) != len(y_train) or x_val is not None and len(x_val) != len(y_val):
        raise TaskFormatError("Tabular feature/label dimensions do not agree.")
    spec = {
        "task_kind": "categorical_tabular_binary",
        "metric": "accuracy",
        "goal_sources": {"primary_metric": "legacy_adapter_default"},
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)) if y_val is not None else 0,
        'provided_validation_available':y_val is not None,
        "feature_count": int(x_train.shape[1]),
        "label_values": sorted(map(str, np.unique(y_train))),
        "source_metadata": info,
        "test_accessed": False,
    }
    from .task_readme import legacy_declarations
    return DevelopmentTask("categorical_tabular", str(path), x_train, y_train, x_val, y_val,
                           legacy_declarations(archive,'categorical_tabular',spec))


def _load_pair_text(path: Path, archive: zipfile.ZipFile, names: list[str]) -> DevelopmentTask:
    train = _read_csv(archive, _member_by_suffix(names, "train.csv"))
    val = _read_csv(archive, _member_by_suffix(names, "val.csv")) if any(n.endswith('val.csv') for n in names) else None
    required = {"headline_A", "headline_B", "label_pairwise"}
    if not train or not required.issubset(train[0]) or val is not None and (not val or not required.issubset(val[0])):
        raise TaskFormatError("CSV task must contain headline_A, headline_B and label_pairwise.")
    x_train = [(row["headline_A"], row["headline_B"]) for row in train]
    x_val = [(row["headline_A"], row["headline_B"]) for row in val] if val is not None else None
    y_train = np.asarray([int(row["label_pairwise"]) for row in train])
    y_val = np.asarray([int(row["label_pairwise"]) for row in val]) if val is not None else None
    spec = {
        "task_kind": "pair_text_binary",
        "metric": "accuracy",
        "goal_sources": {"primary_metric": "legacy_adapter_default"},
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)) if y_val is not None else 0,
        'provided_validation_available':y_val is not None,
        "input_fields": ["headline_A", "headline_B"],
        "label_values": sorted(map(str, np.unique(y_train))),
        "test_accessed": False,
    }
    from .task_readme import legacy_declarations
    return DevelopmentTask("pair_text", str(path), x_train, y_train, x_val, y_val,
                           legacy_declarations(archive,'pair_text',spec))


def load_development_task(path: str | Path) -> DevelopmentTask:
    """Load only development splits.  Test members are never opened."""
    return _load_archive(Path(path).expanduser().resolve())


def load_prediction_features(path: str | Path, kind: str, frozen_spec=None) -> Any:
    """Load test inputs only after development has frozen a selected model.

    This function never opens ``y_test.npy`` and, for CSV data, extracts only
    the declared input columns.  It deliberately has no evaluation function.
    """
    archive_path = Path(path).expanduser().resolve()
    with open_package(archive_path,'test',contract=(frozen_spec or {}).get('shared_table')) as archive:
        from .schema import read_manifest,load_manifest_split
        manifest=read_manifest(archive)
        if frozen_spec and frozen_spec.get('input_schema'):
            if manifest is None:
                # A legacy two-column CSV records its fields and README semantics without input_format.
                legacy_csv = (kind == 'pair_text' and frozen_spec.get('task_kind') == 'pair_text_binary'
                              and frozen_spec.get('input_fields') == ['headline_A', 'headline_B'])
                if 'input_format' in frozen_spec or not legacy_csv:
                    from .native_schema import prediction_manifest
                    manifest=prediction_manifest(archive,frozen_spec)
            else:
                expected=frozen_spec['input_schema']
                for key in ('modality','columns','pair_label_one','pair_semantics','image_preprocessing'):
                    if manifest.get(key)!=expected.get(key):raise TaskFormatError('Prediction schema differs from frozen training schema')
            if frozen_spec.get('image_decoder_policy'):
                from .image_files import DECODER_POLICY
                from importlib.metadata import version
                if frozen_spec['image_decoder_policy']!=DECODER_POLICY:
                    raise TaskFormatError('Image decoder differs from the frozen training contract')
                if frozen_spec.get('image_decoder_version')!=version('Pillow'):
                    raise TaskFormatError('Pillow version differs from the frozen image decoder')
                if manifest.get('splits',{}).get('test',{}).get('format') not in ('csv','image_folder'):
                    raise TaskFormatError('Image-file prediction must use the frozen CSV path interface')
        if manifest is not None:
            detected,x,_,spec=load_manifest_split(archive,manifest,'test',include_labels=False)
            if detected!=kind:raise TaskFormatError('Prediction modality does not match frozen model')
            if frozen_spec and 'feature_shape' in frozen_spec and frozen_spec['feature_shape']!=spec.get('feature_shape'):
                raise TaskFormatError('Prediction dimensions differ from frozen training schema')
            return x
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

