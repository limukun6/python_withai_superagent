"""Dataset summaries passed to the planner and preserved in each run."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .data import DevelopmentTask


def _label_counts(values: np.ndarray) -> dict[str, int]:
    return {str(label): int(count) for label, count in Counter(values.tolist()).items()}


def profile(task: DevelopmentTask) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task_kind": task.kind,
        "train_label_counts": _label_counts(task.y_train),
        "validation_label_counts": _label_counts(task.y_val),
        "test_accessed": False,
    }
    if task.kind == "categorical_tabular":
        x_train = task.X_train
        unique_per_column = [int(len(set(map(str, x_train[:, index])))) for index in range(x_train.shape[1])]
        missing_per_column = [
            int(sum(str(value).strip().lower() in {"nan", "", "none", "null", "na", "?"} for value in x_train[:, index]))
            for index in range(x_train.shape[1])
        ]
        result.update({
            "feature_count": int(x_train.shape[1]),
            "unique_values_per_column": unique_per_column,
            "missing_values_per_column": missing_per_column,
            "constant_columns": [index for index, count in enumerate(unique_per_column) if count == 1],
            "duplicate_train_feature_rows": int(len(x_train) - len(set(map(tuple, x_train.tolist())))),
        })
    else:
        all_train_text = [text for pair in task.X_train for text in pair]
        lengths = sorted(len(text.split()) for text in all_train_text)
        result.update({
            "input_pair_count": len(task.X_train),
            "empty_text_count": int(sum(not text.strip() for text in all_train_text)),
            "word_count": {
                "min": lengths[0],
                "median": lengths[len(lengths) // 2],
                "p95": lengths[min(len(lengths) - 1, round(0.95 * len(lengths)))],
                "max": lengths[-1],
            },
        })
    return result

