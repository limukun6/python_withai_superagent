"""Small append-only records used to make autonomous runs auditable."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.events_path = run_dir / "events.jsonl"
        self.usage_path = run_dir / "llm_usage.jsonl"

    def event(self, kind: str, **payload: Any) -> None:
        record = {"time_utc": utc_now(), "kind": kind, **payload}
        with self.events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_json(self, name: str, value: Any) -> None:
        (self.run_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def write_experiments(self, rows: list[dict[str, Any]]) -> None:
        fields = [
            "experiment_id", "family", "status", "validation_accuracy",
            "train_seconds", "reason", "params",
        ]
        with (self.run_dir / "experiments.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    **row,
                    "params": json.dumps(row.get("params", {}), ensure_ascii=False),
                })

    def initialize_usage_ledger(self) -> None:
        # The first release runs a deterministic local candidate portfolio.
        # Keeping a ledger from day one prevents silent accounting gaps later.
        record = {
            "time_utc": utc_now(),
            "event": "ledger_initialized",
            "provider": None,
            "model": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "note": "No LLM call was made in this local baseline run; billing is disabled.",
        }
        self.usage_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    def usage(self, **payload: Any) -> None:
        record = {"time_utc": utc_now(), **payload}
        with self.usage_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

