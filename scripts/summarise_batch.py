"""Print batch totals and per-task/per-strategy aggregates."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

batch = Path(sys.argv[1]).resolve()
rows = json.loads((batch / "summaries.json").read_text(encoding="utf-8"))

print(f"batch: {batch.name}")
print(f"runs={len(rows)} valid={sum(1 for r in rows if r['valid_policy_run'])}")
print(f"total_cost_rmb={sum(r['estimated_cost_rmb'] for r in rows):.6f}")
print(f"total_calls={sum(r['llm_calls'] for r in rows)}")
print(f"total_tokens={sum(r['tokens'] for r in rows)}")
print(f"total_seconds={sum(r['elapsed_seconds'] for r in rows):.0f}")
print(f"training_timeouts={sum(r['failed_trials'] for r in rows)}")
print()
print(f"{'task':22s}{'strategy':10s}{'n':>3s}{'mean':>8s}{'sd':>8s}{'tokens':>8s}{'rmb':>9s}{'sec':>7s}")
for task in sorted({r["task_kind"] for r in rows}):
    for strategy in ("random", "oneshot", "feedback", "reviewed"):
        group = [r for r in rows if r["task_kind"] == task and r["strategy"] == strategy]
        if not group:
            continue
        values = [r["official_validation_accuracy"] for r in group]
        sd = statistics.stdev(values) if len(values) > 1 else float("nan")
        print(
            f"{task:22s}{strategy:10s}{len(group):3d}{statistics.mean(values):8.4f}{sd:8.4f}"
            f"{sum(r['tokens'] for r in group):8d}{sum(r['estimated_cost_rmb'] for r in group):9.4f}"
            f"{sum(r['elapsed_seconds'] for r in group):7.0f}"
        )
