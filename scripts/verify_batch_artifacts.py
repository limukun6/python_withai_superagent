"""Verify that a batch directory contains every artefact the handoff requires."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_BATCH = ["protocol.json", "summaries.json", "comparison.md"]
REQUIRED_RUN = [
    "summary.json",
    "trials.json",
    "selection.json",
    "events.jsonl",
    "llm_usage.jsonl",
    "data_profile.json",
    "split.json",
]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    batch = Path(sys.argv[1]).resolve()
    problems: list[str] = []

    for name in REQUIRED_BATCH:
        if not (batch / name).exists():
            problems.append(f"缺少批次文件 {name}")

    run_dirs = sorted(p for p in batch.iterdir() if p.is_dir())
    if not run_dirs:
        problems.append("批次里没有运行目录")
    for run in run_dirs:
        for name in REQUIRED_RUN:
            if not (run / name).exists():
                problems.append(f"{run.name}: 缺少 {name}")
        summary_path = run / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for key in (
                "valid_policy_run",
                "fallback_count",
                "usage_complete",
                "official_validation_accuracy",
                "llm_calls",
                "estimated_cost_rmb",
            ):
                if key not in summary:
                    problems.append(f"{run.name}: summary.json 缺字段 {key}")
            if summary.get("test_accessed"):
                problems.append(f"{run.name}: test_accessed 为真（违反协议）")

    print(f"批次: {batch.name}")
    print(f"运行目录: {len(run_dirs)}")
    if problems:
        print(f"问题 {len(problems)} 项:")
        for item in problems:
            print("  -", item)
        return 1
    print("全部必需文件与字段齐备；test_accessed 均为 false。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
