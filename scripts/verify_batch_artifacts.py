"""Verify that a batch directory contains every artefact the handoff requires."""
from __future__ import annotations

import json
import re
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

    protocol_path = batch / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else {}
    summary_path = batch / "summaries.json"
    summaries = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    summary_dirs = [Path(row["run_directory"]).name for row in summaries]
    if len(summary_dirs) != len(set(summary_dirs)):
        problems.append("批次汇总含重复运行目录")

    run_dirs = sorted(p for p in batch.iterdir() if p.is_dir())
    if not run_dirs:
        problems.append("批次里没有运行目录")
    completed_keys = set()
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
            match = re.match(r"task(\d+)_", run.name)
            if match and run.name in summary_dirs:
                key = (int(match[1]), summary["strategy"], summary["seed"])
                if key in completed_keys:
                    problems.append(f"{run.name}: 重复的任务/策略/seed 组合")
                completed_keys.add(key)

    if all(protocol.get(field) for field in ("packages", "strategies", "seeds")):
        expected = {(i, strategy, seed)
                    for i in range(len(protocol["packages"]))
                    for strategy in protocol["strategies"]
                    for seed in protocol["seeds"]}
        for i, strategy, seed in sorted(expected - completed_keys):
            problems.append(f"缺少计划运行 task{i}/{strategy}/seed={seed}")
        for i, strategy, seed in sorted(completed_keys - expected):
            problems.append(f"存在计划外运行 task{i}/{strategy}/seed={seed}")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
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
