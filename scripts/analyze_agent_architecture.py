"""Analyse one agent-architecture batch and print a handoff-ready markdown report.

Usage:
    .venv\\Scripts\\python.exe scripts\\analyze_agent_architecture.py <batch-dir> [--out report.md]

Rules enforced here (from docs/deepseek_experiment_handoff.md):
* Missing runs never leave the denominator.
* Fallback runs stay visible and are not treated as valid LLM results.
* Paired comparisons use matched seeds and same task only.
* Per-trial "best search accuracy" is inner-search performance, never official validation.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path, PureWindowsPath

DEFAULT_STRATEGIES = ["random", "oneshot", "feedback", "reviewed"]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def fmt(value, digits=4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def mean_or_none(values):
    return statistics.mean(values) if values else None


def sd_or_none(values):
    return statistics.stdev(values) if len(values) > 1 else None


def load_batch(batch: Path) -> tuple[dict, list[dict]]:
    protocol = json.loads((batch / "protocol.json").read_text(encoding="utf-8"))
    summary_path = batch / "summaries.json"
    summaries = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    for row in summaries:
        run_dir = Path(row["run_directory"])
        if (batch / run_dir.name).is_dir():
            run_dir = batch / run_dir.name
        match = re.match(r"task(\d+)_", run_dir.name)
        row["_task"] = int(match[1]) if match and protocol.get("packages") else row["task_kind"]
        row["_trials"] = json.loads((run_dir / "trials.json").read_text(encoding="utf-8")) if (run_dir / "trials.json").exists() else []
        row["_events"] = read_jsonl(run_dir / "events.jsonl")
        row["_usage"] = read_jsonl(run_dir / "llm_usage.jsonl")
        row["_reasoning"] = sum(
            (u.get("raw_usage") or {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0
            for u in row["_usage"]
            if u.get("event") == "llm_call"
        )
        row["_transport_failures"] = sum(1 for e in row["_events"] if e.get("kind") == "llm_transport_failure")
        row["_http_failures"] = sum(1 for e in row["_events"] if e.get("kind") == "llm_http_failure")
        row["_finish_reasons"] = [e.get("finish_reason") for e in row["_events"] if e.get("kind") == "llm_response"]
        row["_null_contents"] = sum(1 for e in row["_events"] if e.get("kind") == "llm_response" and not isinstance(e.get("content"), str))
        row["_training_timeouts"] = sum(1 for t in row["_trials"] if t.get("status") == "timeout")
        row["_training_failed"] = sum(1 for t in row["_trials"] if t.get("status") == "failed")
    return protocol, summaries


def main() -> int:
    try:  # Windows consoles may default to GBK; never fail only because of encoding.
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    batch = Path(sys.argv[1]).resolve()
    out_path = None
    if "--out" in sys.argv:
        out_path = Path(sys.argv[sys.argv.index("--out") + 1]).resolve()

    protocol, rows = load_batch(batch)
    packages = protocol.get("packages", [])
    tasks = list(range(len(packages))) if packages else sorted({r["_task"] for r in rows})
    task_labels = {i: f"task{i}: {PureWindowsPath(package).stem}"
                   for i, package in enumerate(packages)} if packages else {task: task for task in tasks}
    seeds = sorted(set(protocol.get("seeds") or [r["seed"] for r in rows]))
    strategies = protocol.get("strategies") or DEFAULT_STRATEGIES
    indexed = {(r["_task"], r["strategy"], r["seed"]): r for r in rows}
    planned = [(task, strategy, seed) for task in tasks for strategy in strategies for seed in seeds]
    lines: list[str] = []
    add = lines.append

    add(f"# 批次分析：`{batch.name}`")
    add("")
    add(f"- 模型：`{protocol.get('model')}`；seed：{protocol.get('seeds')}；"
        f"max_trials={protocol.get('max_trials')}；time_limit={protocol.get('time_limit')}s；"
        f"trial_timeout={protocol.get('trial_timeout')}s；批次预算上限 RMB {protocol.get('batch_budget_rmb')}")
    add(f"- source_sha256：`{protocol.get('source_sha256')}`")
    add(f"- 计划运行数：{len(tasks)} 任务 × {len(seeds)} seed × {len(strategies)} 策略 = "
        f"{len(tasks) * len(seeds) * len(strategies)}；实际记录：{len(rows)}")
    add("")

    # ---------- 结果表 ----------
    add("## 1. 结果表（全部运行，缺失/失败不剔除）")
    add("")
    add("| 任务 | 策略 | seed | 选中候选 | search_acc | **val_acc** | valid | fallback | "
        "trials | failed_trials | 秒 | LLM调用 | tokens | RMB  | usage完整 |")
    add("|---|---|---:|---|---:|---:|:--:|---:|---:|---:|---:|---:|---:|---:|:--:|")
    for key in planned:
        task, strategy, seed = key
        row = indexed.get(key)
        if row is None:
            add(f"| {task_labels[task]} | {strategy} | {seed} | 缺少运行记录 | — | — | — | — | — | — | — | — | — | — | — |")
            continue
        add("| {task} | {strategy} | {seed} | `{cand}` | {search} | **{val}** | {valid} | {fb} | "
            "{trials}/{want} | {failed} | {secs} | {calls} | {tokens} | {cost} | {usage} |".format(
                task=task_labels[task], strategy=strategy, seed=seed,
                cand=row["selected_candidate"], search=fmt(row["search_accuracy"]),
                val=fmt(row["official_validation_accuracy"]),
                valid="✅" if row["valid_policy_run"] else "❌",
                fb=row["fallback_count"],
                trials=row["trials_attempted"], want=protocol.get("max_trials"),
                failed=row["failed_trials"], secs=fmt(row["elapsed_seconds"], 1),
                calls=row["llm_calls"], tokens=row["tokens"], cost=fmt(row["estimated_cost_rmb"], 6),
                usage="是" if row["usage_complete"] else "**否**"))
    add("")

    # ---------- 汇总 ----------
    add("## 2. 逐任务 × 策略汇总")
    add("")
    add("| 任务 | 策略 | 完成/计划 | 有效/计划 | 有效子集 val_acc 均值 | 全部运行 val_acc 均值 | "
        "有效子集样本SD | 平均秒 | tokens | RMB  | 训练失败 | 训练超时 |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for task in tasks:
        for strategy in strategies:
            group = [indexed[(task, strategy, seed)] for seed in seeds if (task, strategy, seed) in indexed]
            if not group:
                add(f"| {task_labels[task]} | {strategy} | 0/{len(seeds)} | 0/{len(seeds)} | N/A | N/A | N/A | — | — | — | — | — |")
                continue
            valid = [r for r in group if r["valid_policy_run"]]
            completed = [r for r in group if r["completed"]]
            vals_valid = [r["official_validation_accuracy"] for r in valid if r["official_validation_accuracy"] is not None]
            vals_all = [r["official_validation_accuracy"] for r in group if r["official_validation_accuracy"] is not None]
            add("| {task} | {strategy} | {comp}/{n} | {nv}/{n} | {mv} | {ma} | {sd} | {secs} | {tok} | {cost} | {tf} | {tt} |".format(
                task=task_labels[task], strategy=strategy, comp=len(completed), n=len(seeds),
                nv=len(valid), mv=fmt(mean_or_none(vals_valid)), ma=fmt(mean_or_none(vals_all)),
                sd=fmt(sd_or_none(vals_valid), 4),
                secs=fmt(mean_or_none([r["elapsed_seconds"] for r in group]), 1),
                tok=sum(r["tokens"] for r in group),
                cost=fmt(sum(r["estimated_cost_rmb"] for r in group), 6),
                tf=sum(r["_training_failed"] for r in group),
                tt=sum(r["_training_timeouts"] for r in group)))
    add("")

    # ---------- 配对比较 ----------
    add("## 3. 同 seed 配对比较（仅使用两侧都有效的配对）")
    add("")
    add("| 任务 | 对比 | 可用配对 | 每 seed 差值 | 均值 | 样本SD |")
    add("|---|---|---:|---|---:|---:|")
    v2_pairs = [("r", "f"), ("f", "o"), ("o", "d"), ("g", "r")]
    legacy_pairs = [("oneshot", "random"), ("feedback", "random"), ("reviewed", "random"), ("reviewed", "feedback")]
    pairs = [(a, b) for a, b in v2_pairs + legacy_pairs if a in strategies and b in strategies]
    for task in tasks:
        for a, b in pairs:
            deltas, detail = [], []
            for seed in seeds:
                ra = indexed.get((task, a, seed))
                rb = indexed.get((task, b, seed))
                if ra and rb and ra["valid_policy_run"] and rb["valid_policy_run"]:
                    va, vb = ra["official_validation_accuracy"], rb["official_validation_accuracy"]
                    if va is not None and vb is not None:
                        deltas.append(va - vb)
                        detail.append(f"{seed}:{va - vb:+.4f}")
            n_note = f"{len(deltas)}/{len(seeds)}"
            add(f"| {task_labels[task]} | {a} − {b} | {n_note} | {', '.join(detail) or '—'} | "
                f"{fmt(mean_or_none(deltas))} | {fmt(sd_or_none(deltas))} |")
    add("")

    # ---------- 诊断 ----------
    add("## 4. 诊断")
    add("")
    total_calls = sum(r["llm_calls"] for r in rows)
    total_cost = sum(r["estimated_cost_rmb"] for r in rows)
    transport = sum(r["_transport_failures"] for r in rows)
    http_fail = sum(r["_http_failures"] for r in rows)
    nulls = sum(r["_null_contents"] for r in rows)
    reasons = [fr for r in rows for fr in r["_finish_reasons"]]
    reasoning = sum(r["_reasoning"] for r in rows)
    tokens = sum(r["tokens"] for r in rows)
    incomplete_usage = [r for r in rows if not r["usage_complete"]]
    add(f"- LLM 调用总数：{total_calls}；估算费用合计 RMB {total_cost:.6f}")
    add(f"- 传输失败：{transport} 次；HTTP 非 2xx：{http_fail} 次；`content` 非字符串：{nulls} 次")
    add(f"- `finish_reason` 分布：{ {k: reasons.count(k) for k in sorted(set(reasons))} if reasons else '（无响应记录）'}")
    add(f"- 推理 token 占总 token 比例：{reasoning}/{tokens} = "
        f"{(reasoning / tokens * 100):.1f}%" if tokens else "- 推理 token：无")
    add(f"- `usage_complete=false` 的运行：{len(incomplete_usage)} 个"
        + ("" if not incomplete_usage else "（" + ", ".join(f"{r['task_kind']}/{r['strategy']}" for r in incomplete_usage) + "）"))
    add(f"- 训练超时：{sum(r['_training_timeouts'] for r in rows)} 次；训练失败：{sum(r['_training_failed'] for r in rows)} 次")
    add("")

    # ---------- 曲线 ----------
    add("## 5. 每轮 best inner search accuracy（内部搜索表现，不等于官方验证）")
    add("")
    header = "| 任务 | 策略 | seed | " + " | ".join(f"trial{i+1}" for i in range(protocol.get("max_trials", 3))) + " |"
    add(header)
    add("|---|---|---:|" + "---:|" * protocol.get("max_trials", 3))
    for row in sorted(rows, key=lambda r: (r["task_kind"], r["strategy"], r["seed"])):
        cells = []
        for i in range(protocol.get("max_trials", 3)):
            trial = next((t for t in row["_trials"] if t.get("trial") == i + 1), None)
            cells.append(fmt(trial.get("best_search_accuracy")) if trial else "—")
        add(f"| {row['task_kind']} | {row['strategy']} | {row['seed']} | " + " | ".join(cells) + " |")
    add("")

    text = "\n".join(lines) + "\n"
    if out_path:
        out_path.write_text(text, encoding="utf-8")
        print(f"written: {out_path}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
