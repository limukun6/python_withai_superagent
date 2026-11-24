# A151 端到端验收结果：两条路径都在压缩菜单下跑完

2026-11-24。按 `docs/batches/A151_e2e_acceptance_plan_20261124_zh.md` 执行。**两条路径全部完成，6 次 fit、3 次 Flash、¥0.055246、没有新增 unknown。**

## 结果

| 项 | L0（`calibrated-v2`，分层 off） | L1（`grounded-v1` + `by-class`） |
|---|---|---|
| 搜索试验 | 2（anchor `word_C0.05` → `word_char_C0.05`） | 2（anchor `word_C0.05` → `word_C0.1`） |
| 实际 fit | 3（2 搜索 + 1 refit） | 3（2 搜索 + 1 refit） |
| Flash 调用 | **1** | **2**（1 次被拒 + 1 次修复后接受） |
| 选中候选 | `word_char_C0.05` | `word_C0.05`（anchor） |
| 内部主指标 | 0.6520 | 0.6228 |
| 验证集 | 0.7011 | 0.6755 |
| 独立进程预测 | **3,232 行** | **3,232 行** |
| `requested_planner_completed` | true | true |
| 自动重试 | 无 | 无 |

合计：**6 fit / 3 调用 / ¥0.055246**，低于预登记上沿（6 fit / 6 调用）。

## L1 的 grounded 链路确实被执行

| 事件 | 内容 |
|---|---|
| 第 1 次提案 | **被拒**：`Assertion is unknown: no_same_penalized_parameterization_and_observed_reference` |
| 第 2 次提案（语义修复） | **接受**：8 条断言全部 `confirmed / exact_bound_fact` |
| 遗留 issue | 0 条 |

也就是说：压缩表示下，`grounded-v1` 的"断言校验 → 拒绝 → 一次语义修复 → 接受"整条链路都真实跑到了，`--issue-layering by-class` 也在该路径上生效。

## 验收项核对

| 判据 | 结果 |
|---|---|
| 两条路径 `requested_planner_completed` | 均 true |
| fit 2–3 且 attempted == completed | 均 3/3 |
| 调用 ≤3 且等于账本增量 | L0=1、L1=2，逐条对账一致 |
| 新增 unknown / pending | **0 / 0** |
| 独立进程预测产出 | 两条各 3,232 行 |
| 运行前后源码一致 | 是 |

## 账本

| 项 | 之前 | 之后 |
|---|---|---|
| 请求数 | 397 | **400** |
| known 课程折算 | ¥7.14048 | **¥7.195726** |
| unknown | 16 | **16（未增加）** |
| pending | 0 | 0 |
| 账本 SHA-256 | `d83dde66…` | **`9031531e…`** |

`a151_loop_off` 与 `a151_loop_by_class` 各自 1 条与 2 条完整结算的请求；16 条接受清单不变。

## 能说什么、不能说什么

**能说**
1. 压缩菜单表示下，正式入口能自主完成"提案 → 执行 → 真实反馈 → 复核 → 重训 → 独立进程预测"。
2. `grounded-v1` 在压缩表示下可用：断言核验、拒绝、一次语义修复、接受都发生了。
3. 两条路径的实际资源是 6 fit / 3 调用 / ¥0.055246。

**不能说**
1. **不能说提分**：单任务、单划分、每路径一次运行，没有对照分布。L1 反而停在 anchor，因为第 2 次试验（0.6216）低于 anchor（0.6228）。
2. **不能说压缩省时**：没有同条件下 full vs compact 的配对计时。
3. **不能说通用性**：`released_task2` 是已见任务。
4. **不能说 L1 优于 L0**：两条路径的候选不同只是调度与提案不同，不构成收益证据；本轮样本量为 1。

## 与 A147 的关系

A147 的结论是"改变动作不保证更好"。本轮 L1 在同一 seed、同一预算下把第二次探索换成 `word_C0.1`（0.6216），低于 L0 同一步的 `word_char_C0.05`（0.6520）。这是**又一次同向的负结果**，不是新结论，也不改写 A147。

## 复跑

```powershell
Set-Location -LiteralPath 'D:\codex\python_withai\runs\integration'
$py = 'D:\codex\python_withai\runs\a79_encoder_assets\venv\Scripts\python.exe'
& $py -m scripts.run_a151_small_trial --task 'D:\codex\python_withai\data\released_task2.zip' `
  --ledger 'D:\codex\python_withai\runs\a129_B_ledger_return\private\ledger.sqlite' `
  --authorization 'research\a130_accepted_history_v3\accepted_history.json' `
  --api-key-file 'D:\codex\python_withai\api_key.txt' --menu-view compact `
  --output '...\a151_e2e_acceptance_attempt_02' --execute
```

证据：`research/a151_e2e_acceptance/{protocol,completion,command_L0,command_L1}.json`、`L0/run/`、`L1/run/`、`predictions_L0.csv`、`predictions_L1.csv`。
