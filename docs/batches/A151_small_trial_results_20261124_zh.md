# A151 条件小试结果：第一条路径被传输故障中断，未完成闭环

2026-11-24。按 `docs/batches/A151_small_trial_plan_20261124_zh.md` 执行，**在第一条路径的第一次 Flash 提案处遇到连接中断，按停止规则中止，没有重试，也没有开始第二条路径。**

## 结果

| 项 | 值 |
|---|---|
| 到达路径 | 仅 L0（`calibrated-v2`，分层 off） |
| L1 | **未启动**（`grounded-v1` + by-class 没有得到执行机会） |
| 实际分类器拟合 | **1**（目录首项 anchor `word_C0.05`），上限 3 |
| 学校 API 调用 | **1 次，失败且用量未知** |
| 失败类型 | `ConnectionError: RemoteDisconnected('Remote end closed connection without response')` |
| 墙钟 | 430.2 秒（其中约 420 秒在等待这次调用） |
| 自动重试 | 无 |
| 部分交付 | 保留了 trial_1 检查点（6660/8301 行），未做 refit |
| 独立进程预测 | `python -m superagent predict` 成功输出 **3232 行** |

## 账本变化（必须记录）

| 项 | 之前 | 之后 |
|---|---|---|
| 请求数 | 387 | **388** |
| 已知课程折算 | ¥7.0165495 | **¥7.0165495（未增加）** |
| 未解决 unknown | 12 | **13** |
| pending | 0 | 0 |
| 账本 SHA-256 | `5b1f5c58…` | `d671c424…` |

新 unknown 的请求 ID 是 `153d96d3-2fd3-4e87-bb32-f480d69b5919`（scope `a151_loop_off`）。用量未知，费用保持 null，**没有算成 0**。

## 这条 unknown 阻断了后续付费

`tools/accepted_usage_snapshot.check_snapshot` 会遍历账本里所有 `unknown` 行；只要出现不在已接受快照里的新行，就抛 `BudgetStateError('New or modified unknown request blocks accepted-history continuation')`。

因此：**在接受这份快照的前提下，下一次付费运行会被直接拒绝。** 需要用户决定怎么处理这一条（加入接受清单 / 向平台对账 / 其他），在那之前不应再发起真实请求。

## 已经得到的东西

- **零成本的 A151 机制证据不变**：`research/a151_issue_layering/` 的 5/5 oracle 与 6 条封存回放仍然有效。
- **部分交付链路可用**：模型冻结 → 独立进程预测 → 3232 行输出，这一段在失败路径上也走通了。
- **失败语义正确**：请求失败即停、不重试、未知用量不入账、保留原件。

## 没有证明的

- **闭环没有跑通**：事实 → 提案 → 执行 → 反馈 → 问题状态 → 重训这条链，断在"第一次提案"。
- **两条路径的对照没有发生**：L1 未运行，因此不能说分层在真实闭环里表现如何，也不能比较两条路径的 fit/调用/费用。
- 没有任何分数结论；`word_C0.05` 只是 anchor，不是选型结果。

## 需要进一步核实的

这次失败发生在发出请求后约 7 分钟才报 `RemoteDisconnected`，而客户端文档里的读超时是 60 秒。两种可能未能从现有证据区分：

1. 服务端长时间无数据后断开，客户端在多个内部读阶段累计等待；
2. 底层 HTTP 库对同一请求做了传输级重试。

应用层只记录到 **1 次** `unresolved_call`，但若要再次付费，建议先确认底层是否会自动重试同一请求——否则"失败不重试"的约定在传输层并不成立。

## 复跑

```powershell
Set-Location -LiteralPath 'D:\codex\python_withai\runs\integration'
$py = 'D:\codex\python_withai\runs\a79_encoder_assets\venv\Scripts\python.exe'
& $py -m scripts.run_a151_small_trial --task 'D:\codex\python_withai\data\released_task2.zip' `
  --ledger 'D:\codex\python_withai\runs\a129_B_ledger_return\private\ledger.sqlite' `
  --authorization 'research\a130_accepted_history_v2\accepted_history.json' `
  --api-key-file 'D:\codex\python_withai\api_key.txt' `
  --output 'D:\codex\python_withai\runs\a151_small_trial_attempt_03' --execute
```

注意：复跑会再产生调用；在当前 13 条 unknown 与旧快照不匹配的情况下，它会被账本守卫直接拒绝，而不是花钱。

证据：`D:\codex\python_withai\runs\a151_small_trial_attempt_02\`（`protocol.json`、`command_*.json`、`L0/run/run_outcome.json`、`completion.json`、`attempt_receipt.json`）。失败的第一次尝试（`ab-fixed` 与 3-fit 上限冲突，0 花费）保留在 `runs\a151_small_trial\`。
