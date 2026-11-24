# A151 单调用机制探测：压缩菜单后一次合法提案返回成功

2026-11-24。按 `docs/batches/A151_menu_probe_plan_20261124_zh.md` 执行，**1 次真实 Flash、0 次分类器拟合**，账本未产生新的 unknown。

## 结果

| 项 | 值 |
|---|---|
| 结论 | `accepted=true`，模型返回合法提案 |
| 选中候选 | `word_char_C0.1` |
| 候选在原始 29 个菜单内 | 是 |
| 参数与目录一致 | 是 |
| 输出契约 | `calibrated-v2` 原样，11 个字段齐全 |
| `purpose` / 分类器声明 | `improve` / `logistic`（与目录一致） |
| 请求数 | **1**（`max_requests=1`，无重试） |
| 分类器拟合 | **0** |
| 输入 / 输出 token | 4,724 / 1,297 |
| `finish_reason` | `stop` |
| 耗时 | 52.7 秒（首块 4.55 秒，SSE `data_events` 444） |
| 课程折算 | **¥0.009912** |

失败的原调用是 6,821 输入 token、417 秒零事件、无 usage；这一次是 4,724 输入、52.7 秒、444 个数据事件、正常结束。**表示压缩把输入 token 减少约 30.7%**（字节 −35.039%），并且这一次模型自己停下来了。

## 账本

| 项 | 之前 | 之后 |
|---|---|---|
| 请求数 | 396 | **397** |
| known 课程折算 | ¥7.130568 | **¥7.14048** |
| unknown | 16 | **16（未增加）** |
| pending | 0 | 0 |

这一次用量完整结算，**没有新增 unknown**，所以 16 条接受清单仍然有效、闸门仍然开着。

## 这是什么、不是什么

**是**：同一个模型、同一个输出契约、同一个既有状态下，把候选菜单换成无损压缩表示后，能返回一次合法提案。

**不是**：
- 不是"闭环跑通"：没有执行、没有反馈、没有复核、没有重训、没有交付预测。这只是一次提案。
- 不是"压缩降低了推理时间"的受控结论：只有一次成功样本，和原失败相比变量不止一个（原失败还叠加了 417 秒断流，原因至今未定位）。
- 不是"grounded 可用"：本次是 `calibrated-v2`，grounding 开关没开。
- 不是 token 收益的证明：字节减少 35% 只导致输入 token 减少约 30.7%，没有做多次测量。

## 一次失败尝试也保留

`research/a151_menu_probe/` 是第一次执行：运行目录未创建导致 `FileNotFoundError`，**在发出任何请求之前就失败**，账本未变（0 请求）。修复后重跑为 `attempt_02`。原件保留，未覆盖。

## 下一步（按复核第 4 节）

只有机制成功后才能另立端到端验收，且仍每路径 ≤3 次调用。建议顺序：

1. 先把表示开关纳入 A151 小试计划的正式配置，重跑一次**两条路径**的端到端验收（L0 `calibrated-v2`、L1 `grounded-v1` + `by-class`），每路径 ≤3 次调用、≤3 fit。
2. 若 L1 在压缩表示下仍不可用，就按复核第 5 节把 grounded 标为受限实验开关，正式路径保留本地规划。

## 复跑

```powershell
Set-Location -LiteralPath 'D:\codex\python_withai\runs\integration'
$py = 'D:\codex\python_withai\runs\a79_encoder_assets\venv\Scripts\python.exe'
# 只校验（0 请求）
& $py -m scripts.run_a151_menu_probe --events 'D:\codex\python_withai\runs\a151_small_trial_attempt_02\L0\run\events.jsonl' `
  --output 'D:\codex\python_withai\runs\a151_menu_probe_validate2'
# 真实执行（会花 1 次调用，输出目录必须是新目录）
& $py -m scripts.run_a151_menu_probe --events 'D:\codex\python_withai\runs\a151_small_trial_attempt_02\L0\run\events.jsonl' `
  --ledger 'D:\codex\python_withai\runs\a129_B_ledger_return\private\ledger.sqlite' `
  --authorization 'research\a130_accepted_history_v3\accepted_history.json' `
  --api-key-file 'D:\codex\python_withai\api_key.txt' --output '...\a151_menu_probe_attempt_03' --execute
```

证据：`research/a151_menu_probe_attempt_02/{protocol,results,completion}.json`、`run/events.jsonl`、`run/llm_usage.jsonl`、`compact_payload.json`。
