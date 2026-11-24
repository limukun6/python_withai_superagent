# A149 片段探测：模型能不能产出合法计划片段

2026-11-24。前置：片段机制已接入并通过零成本验证（`batches/A149_fragment_wiring_20261124_zh.md`），但片段内容是手工构造的。本计划只回答一个问题：**真实模型在收到片段指令后，会不会返回合法片段？**

## 1 设计

单条路径、单次运行即可回答，不追求统计：

| 项 | 值 |
|---|---|
| 任务 | `sms_spam_single_text.zip`（14 候选，菜单小，full 视图安全） |
| seed | 707 |
| 调度 | `ab-v1`，`--fit-limit 4`（anchor + 最多 2 次探索 + refit） |
| 调用上限 | 3（`--llm-request-limit 3`） |
| 契约 | `calibrated-v2`（片段字段与它兼容） |
| 视图 | 默认 full、反馈 compact |
| 开关 | `--plan-fragment advisory`（新） |
| 账本 | 唯一真实账本 + `accepted_history_v3`，`task_budget_id=a149_fragment_probe` |

上沿：**4 fit / 最多 3 次调用**，预计 ¥0.03。

## 2 观测项（预登记）

1. 第 2 次决策的响应里有没有 `plan` 字段；
2. 有的话，`validate_fragment` 是否通过（不通过则记 `plan_fragment_rejected` 及原因）；
3. 通过的话，第 3 次决策是否由片段执行（`plan_fragment_used`）而**不产生新调用**；
4. 实际调用数：1（片段命中）还是 2（未命中）。

## 3 判定

- **机制可用**：至少出现一次 `plan_fragment_offered`，且（命中或被后续证据作废）都算机制走通。
- **省调用**：第 3 次决策由片段执行，实际调用 1 次。
- **不可用**：模型从不返回 plan，或返回的 plan 一律被拒。

不做显著性、不比分数。若模型完全不产出 plan，则记录并停在这里，不调提示词反复试。

## 4 停止规则

任一路径失败或出现新 unknown/pending 立即停止，不重试。

## 5 交付物

- 本文 + `research/a149_fragment_probe/`
- 报告：`docs/batches/A149_fragment_probe_results_20261124_zh.md`
