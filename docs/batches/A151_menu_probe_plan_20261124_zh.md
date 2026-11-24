# A151 单调用机制计划：压缩菜单表示，先验一次合法提案

2026-11-24，基线上游复核 `docs/batches/A151_blocker2_codex_resolution_20261124_zh.md`。本计划按该复核第 4 节的顺序写成，**尚未执行任何真实调用**。

## 1 只回答一个问题

把候选菜单换成无损压缩表示后，**同一个模型、同一个输出契约、同一个既有状态**能不能返回一次合法提案。

不回答能不能提分、不回答压缩是否降低推理时间、也不回答 grounded 闭环是否可用。一次成功只证明"合法提案可以返回"。

## 2 表示改动（已实现，默认关闭）

| 项 | 内容 |
|---|---|
| 模块 | `superagent/menu_view.py` |
| 开关 | `--menu-view full|compact`，默认 `full` |
| 编码 | `menu-tables-v3`：候选表按列共享 + 重复值共享；`decode_menu` 必须还原成原 payload |
| 作用范围 | **只改表示**。候选全集、精确参数、omitted/null 区别、history/checks/memory、输出契约均不变 |
| 验证 | 提案仍按原 `capabilities` 严格校验；B 的排序、eligibility、fit 预算、选型不受影响 |

真实失败 payload 上的离线实测：

| 项 | 值 |
|---|---|
| 原字节 | 21,473 |
| 压缩后 | 13,949 |
| 减少 | **−35.039%** |
| 往返相等 | 是 |
| 候选 / 参数 | 29/29 保留，参数逐项相等 |

这是字节口径，**不是 token 实测**，不能声称省 35% 推理或费用。

## 3 预登记（执行前冻结）

| 项 | 值 |
|---|---|
| 状态来源 | `runs/a151_small_trial_attempt_02/L0/run/events.jsonl` 里那条 `llm_request`（原 payload 21,473 字节，SHA 与本计划一并记录） |
| 模型 | `DeepSeek-V4.1-Flash`（不换模型） |
| 输出契约 | `calibrated-v2` 原样（不换协议、不裁字段） |
| 预算 | **最多 1 次新 Flash、0 次 fit**；不续跑原失败、不重试 |
| 账本 | 唯一真实账本，计入 `a151_menu_probe` scope 与共享请求上限；出现新 unknown/pending 立即停止 |
| 成功判据 | `assess` 返回 `accepted`，且候选属于原 29 个、参数与目录一致 |
| 失败判据 | 传输失败 / 空 content / 非法提案：停止，不换模型、不加 token、不关 thinking 重试 |
| 记录 | 表示变更前后 SHA、编码版本、请求用量、耗时、finish_reason、`data_events`/`other_events` |

## 4 执行顺序

1. `--validate-only`：确认入口、账本预检、payload 编码与还原、预算上限。
2. 一次真实调用（≤1 次请求、0 fit）。
3. 无论成败都落盘：protocol、请求前 payload 的 SHA、编码版本、usage/unknown 状态、completion。
4. 失败即停；**不自动尝试 Qwen**。Qwen 是另一个合规研究臂，需要单独接 preflight、served-model 校验、计价与输出协议，不是换个 `--model` 就能跑。

## 5 不做什么

- 不把一次拆成两次调用（重复输入、额外时延、占满 3 次额度，且第二次仍可能失败）。
- 不把压缩后的表示当作新的证据身份：grounded 的事实仍绑定原目录。
- 不因为一次成功就说闭环或分层有效；端到端验收要另立批次，仍每路径 ≤3 次调用。
- 不为收集失败证据再付费重放原请求。

## 6 交付物

- 代码：`superagent/menu_view.py`、`ab_flash.context_for(menu_view=…)`、`--menu-view`
- 测试：`tests/test_menu_view.py`（11 项：往返相等、候选/参数保留、omitted/null 区分、保留字转义、未知版本拒绝、默认上下文不变、CLI 默认值）
- 运行器：`scripts/run_a151_menu_probe.py`（默认只校验）
- 结果：`research/a151_menu_probe/`
