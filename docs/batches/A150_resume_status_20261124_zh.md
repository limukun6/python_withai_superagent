# A150 幂等安全续跑：只读判定已落地，续跑执行未接入

2026-11-24。交接把 A150 定义为"幂等安全续跑（继承预算/账本、避免重复 fit/API）"。本轮实现**续跑前的只读判定与护栏**，不改执行循环。

## 1 为什么先做判定

续跑的风险不在"能不能跑"，而在"会不会重复付钱"。一次续跑如果要重做一次已结算的 fit 或请求，账本就错了，而且事后无法分辨。所以真正的续跑必须由一组硬条件把关，这组条件本身可以离线、零成本地实现并测透。

## 2 已实现

模块 `superagent/resume.py`，唯一入口 `inspect(run_dir, ledger_path=None, sources_now=None, task_archive_sha256=None)`，返回三种状态之一：

| 状态 | 含义 |
|---|---|
| `complete` | 运行已结束，续跑只会重复劳动 |
| `resumable` | 记录与账本一致，且还有预算 |
| `blocked` | 有任一条护栏不满足，附 `blockers` 列表 |

### 护栏（全部来自已记录证据）

| 检查 | 条件 |
|---|---|
| 账本未结清 | 该 scope 存在 `unknown` 或 `pending` 行 |
| 账本落后 | 账本已结算请求数 < 运行日志里的 `llm_call` 数 |
| 源码漂移 | 当前 `sources()` 与 `run_manifest.source_files` 不一致 |
| 任务包变化 | `task_archive_sha256` 与清单不符 |
| 试验编号不连续 | 已完成试验不是从 1 连续 |
| fit 未完成 | `attempted != completed`（费用已记，结果没有） |
| 预算不足 | 剩余 fit ≤ 1（含预留 refit）或剩余请求 ≤ 0 |
| 清单缺失 | 没有 `run_manifest.json`，无法判断 |

同时给出 `next_trial`、`remaining_fits`、`remaining_requests` 与该 scope 的账本视图。

**`inspect` 只读**：不写文件、不发请求、不动账本，由测试断言账本文件哈希不变、运行目录不新增文件。

## 3 命令行

```powershell
python -m superagent resume-status <run_dir> --ledger <ledger.sqlite>
```

退出码：`complete`=0、`resumable`=0、`blocked`=2。

## 4 在真实中断件上的验证（0 花费）

| 对象 | 结果 |
|---|---|
| `research/a151_breadth/attempt_02/pair_s707_b2/run`（已完成） | `complete`，3 fit / 1 请求，退出码 0 |
| `research/a151_menuview/attempt_01/text_s911_full/run`（被墙钟停止） | **`resumable`**：0 fit / 0 请求、剩余 3 fit / 3 请求、scope `a151_mv_text_full` 账本 3 条已结算、无 blocker |
| 不存在的路径 | `blocked`：`run_manifest.json is missing`，退出码 2 |

账本 SHA 在三次调用前后完全一致。

## 5 续跑锚点：状态文件，不是日志

先试过"从日志重建状态"，**结论是做不到**：

1. `llm_usage.jsonl` 的每条 `llm_call` 只有 `role`（如 `ab_checked_history`），**没有试验编号**，因此无法判断某个请求属于第几次决策；而语义修复会让一次决策产生两个请求，顺序也无法唯一还原。
2. 用最终文件重建的"状态"与中间态必然不等（中间态是逐决策快照），这个对比本身无意义——那份错误的对比产物已删除，不留误导。

改测真正要紧的性质：**每个 `state_k.json` 是不是它那次运行的忠实前缀快照**。对 139 个状态文件逐项检查：

| 检查 | 结果 |
|---|---|
| `history` 是最终 `history.json` 的前缀 | 139/139 |
| `prediction_evidence` 是对应前缀 | 139/139 |
| `attempted_classifier_fits` == 已记录试验数 | 139/139 |
| `remaining_fit_allowance` == `fit_limit − 试验数` | 139/139 |
| 已用请求 ≤ 请求上限 | 139/139 |
| **不一致的状态文件** | **0 / 139** |

所以续跑的正确锚点是**最后一个状态文件**：它已经完整描述"停在哪、花了多少、下一步是第几次"。日志只用于对账，不用于重建。

证据：`research/a150_resume/state_prefix_check.json`。

## 6 尚未做（明确）

1. **续跑执行未实现**：`inspect` 只回答"能不能继续"，没有"继续"这个动作。
2. **预算继承未接线**：`remaining_fits` / `remaining_requests` 只是算出来给人看，执行器还没读它；从状态文件加载并继续的代码不存在。
3. **账本幂等键未实现**：真正的幂等还需要"同一请求只结算一次"的键，目前依赖 `ExperimentLLM` 的 `max_requests` 与账本预留。
4. **没有测过"续跑后总账等于一次跑完"**，因为续跑本身不存在。

## 7 接续时的验收口径（预登记）

1. 续跑必须在**新目录**写结果，并记录来源运行目录与 SHA，不覆盖原件。
2. 续跑不得重做任何已记录完成的 fit：以状态文件的试验编号为准。
3. 续跑不得重发已结算的请求：以账本 `known` 行数 ≥ 本地 `llm_call` 数为前提。
4. 续跑后总 fit/请求必须等于"一次跑完"的数目，逐个试验对账。
5. 任一护栏不满足即拒绝续跑，不"尽力而为"。

## 8 资源

0 fit / 0 API。账本仍 453 请求、known ¥7.882065、unknown 16、pending 0。

## 9 测试

`tests/test_resume.py`，11 项：完成态、可续跑态、只读性、unknown 阻断、源码漂移阻断、任务包不符阻断、fit 未完成阻断、请求耗尽阻断、仅剩预留 refit 阻断、清单缺失阻断、命令行接线。
