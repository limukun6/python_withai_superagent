# A151 传输失败归因：不是网络，是推理吃光输出预算

2026-11-24。针对 `runs/a151_small_trial_attempt_02` 里 L0 的 `RemoteDisconnected`。用户用同一端点、同一把 key 原样重放失败 payload，本文把重放结果与仓库已有证据合并，给出结论和可选路径。

原诊断文档（2026-11-18）在外层工作区：`D:\codex\python_withai\docs\campus_api_timeout_diagnosis.md`，不在本仓库内。

## 1 重放事实（用户执行，3 次调用，约 ¥0.05）

| 模型 | 首块 | 总耗时 | 结果 |
|---|---:|---:|---|
| `DeepSeek-V4.1-Flash` | 4.4s | 45.1s | 1024 token 全在 reasoning，`content` 空，`finish=length` |
| `DeepSeek-V4-Pro` | 3.9s | 49.3s | 2048 token 全在 reasoning，`content` 空，`finish=length` |
| `gpt-5.6-luna` | 3.1s | 4.6s | 合法 JSON，`finish=stop` |

payload 约 21,600 字符（~6,750 token），要求返回 10 字段 JSON。

## 2 与既有证据的关系

外层诊断文档 2026-11-18 已经写过两条，本次重放是它们的直接确认：

- **§3.2「推理关不掉」**：`reasoning_effort="none"`、`enable_thinking=false`、`thinking={"type":"disabled"}`、`chat_template_kwargs` 四种写法实测都被服务端忽略，reasoning token 不降。**所以"显式关闭推理"这条路在本端点已经试过，不是新杠杆。**
- **§6 最后一条**："如果服务端在整个 reasoning 阶段不吐任何块，read timeout 仍然会触发——必须在真实负载下验证。" 本次失败正是这个情形：`data_events: 0`。

也就是说，2026-11-18 的流式修复解决了"内容在流、但总量超过 60 秒"的问题（v2 实验 8/8 成功即为此），但**没有解决"推理阶段服务端不吐块"**这一类。本次 L0 就是后者。

## 3 失败时间线的修正

| 时间 | 事件 |
|---|---|
| 06:25:00.43 | `llm_request`，`max_tokens=16384`，`timeout=(10,60)` |
| 06:31:57.27 | `llm_transport_failure`，`data_events: 0` |
| 差值 | **416.8 秒** |

读超时是 60 秒却等了 417 秒，两者只有在"连接上持续有非 `data:` 字节（心跳/注释行）"或"底层 HTTP 库对同一 POST 做了传输级重试"时才同时成立。旧代码把非 `data:` 行直接 `continue` 丢弃，`data_events` 停在 0，因此**无法区分这两种情形**。

本批已改成同时统计 `other_events` / `first_other_seconds` / `last_other_seconds`（`superagent/experiment_llm.py`）。下一次同类失败会给出判别证据：若 `other_events>0` 则是心跳保活；若仍为 0 且耗时接近 60 秒的整数倍，则更像传输级重试。

这件事影响记账：若底层真的重试过同一请求，服务端可能生成了多次，而账本只记 1 条 unresolved（第 13 条），**费用被低估**。

## 4 结论

1. 失败原因在模型与负载，不在网络、也不在超时设置：这两个 DeepSeek 模型在此 payload 的默认推理模式下把输出预算全部用于 reasoning，`content` 始终为空。
2. 这不是"DeepSeek 模型族不可用"。同一模型在 v2 那批较小 payload 上 8/8 成功，A146 的 12 次调用全部返回合法 JSON。**是这个 payload 规模 + 10 字段输出超出了它能在一次请求里收敛的范围。**
3. **不能把 `gpt-5.6-luna` 用作运行时模型**：课程约束语言模型限 Qwen/DeepSeek。它可以作为诊断对照（证明 payload 本身可答），但不能进正式实验，否则结果不可用。

## 5 可选路径（按代价排序）

| 路径 | 代价 | 先验判断 |
|---|---|---|
| A. 把 grounded payload 砍到 4 个输出字段、去掉 `checks`/`memory`/`evidence_projection`，只留 history + 候选 | 0 API；1 次探测调用 | 最可能有效。重放显示 2048 token 仍全在 reasoning，说明要让它"想完"，输入约束必须显著变简单 |
| B. 用允许族内的 Qwen 试同一 payload | 1 次调用 | 课程允许；但 `Qwen3.8-27B` 已知会输出思维过程需要清洗，是否收敛未知 |
| C. 正式路径退回本地规划（F*=L0），grounded/Flash 仅保留为受限研究开关 | 0 | 与现有证据一致：A147 已显示分层动作没有稳定收益，且现在连闭环都跑不通 |
| D. 换 `gpt-5.6-luna` | 0 训练成本 | **不可用于正式结果**，违反课程模型限制 |

## 6 现在的硬阻塞

账本 `runs/a129_B_ledger_return/private/ledger.sqlite` 有 **13 条 unknown**，第 13 条（`153d96d3…`，scope `a151_loop_off`）是本次失败调用，**不在已接受快照内**。`check_snapshot` 要求账本里每条 unknown 都在快照中，否则拒绝继续。

所以在用户明确处理这条之前，A/B/C 里的任何付费探测都会被账本守卫直接拒绝——不会花钱，但也做不了。

另：用户那 3 次重放（约 ¥0.05）**没有经过账本**，账本会低估。建议把这 3 条的模型、输入/输出 token 与时间交回，按 shared 诊断额外登记，而不是留作账外消耗。
