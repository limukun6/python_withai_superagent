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

## 6 账外调用已登记，账本闸门已重开

用户实际记录的是 **8 次**调用（不是 3 次）：#1–#2 用 31,056 字符压测 prompt 做延迟对比，#3–#5 重放 A151 原始 messages，#6–#8 为取精确 `prompt_tokens` 补发的 `max_tokens=1` 探测。

### 按仓库价格表重算（不是按调用者的估算）

| # | 模型 | 输入 | 输出 | 仓库单价（¥/M） | 课程折算 |
|---:|---|---:|---:|---|---:|
| 1 | DeepSeek-V4.1-Flash | 6,645 | 26 | 1.0 / 4.0 | ¥0.006749 |
| 3 | DeepSeek-V4.1-Flash | 6,821 | 1,024 | 1.0 / 4.0 | ¥0.010917 |
| 5 | DeepSeek-V4-Pro | 6,874 | 2,048 | 4.5 / 13.5 | ¥0.058581 |
| 6 | DeepSeek-V4.1-Flash | 6,821 | 1 | 1.0 / 4.0 | ¥0.006825 |
| 7 | DeepSeek-V4-Pro | 6,874 | 1 | 4.5 / 13.5 | ¥0.030947 |
| 2 / 4 / 8 | gpt-5.6-luna | 6,421 / 6,294 / 6,294 | 20 / 407 / 0 | **无课程价** | **null（保持 unknown）** |

**合计 ¥0.1140185。** 调用者按 0.50/2.00 与 2.25/6.75 估的 ≈¥0.05 偏低一倍：仓库 `pricing.py` 的现价是 **1.0/4.0** 与 **4.5/13.5**，且用这组单价能把 A146 的 ¥0.162043（74,319 输入 / 21,931 输出）逐位复现，说明表本身是自洽的。

### 导入结果

用仓库自带的 `TaskBudgetLedger.import_history` 走历史导入通道，scope 统一 `shared_diagnostics`：

| 项 | 之前 | 之后 |
|---|---|---|
| 请求数 | 388 | **396** |
| known 课程折算 | ¥7.0165495 | **¥7.130568** |
| unknown | 13 | **16**（新增 3 条 gpt，成本 null） |
| pending | 0 | 0 |
| 账本 SHA-256 | `d671c424…` | **`6f1867ba…`** |

`accepted_history_v3` 列出全部 **16** 条 unknown（12 条历史 + 第 13 条 + 3 条 gpt），SHA `7747be18…`。校验结果：

- 未打补丁的 `research_preflight` **仍然拒绝**（第 13 条不是 `historical:` 前缀）——这是对的，接受快照是唯一放行通道。
- 套上 `accepted_history_v3` 后 `research_preflight('a151_probe', .15)` **ALLOWED**，`accepted_unknown_request_count=16`。

因此 A/B/C 任一路径现在都**可以**再次发起真实调用；仍然保留"出现新的 unknown/pending 立即停止"。

### 需要用户确认的一点

3 条 `gpt-5.6-luna` 因为不在课程价格表里，我按 unknown 导入、成本保持 null，并**一并放进了 v3 的接受清单**。如果你只想接受第 13 条而不接受这 3 条，告诉我，我重新生成 v3 把它们排除——排除后它们仍会阻断继续。

