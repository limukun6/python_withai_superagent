# A151 堵点二：真实闭环的第一次提案失败（给 Codex 复核）

> Codex复核已完成：见[A151_blocker2_codex_resolution_20261124_zh.md](A151_blocker2_codex_resolution_20261124_zh.md)。本机默认HTTP零重试已核实；原16384-token失败的唯一根因仍未证实。无损菜单编码原型减35.50%字节，尚未接正式入口/真实调用。下文保留原提问与归因，不能将原确定性措辞当成复核结论。

2026-11-24。本文把"小试为什么停、停在什么上、现在有哪些证据、下一步有哪几条路"写成一份自足材料，供 Codex 独立判断。文中数字都能在仓库或指定目录复核。

## 0 一句话

小试的第一条路径在**第一次 Flash 提案**处失败：`data_events: 0`，417 秒后连接被服务端关闭，最终没有 usage，账本记为 unknown。

两次短预算重放（1,024 / 2,048）都出现"推理耗尽、`content` 为空、`finish=length`"，说明**在这个 payload 上推理长度超过 2,000 token 仍不出答案**；但原始调用用的是 16,384 预算且没有最终 usage，**它到底是长推理、排队、网关保活/转发还是上游断开，缺服务端 trace 无法唯一定位**。不能据此称该 payload "不可解"，也不能把"输入大 / 字段多 → 推理长"当成已证因果。

## 1 小试怎么停的

| 项 | 值 |
|---|---|
| 计划 | `docs/batches/A151_small_trial_plan_20261124_zh.md`（先落盘、先 validate-only） |
| 入口 | `python -m superagent run … --planner checked-history --scheduler ab-v1 --fit-limit 3` |
| 契约 | L0=`calibrated-v2`+`compact`+分层 off；L1=`grounded-v1`+`by-class` |
| 到达 | 仅 L0；L1 未启动 |
| 第 1 次试验 | anchor `word_C0.05`，1 次 fit，0 调用（不需要模型） |
| 第 2 次决策 | 调用 Flash → `ConnectionError: RemoteDisconnected`，`data_events: 0` |
| 墙钟 | 430.2 秒（其中 416.8 秒在等这次调用） |
| 处置 | 按停止规则中止、不重试、保留 trial_1 检查点 |

## 2 用户重放：8 次调用（关键证据）

#1–#2 用 31,056 字符压测 prompt 做延迟对比；#3–#8 用失败那条 `llm_request` 的原始 messages（21,600 字符）。

| # | 模型 | 输入 | 输出 | 其中 reasoning | 耗时 | 首块 | finish |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | DeepSeek-V4.1-Flash | 6,645 | 26 | 19 | 2.3s | 2.3s | stop |
| 2 | gpt-5.6-luna | 6,421 | 20 | 6 | 1.5s | 1.5s | stop |
| 3 | DeepSeek-V4.1-Flash | 6,821 | 1,024 | **1,024** | 45.1s | 4.4s | **length** |
| 4 | gpt-5.6-luna | 6,294 | 407 | 153 | 4.6s | 3.1s | stop（合法 JSON） |
| 5 | DeepSeek-V4-Pro | 6,874 | 2,048 | **2,048** | 49.3s | 3.9s | **length** |
| 6 | DeepSeek-V4.1-Flash | 6,821 | 1 | — | 1.4s | — | 探测（取 prompt_tokens） |
| 7 | DeepSeek-V4-Pro | 6,874 | 1 | — | 1.3s | — | 探测 |
| 8 | gpt-5.6-luna | 6,294 | 0 | — | 1.3s | — | 探测 |

同一份文本在两个 tokenizer 下的输入差约 500 token（6,821 / 6,874 / 6,294）。

## 3 仓库里已有的相关证据

1. **2026-11-18 诊断文档 §3.2「推理关不掉」**：`reasoning_effort="none"`、`enable_thinking=false`、`thinking={"type":"disabled"}`、`chat_template_kwargs` 四种写法实测被服务端忽略，reasoning token 不降。→ **"显式关推理"不是未试过的新杠杆。**
2. **同一文档 §6 最后一条**：如果服务端在整个 reasoning 阶段不吐任何块，流式下的 read timeout 仍会触发，"必须在真实负载下验证"。本次 `data_events: 0` 正是这个情形。
3. **同族模型在较小 payload 上是好的**：v2 批次 8/8 成功；A146 的 12 次真实调用全部返回合法 JSON。→ 不是"DeepSeek 不可用"，是**这个 payload 在这个运行模式下不可解**。
4. **课程约束**：运行时语言模型限 Qwen/DeepSeek。`gpt-5.6-luna` 只能作诊断对照，**不能进正式结果**。

## 4 失败 payload 的构成（重新量过）

`max_tokens=16384`、`streaming=True`、`thinking=None`、`reasoning_effort=None`、`timeout=(10,60)`、deadline = 起点+600 秒。

用户 payload 21,473 字符，按字节占比：

| 块 | 占比 |
|---|---:|
| `available_candidates`（**29 个候选，含完整参数**） | **60.7%** |
| `candidate_facts` | 13.6% |
| `memory` | 6.2% |
| `instruction` | 5.7% |
| `task_spec` | 5.2% |
| `history`（仅 1 次试验） | 3.3% |
| `checks` | 2.2% |
| `evidence_projection` | 1.3% |
| `budget` | 0.9% |

**这个分布很重要**：先前设想的"去掉 checks/memory/evidence_projection"只能省约 10%，而**候选菜单本身占六成**。输出侧的约束是 calibrated-v2 的 11 个顶层字段；**这次 L0 没有 `grounding` 输出包**（grounding 只在 `grounded-v1`，L1 未启动）。

## 5 尚未区分的两种解释（已按复核意见更正）

读超时是 60 秒，却等了 416.8 秒。原稿把可能原因写成"心跳保活"或"传输级重试（约 7×60 秒）"两条，**其中"传输级重试"这一条已被复核否掉**：

- 本机实际环境 requests 2.34.2 / urllib3 2.7.0，`Session` 默认 `Retry(total=0, read=False)`；对连接、读取、协议错误分别调用实际 retry 策略都立即抛出，不允许重试。
- `ExperimentLLM` 自己新建 `Session`、没有重试循环、禁止 redirect；`campus_direct` 只改进程内域名解析。

因此**不能由 417≈7×60 推断客户端重复调用过**。网关内部是否重试，客户端无法免费查清，需要服务端请求 trace；这也不是把 unknown 结算成零的理由。

关于观测：原稿说"已加 `other_events` 就能区分"，同样不足。`Response.iter_lines` 默认 `chunk_size=512`，要经 `iter_content` 缓冲后才分行——异常前若不足一块，即便 TCP 收到字节，计数仍可能为 0；而新代码里 `if line:` 又排除了空行，只收到空行时仍是 0。所以 **`other_events=0` 不能证明连接没有字节**，`data_events=0` 也不等于 `reasoning_tokens=0`。

要真正观测，需要在 `iter_content` 层记字节数、首末字节时间、终止类型与服务端 request-id，并显式声明零重试策略；观测本身仍不保证打断阻塞读。**不要为了收集失败证据再付费重放原请求。**

## 6 账本处置

| 项 | 值 |
|---|---|
| 账本 | `runs/a129_B_ledger_return/private/ledger.sqlite` |
| 请求数 | 396（388 + 8 条重放导入） |
| known 课程折算 | **¥7.130568**（其中 8 条重放 ¥0.1140185） |
| unknown | **16**（12 历史 + 第 13 条失败调用 + 3 条 gpt） |
| pending | 0 |
| 接受快照 | `research/a130_accepted_history_v3/accepted_history.json`，SHA `7747be18…` |

代价说明：调用者按 0.50/2.00 与 2.25/6.75 估的 ≈¥0.05 偏低一倍；仓库表是 **1.0/4.0** 与 **4.5/13.5**，用这组单价能逐位复现 A146 的 ¥0.162043。3 条 `gpt-5.6-luna` 无课程价，按 unknown 导入、成本保持 null。

## 7 可选路径

| 路径 | 做法 | 代价 | 我的判断 |
|---|---|---|---|
| **A1** | 先把**输出字段**砍到 3–4 个（candidate / params / reason），保留菜单 | 0 API + 1 次探测 | 直击 reasoning 膨胀的原因：输出结构越复杂，它越想 |
| **A2** | 再把菜单压成 `id + 一行参数摘要`（去掉 `candidate_facts` 重复项），可省约 50–60% | 0 API | 不改变动作空间，只改表示 |
| **B** | 允许族内的 Qwen 试同一 payload | 1 次探测 | 合规，但 Qwen 已知会吐思维过程需清洗 |
| **C** | 正式路径退回本地规划（F\*=L0），grounded 只留研究开关 | 0 | 与 A147 一致：分层动作本就没有稳定收益 |
| **D** | 换 `gpt-5.6-luna` | — | **不可用于正式结果**（课程模型限制） |

## 8 已经做过的、不要再做

- 调大 `max_tokens`：只会让它想更久（11-18 已记录）。
- 关 reasoning：四种参数写法已被服务端忽略（11-18 §3.2）。
- 自动重试失败请求：违反"失败即停"，且可能重复计费。
- 用 GPT 出正式结果：违反课程模型限制。

## 9 想请 Codex 判断的四点

1. 你是否同意"根因是输出预算被 reasoning 吃光"？有没有我们尚未排除的第三种解释（例如网关对长输出直接不转发）？
2. 在只允许 Qwen/DeepSeek 的前提下，**A1（先砍输出字段）与 B（换 Qwen）**你更看好哪个？是否有更省的做法，例如把一次 11 字段调用拆成两次 3 字段调用？
3. 417 秒那两种解释，有没有**不花钱**就能判定的办法？
4. 如果最终只能走 C（本地规划为正式路径），对报告里"通用性"的结论影响有多大？是否应该在报告里明确把 grounded/Flash 标为"受限研究开关"？

## 10 复现

```powershell
Set-Location -LiteralPath 'D:\codex\python_withai\runs\integration'
$py = 'D:\codex\python_withai\runs\a79_encoder_assets\venv\Scripts\python.exe'
& $py -m superagent budget-status 'D:\codex\python_withai\runs\a129_B_ledger_return\private\ledger.sqlite'
& $py -m unittest discover -s tests -p "test_reasoning_switch.py" -q
```

| 文件 | 内容 |
|---|---|
| `docs/batches/A151_small_trial_plan_20261124_zh.md` | 预登记计划 |
| `docs/batches/A151_small_trial_results_20261124_zh.md` | 失败记录与账本变化 |
| `docs/batches/A151_transport_diagnosis_20261124_zh.md` | 归因、8 次重放入账、价格校正 |
| `research/a151_small_trial/attempt_02/` | 失败回执、`run_outcome.json`、`llm_usage.jsonl` |
| `runs/a151_small_trial_attempt_02/` | 原始运行目录（本机，未入库） |
