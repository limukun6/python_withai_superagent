# SuperAgent 实验交接说明（可整份发送给 DeepSeek）

你接手的是课程期末项目的实验设计、运行协助与分析工作。请根据真实代码和日志开展实验，给出可追溯的 experimental conclusion，并把结果交回 Codex。不要只写一份概念方案，也不要编造结果。如果你没有本地文件和终端权限，请明确说明，由我在本机执行你给出的完整命令；没有执行证据就标记为“未执行”。

## 1. 项目与任务范围

- 项目：General Algorithm SuperAgent，目标是自动分析分类任务、选择算法配置、执行训练、利用反馈改进并输出结果。
- 小组：2 人；用户报告的显卡为 RTX 5070、8 GB 显存。当前实验主线可用 CPU 运行。
- 本地仓库：`D:\codex\python_withai`。
- GitHub：https://github.com/limukun6/python_withai_superagent.git
- 课程原始资料：`D:\学校\python with ai`。开始前阅读其中实际要求，尤其是评分标准和 experimental conclusion。若拿不到这些资料，明确列为待核实，不要自行补造课程规则。
- 你的范围是实验部分：验证实验流程、必要的实验代码修复、架构比较、消融设计和结果解释。不要擅自重写整个项目。
- 目前框架是“在固定候选集合内选择下一项实验”，尚不是任意生成算法代码、支持所有数据格式的完整通用 Agent。报告必须如实说明这一边界。

## 2. 先读这些文件

相对路径均以本地仓库为根：

1. `README.md`、`docs/agent_experiments.md`。
2. `scripts/run_agent_architecture_experiment.ps1`：运行入口和参数。
3. `superagent/benchmark.py`：策略、划分、训练循环、汇总口径。
4. `superagent/experiment_llm.py`、`superagent/pricing.py`：API、使用量、预算。
5. `superagent/models.py`、`superagent/experiment_worker.py`、`superagent/data.py`：候选空间和数据访问。
6. `tests/test_benchmark.py`。

先记录 `git status --short` 和 `git rev-parse HEAD`，保留已有修改。交接时最新已推送提交为 `aa74003`；开始工作时以实际仓库为准。不要覆盖其他人的工作。

## 3. 已实现的实验与公平性约束

比较以下四种策略：

| 策略 | 行为 | 要检验的问题 |
|---|---|---|
| random | 无放回随机选择配置 | 不使用 LLM 的搜索基线 |
| oneshot | 只看初始数据画像，一次给出全部计划 | 初始规划是否优于随机 |
| feedback | 每轮看历史训练和搜索集指标，再选下一项 | 实验反馈是否改善搜索 |
| reviewed | 同一个 LLM 先提案，再独立调用审查并给最终选择 | 增加审查是否值得额外调用 |

当前 reviewed 是同模型的两个角色调用，不是两个独立模型，也没有第三次修改调用。

固定主协议如下，修改前必须记录原因，所有对照使用相同最终版本：

- 两个开发任务：`data/released_task1.zip`（类别型表格）、`data/released_task2.zip`（文本对）。
- 每个任务 14 个候选配置，具体以 `search_candidates()` 为准。LLM 模型和候选分类器是两个不同概念。
- 每个策略最多尝试 3 个候选；训练失败也占一次尝试。
- 单次训练限时 120 秒，每个策略搜索限时 600 秒。HTTP 调用可能跨过搜索截止时间；最终导出、验证时间也计入总耗时，不能把 600 秒描述为严格总时长上限。
- 在官方 train 内做分组、分层约 80/20 划分；重复特征行或交换顺序的同一文本对保持同组。
- 相同 seed 的策略使用相同内部划分。主实验 seed 固定为 11、22、33。
- 分类器自身随机种子保持 42；这些重复主要衡量数据划分和搜索随机性，不代表独立训练随机性的完整估计。
- Agent 只接收画像与内部 search 指标；选定模型后才评估官方 validation，不做 refit。
- 测试集不能用于选择策略、调参或诊断模型好坏。
- 官方 validation 曾在早期开发中被看过，因此不是全新保留集；两道任务也不能支持“对所有未知任务均有效”的结论。

## 4. 已知故障与证据边界

学校 API：`https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions`，当前默认模型 ID：`DeepSeek-V4-Pro`。密钥仅从本机 `api_key.txt` 读取，不得粘贴进聊天、报告或 Git。

1. 用户曾用最小请求得到 HTTP 200、正文 `ok` 和 usage，说明那次请求的地址、密钥、模型有效。
2. 环境曾残留 `127.0.0.1:7897` 的 VPN 代理。实验客户端已设置 `requests.Session().trust_env=False`。连接状态变化后仍需实际检查；不要把所有网络错误都归为同一原因。
3. 旧 Pilot 多次返回 `content=null`，随后策略触发随机恢复。旧日志未记录 finish_reason，故“2048 token 被推理耗尽”是当前合理怀疑，尚未由 finish_reason 证实。不要把它写成已证实根因。
4. `aa74003` 已将输出上限从 2048 调到 4096，并记录 finish_reason 和 reasoning_content 是否存在。8 项本地测试通过，但尚未有修复后完整成功 Pilot 的证据。读取新日志后验证修复效果。
5. 不要用 reasoning_content 冒充最终 JSON。若 finish_reason=length，检查截断；若 timeout 或 DNS/TLS 失败，分别调查。若需修改输出上限或请求超时，在 Pilot 阶段统一修改并记录，不要只给某个策略特殊待遇。
6. 一旦有 fallback，当前运行的 `valid_policy_run=false`。随机恢复产生的分数不能充当原 LLM 策略的有效结果。

旧 Pilot：`runs/agent_architecture/20261116T124824740402Z`。表格任务 random 为 0.7330，feedback 为 0.7389 且无 fallback，但 feedback 有一次训练失败；其余 LLM 组合均有 fallback。这只是故障诊断材料，不能据此宣称反馈式胜出。

修复后尝试目录 `20261116T131814568805Z` 曾出现 DNS 解析失败且未完成有效比较。开始时检查最新目录与正在运行的进程，避免重复启动实验。

## 5. 分阶段执行

### A. 预检查与 Pilot

以下每行是一条完整 PowerShell 命令，路径参数必须与 `-File` 写在同一行：

```powershell
cd D:\codex\python_withai
git status --short
git rev-parse HEAD
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_agent_architecture_experiment.ps1" -ApiKeyFile ".\api_key.txt" -ValidateOnly
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_agent_architecture_experiment.ps1" -ApiKeyFile ".\api_key.txt" -Mode Pilot
```

ValidateOnly 只检查文件是否存在，不验证 API 连通性。Pilot 是 4 策略 × 2 任务 × 1 seed，共 8 个运行；批次估算额度上限 ¥2。

逐一检查 summary、events、usage：是否完成、是否 fallback、是否记全账、是否达到计划尝试数、是否存在训练超时，以及最终 JSON 是否可用。训练超时本身可以是合法实验结果，不要为了让分数好看删掉。网络、协议与记账问题应在正式实验前解决。

若需要修复，保留旧失败目录，记录修改，跑相关本地测试，再创建新 Pilot 批次。禁止自动无限重试。当前 HTTP 超时为 60 秒，增加输出长度后能否在此时间内完成仍需检查。

### B. 正式架构比较

Pilot 流程验证后，冻结代码和参数，登记 commit、source hash、种子和评价指标，再执行：

```powershell
cd D:\codex\python_withai
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_agent_architecture_experiment.ps1" -ApiKeyFile ".\api_key.txt" -Mode Main -BatchBudgetRmb 10
```

预期 4 × 2 × 3 = 24 个运行。¥10 是整个批次的估算上限，不是额外充值指令，也不是严格 token 硬限额。Pilot 与 Main 分开分析，不能把 seed=11 的 Pilot 当作额外独立重复。不要看到结果后增加种子直到偏好的架构胜出。

正式阶段如需改代码，将原批次标记为中断/旧版本，保留证据，按新冻结版本重建可比批次。网络或用量不明时暂停付费调用，先解释原因。

### C. 消融与资源公平性

主实验本身包含“有无反馈”和“有无审查”的初步比较。但 reviewed 调用更多，不能把收益全部归因于审查结构。

根据主实验诊断，给出一个小规模、预先登记的后续方案：在同任务、同种子、同训练尝试数下，为 feedback 与 reviewed 设相同的每次运行累计 token 上限，报告预算耗尽行为、实际消耗、准确率和成功率。当前脚本只有共享人民币估算上限，没有实现每运行 token 配额；先明确这一实现缺口，不要仅把 BatchBudgetRmb 调成一样就声称 token 已匹配。

在运行新增实验前说明所需代码、预算和预计调用数；本次直接执行范围是上述 Pilot 与 Main，新增付费批次先交我确认。若无法实现资源对照，写成研究局限，不能伪造消融结果。

## 6. 必须交付的分析

逐任务比较，不要只报混在一起的总平均值：

- 验证准确率的每 seed 原值、均值、样本标准差；同 seed 的配对差值。
- 有效运行数/计划运行数、fallback 数、训练失败数、未完成运行及原因。缺失的运行不能从分母里消失。
- 实际耗时、LLM 调用次数、输入/输出 token、估算费用、usage 是否完整。用量未知不是费用为零。
- 每轮最佳内部 search accuracy 曲线；这是内部搜索表现，不能冒充逐轮官方验证结果。
- reviewed 相对 feedback 的增益与额外 token/时间；胜出不明显时允许结论是“不值得增加复杂度”。
- 每个结论指向具体原始记录。3 个 seed 的证据有限，不得轻率声称统计显著或广泛泛化。

现有 comparison.md 只对有效策略运行汇总准确率，可能有幸存者偏差；展示它时必须同时展示成功率、失败原因和所有运行的资源开销。不能单凭有效子集均值决定最终方案。

按截图计价仅为暂估，不能冒称已核验课程要求的官方历史价格。保留原始 usage 供后续重新计价。已有课程约束记录包括允许模型范围/发布时间以及每任务 ¥20，需以原始要求复核模型资格，不能用平台存在该 ID 作为资格证明。

最终报告需要回答：哪种方案在什么任务和预算下表现更好？收益来自何种观察到的行为？代价是什么？失败在哪里？哪些结论证据不足？下一步最值得做的一项实验是什么？

## 7. 交回 Codex 的材料与格式

请给我一份可直接转发的中文结果说明，使用下面的格式，并附文件；不要只发“跑完了”。

```text
【代码版本】commit / source_sha256；是否有未提交修改
【执行环境】Python、主要库版本、CPU/GPU使用方式
【批次】Pilot/Main目录；开始/完成时间；实际执行的完整命令
【代码改动】文件、原因、测试；无改动则写无
【协议偏离】偏离项及影响；无则写无
【完成情况】计划24个，完成X个，有效Y个；未完成原因
【结果表】任务 / 策略 / seed / validation_accuracy / valid_policy_run /
           fallback_count / failed_trials / elapsed_seconds / llm_calls /
           tokens / estimated_cost_rmb / usage_complete
【配对比较】逐任务、同seed差值，均值及样本标准差
【诊断】null content的新证据、finish_reason、网络错误与训练超时
【实验结论】证据支持的发现、推荐方案、限制和未证实假说
【费用】已知估算总额、未知用量、定价状态
【待办】尚未完成的消融、资源对照和泛化验证
```

至少附上完整批次的 `protocol.json`、`summaries.json`、`comparison.md`，以及各子运行的 `summary.json`、`trials.json`、`selection.json`、`events.jsonl` 和实际命名的 usage 日志。出错运行也要保留；无需发送大型模型文件和原始数据。分享前检查敏感信息，绝不附 API key。

如仅能发一段文字，优先发版本、完整结果表、失败原因及结论；后续 Codex 将基于这些真实材料决定下一步开发与最终报告。
