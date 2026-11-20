# A2-01：四个真实任务的冻结历史与 Flash 同状态对照

运行前计划，负责人 A。先 prepare（无 API），保存真实任务、源码、索引、三条历史前缀与上下文哈希；再由 ask 对同一状态逐对调用 raw/checked。账本未核清时只做 prepare，不能新建空账本绕过历史。

任务：released_task1（类别表格）、released_task2（偏好文本对）、Dry Bean（数值多分类）、SMS（单文本 macro-F1）。仅 load_training_task，既有 reserve seed=41101；本批外层 seed=45101；历史搜索 seed=707。外部 validation/test 不参与历史、选择或本批动作比较。

每任务运行本地 interleaved_routes 三次搜索，保存前缀1、2、3。每条历史由同一当前实现生成、无额外确认与refit，共12次拟合。它们是开发历史，非自主 Agent 最终任务成绩。

ask：12个状态×raw/checked=24个请求上限，每状态每条件1次，顺序按 task/state 奇偶轮换；同一状态两组仅 checks/memory 不同。Flash、温度0、stream=True、请求超时沿用60秒读取、无自动重试。调用 allowance=0.30元/任务（计划估算），累计任务及共享账本遵守20元限制。不得把相同 response_id 的复用当独立采样。

错误处理：prepare/ask 都唯一输出目录。每次响应即时落盘；任何传输、usage、提案校验或源码变化立即结束整个批次。未知usage通过持久账本阻断后续。进程中断后不自动重发；接手人先核对 completion、requests/usage与账本，再决定新批次。ask 必须传已核验真实历史的累计账本，不能用 A1 模拟账本。

行为指标：有效提案/计划请求、配置和测量陈述、回应前次失败的事实引用、两组动作一致性。动作改变不代表质量提升。之后另写 evaluate 小批代码，每个不同动作在两份事先冻结的内部评价划分上训练（上限48拟合），保留外层不用于选动作。此批先准备状态与收集决策，不自动扩为性能实验。

命令（主仓库根目录）：

```powershell
& .\.venv\Scripts\python.exe .\runs\integration\scripts\run_a2_state_study.py prepare --output .\runs\a2_01_states
& .\.venv\Scripts\python.exe .\runs\integration\scripts\run_a2_state_study.py ask --prepared .\runs\a2_01_states --output .\runs\a2_01_calls --ledger REAL_HISTORY_LEDGER.sqlite --api-key-file .\api_key.txt
```

prepare 自带历史账本预检报告，但不会读取密钥、访问网络或修改原日志。共享未知费用需要对账，不能当0；data/task 名称仅给实验runner定位文件，不进入规划器用来调用专属解法。历史清单中的11条只是本机已识别缺口，不代表完整项目账本审计已完成。

验收：4任务×3状态齐全，训练/留出无分组重叠，状态历史长度与next_trial一致，primary_metric正确，R/E共享字段逐项相同，无formal test访问。ask只有真实账本preflight通过且与protocol scope一致才发请求；输出明确标识prepared/complete/stopped与实际用量。

预计prepare 1–3分钟（类别表格可能更慢）；ask 24调用约10–20分钟，价格仅按本地历史估算，学校扣款待平台核对。DeepSeek续跑读取 EXPERIMENT_PROGRESS.md，已完成prepare可按指纹复用，不重新训练。
