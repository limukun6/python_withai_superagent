# SuperAgent 架构比较实验：执行结果与结论（交回 Codex）

> 执行者：本机 DeepSeek agent（有本地文件与终端权限）。所有数字来自磁盘上的原始记录，未做任何手填或估算替代。

## 【代码版本】

- `git rev-parse HEAD`：`dc0a822c2ab245165b37e1b574a97b10e9c397ab`
- 工作区**有未提交修改**（详见【代码改动】），`git status --short`：
  - `M scripts/run_agent_architecture_experiment.ps1`
  - `M superagent/experiment_llm.py`
  - `M tests/test_benchmark.py`
  - `?? scripts/analyze_agent_architecture.py`、`?? scripts/summarise_batch.py`、`?? scripts/verify_batch_artifacts.py`
- Main 批次的 `source_sha256`（`superagent/*.py` 汇总，记录在 `protocol.json`）：
  `b0b22945ceee89ebef6567408d9c572328877167c372dcc2aef97365e08229d4`
- 各 Pilot 的 `source_sha256` 不同，说明它们对应不同代码版本（这正是它们不能合并统计的原因）。

## 【执行环境】

- Python 3.14.4；numpy 2.5.4、scipy 1.18.1、scikit-learn 1.9.1、joblib 1.6.0、catboost 1.2.10
- Windows 11（10.0.26200）；31.5 GB RAM；NVIDIA RTX 5070 Laptop 8 GB
- **GPU 未使用**：sklearn 与 catboost 均以默认 `task_type=CPU` 运行，`thread_count=-1` 用满 CPU 核；训练在独立子进程中执行（`superagent.experiment_worker`），每 trial 120 s 上限
- 模型：`DeepSeek-V4-Pro`（平台默认）；`requests.Session().trust_env=False`（绕过残留的 `127.0.0.1:7897` 代理）

## 【批次】

| 批次 | 目录 | 运行数 | 有效 | 费用 | 说明 |
|---|---|---:|---:|---:|---|
| Pilot 1 | `runs/agent_architecture/20261117T003810309072Z` | 8 | 3 | ¥0.0268 | 60 s 超时 → 首次传输失败即熔断 |
| Pilot 2 | `20261117T004851244308Z` | 8 | 6 | ¥0.3207 | 超时改 180 s；仍有 2 次 `finish_reason=length` |
| Pilot 3 | `20261117T011641872660Z` | 8 | 7 | ¥0.3213 | 输出上限 8192；180 s 仍超时一次 |
| Pilot 4（中止） | `20261117T014834930453Z` | 3 | 2 | ¥0.0978 | 8192 被 one-shot 计划吃穿，配置已知无效，主动中止 |
| Pilot 5（最终配置验证） | `20261117T020000637028Z` | 8 | 6 | ¥0.2269 | 16384 / 720 s / 1200 s；两个失败仍是连接重置 |
| **Main** | **`20261117T023216973488Z`** | **24** | **24** | **¥1.3045** | 最终冻结配置，0 fallback |

- Main 起止：本地 2026-11-17 10:32:16 → 12:40:14（约 2 h 08 min）；各运行累计耗时 7742 s
- Main 实际执行的完整命令：

```powershell
cd D:\codex\python_withai
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_agent_architecture_experiment.ps1" -ApiKeyFile ".\api_key.txt" -Mode Main -BatchBudgetRmb 10
```

- 该命令等价于：`python -m superagent.benchmark data/released_task1.zip data/released_task2.zip --strategies random oneshot feedback reviewed --seeds 11 22 33 --max-trials 3 --time-limit 1200 --trial-timeout 120 --output-dir runs/agent_architecture --model DeepSeek-V4-Pro --batch-budget-rmb 10`

### 本机既有的更早批次（非本次执行，未合并统计）

`runs/agent_architecture/` 下还有 5 个本次工作之前就存在的目录，它们对应更旧的代码版本（`source_sha256` 与 Main 不同），因此**不参与本次任何结论**，仅登记如下（usage 不完整数 / 无效运行数 / 总数）：

| 目录 | usage 不完整 | 无效运行 | 总数 |
|---|---:|---:|---:|
| `20261115T071447271550Z` | 1 | 6 | 8 |
| `20261116T120142776558Z` | 1 | 6 | 8 |
| `20261116T123529653145Z` | 1 | 3 | 5 |
| `20261116T124824740402Z`（即交接文档中的"旧 Pilot"） | 0 | 5 | 8 |
| `20261116T131814568805Z` | 1 | 6 | 8 |

这些目录保留未删，可作为"修复前故障模式"的对照证据。本次的 5 个 Pilot 与 Main 全部在 `2026-11-17`，且每个批次的 `protocol.json` 都记录了各自的 `source_sha256` 与参数，可直接区分版本。

## 【代码改动】

| 文件 | 改动 | 证据/原因 | 测试 |
|---|---|---|---|
| `superagent/experiment_llm.py` | `max_output_tokens` 4096 → 8192 → **16384** | Pilot 1 用 2048 被截断；Pilot 2/3 在 4096 出现 `finish_reason=length` + `content=null`（历史最长的第 3/6 次调用）；Pilot 4 用 8192 仍被 one-shot 计划吃穿（`output_tokens=8192`）。实测 61 次调用：输出中位数 1610、均值 1983、最大撞顶 8192，one-shot 角色最重（中位数 2614）。16384 已用探针确认端点接受（HTTP 200） | 既有断言改为引用类常量，并断言 `> 4096` |
| 同上 | `request_timeout_seconds` 60 → 180 → 240 → **720** | Pilot 1 在 60 s 读到超时（前一次调用耗时 26.6 s、936 推理 token）；Pilot 3 在 180 s 仍超时。重调用生成速率约 26 token/s，16384 token 回答可需约 10 分钟 | 同上 |
| 同上 | 新增 `max_transport_retries = 2`（有界重试） | 5 个 Pilot 共观测 5 次传输失败（`ConnectionResetError 10054`、`Max retries exceeded`、`ReadTimeout`），约占调用 7%。旧策略"一次失败即整批熔断"使 24 运行的 Main 几乎必然中断 | 新增 3 个测试：瞬时失败重试后成功；重试耗尽后熔断；HTTP 4xx 不重试 |
| `scripts/run_agent_architecture_experiment.ps1` | `--time-limit` 600 → 900 → **1200** | Pilot 3 的 reviewed 用 6 次调用吃掉 557 s，第 3 次训练被截在 42.9 s（正常需约 57 s），破坏"3 trials 匹配"的预声明前提 | 无单测（脚本参数） |
| `tests/test_benchmark.py` | 断言更新 + 3 个重试测试 | — | `python -m unittest discover -s tests` → **11 passed** |
| 新增 `scripts/analyze_agent_architecture.py` | 批次分析（结果表 / 汇总 / 配对 / 诊断 / 曲线） | 交付需要 | 在 5 个 Pilot + Main 上跑通 |
| 新增 `scripts/summarise_batch.py`、`scripts/verify_batch_artifacts.py` | 总账汇总；交付物完整性校验 | 交付需要 | 在 Main 上校验通过（24/24 运行文件齐备，`test_accessed` 全为 false） |

**未改动**：候选空间（14 个）、`max_trials=3`、`trial_timeout=120 s`、内部分层分组 80/20、seed 11/22/33、分类器 `random_state=42`、`test_accessed` 逻辑、策略实现。

## 【协议偏离】

| 偏离项 | 原协议 | 实际 | 影响 |
|---|---|---|---|
| 输出上限 | 2048（`aa74003` 后 4096） | 16384 | 只放宽"模型能写多长"，不放宽其决策自由度；所有策略统一 |
| 请求超时 | 60 s | 720 s | 只影响能否等到回答，不改变回答内容；所有策略统一 |
| 策略时限 | 600 s | 1200 s | 所有策略同一上限；使 reviewed 能跑满 3 个 trial |
| 传输重试 | 单次失败即熔断整批 | 每次调用最多重试 2 次（有界） | 失败的尝试仍记为 `unresolved_call`（`usage_known=false`），会计不隐藏缺口 |

未偏离：候选集、trial 数、划分方式、种子、分类器种子、官方 validation 只在选择后评估、测试集从不读取。

## 【完成情况】

- 计划 **24**，完成 **24**，有效 **24**（`valid_policy_run=true` 全为真），fallback **0**，`usage_complete=false` 的运行 **0** 个。
- 训练超时 **6** 次（占 72 个 trial 的 8.3%），训练失败 0 次；全部集中在 `categorical_tabular` 的 feedback/reviewed（catboost depth6/8 × 600 iterations 单次训练本身可超 120 s，或被 LLM 延迟挤占后剩余时间不足）。
- 未完成运行：无。

## 【结果表】

任务 `categorical_tabular`（官方 validation accuracy）：

| 策略 | seed | 选中候选 | search_acc | **val_acc** | valid | fallback | trials | failed | 秒 | 调用 | tokens | RMB | usage |
|---|---:|---|---:|---:|:--:|---:|---:|---:|---:|---:|---:|---:|:--:|
| random | 11 | `catboost_depth4_reg10.0` | 0.7344 | **0.7330** | ✅ | 0 | 3/3 | 0 | 155.1 | 0 | 0 | 0 | 是 |
| random | 22 | `extra_trees_leaf1` | 0.7104 | **0.7246** | ✅ | 0 | 3/3 | 0 | 21.2 | 0 | 0 | 0 | 是 |
| random | 33 | `catboost_depth4_reg10.0` | 0.7482 | **0.7350** | ✅ | 0 | 3/3 | 1 | 194.5 | 0 | 0 | 0 | 是 |
| oneshot | 11 | `catboost_depth6_reg3.0` | 0.7381 | **0.7443** | ✅ | 0 | 3/3 | 0 | 390.3 | 1 | 8064 | 0.0504 | 是 |
| oneshot | 22 | `catboost_depth6_reg10.0` | 0.7301 | **0.7424** | ✅ | 0 | 3/3 | 0 | 146.2 | 1 | 2089 | 0.0101 | 是 |
| oneshot | 33 | `catboost_depth6_reg10.0` | 0.7512 | **0.7350** | ✅ | 0 | 3/3 | 0 | 157.9 | 1 | 2364 | 0.0119 | 是 |
| feedback | 11 | `catboost_depth6_reg10.0` | 0.7394 | **0.7389** | ✅ | 0 | 3/3 | 1 | 563.8 | 3 | 11178 | 0.0616 | 是 |
| feedback | 22 | `catboost_depth6_reg3.0` | 0.7363 | **0.7355** | ✅ | 0 | 3/3 | 1 | 525.6 | 3 | 10648 | 0.0580 | 是 |
| feedback | 33 | `catboost_depth6_reg10.0` | 0.7512 | **0.7350** | ✅ | 0 | 3/3 | 1 | 544.2 | 3 | 10554 | 0.0576 | 是 |
| reviewed | 11 | `catboost_depth6_reg10.0` | 0.7394 | **0.7389** | ✅ | 0 | 3/3 | 0 | 779.0 | 6 | 24336 | 0.1352 | 是 |
| reviewed | 22 | `catboost_depth6_reg3.0` | 0.7363 | **0.7355** | ✅ | 0 | 3/3 | 1 | 685.4 | 6 | 19769 | 0.1043 | 是 |
| reviewed | 33 | `catboost_depth6_reg10.0` | 0.7512 | **0.7350** | ✅ | 0 | 3/3 | 1 | 605.5 | 6 | 17076 | 0.0861 | 是 |

任务 `pair_text`（官方 validation accuracy）：

| 策略 | seed | 选中候选 | search_acc | **val_acc** | valid | fallback | trials | failed | 秒 | 调用 | tokens | RMB | usage |
|---|---:|---|---:|---:|:--:|---:|---:|---:|---:|---:|---:|---:|:--:|
| random | 11 | `word_char_C0.3` | 0.6916 | **0.7053** | ✅ | 0 | 3/3 | 0 | 28.0 | 0 | 0 | 0 | 是 |
| random | 22 | `word_C1.5` | 0.6783 | **0.6745** | ✅ | 0 | 3/3 | 0 | 11.7 | 0 | 0 | 0 | 是 |
| random | 33 | `word_char_C0.3` | 0.6843 | **0.6894** | ✅ | 0 | 3/3 | 0 | 25.3 | 0 | 0 | 0 | 是 |
| oneshot | 11 | `word_char_C1.5` | 0.6964 | **0.7053** | ✅ | 0 | 3/3 | 0 | 80.0 | 1 | 2693 | 0.0147 | 是 |
| oneshot | 22 | `word_char_C0.7` | 0.6970 | **0.6904** | ✅ | 0 | 3/3 | 0 | 50.1 | 1 | 1697 | 0.0080 | 是 |
| oneshot | 33 | `word_char_C0.3` | 0.6843 | **0.6894** | ✅ | 0 | 3/3 | 0 | 66.7 | 1 | 2192 | 0.0114 | 是 |
| feedback | 11 | `word_char_C1.5` | 0.6964 | **0.7053** | ✅ | 0 | 3/3 | 0 | 193.5 | 3 | 8136 | 0.0428 | 是 |
| feedback | 22 | `word_char_C0.7` | 0.6970 | **0.6904** | ✅ | 0 | 3/3 | 0 | 326.6 | 3 | 14403 | 0.0851 | 是 |
| feedback | 33 | `word_char_C0.7` | 0.6825 | **0.6862** | ✅ | 0 | 3/3 | 0 | 205.6 | 3 | 9236 | 0.0502 | 是 |
| reviewed | 11 | `word_char_C1.5` | 0.6964 | **0.7053** | ✅ | 0 | 3/3 | 0 | 698.4 | 6 | 30873 | 0.1829 | 是 |
| reviewed | 22 | `word_C1.5` | 0.6783 | **0.6745** | ✅ | 0 | 3/3 | 0 | 715.2 | 6 | 31789 | 0.1891 | 是 |
| reviewed | 33 | `word_char_C0.3` | 0.6843 | **0.6894** | ✅ | 0 | 3/3 | 0 | 572.5 | 6 | 25276 | 0.1451 | 是 |

按策略汇总（n=3）：

| 任务 | 策略 | val_acc 均值 | 样本 SD | tokens（3 run 合计） | RMB | 秒 |
|---|---|---:|---:|---:|---:|---:|
| categorical_tabular | random | 0.7309 | 0.0055 | 0 | 0.0000 | 371 |
| categorical_tabular | oneshot | **0.7406** | 0.0049 | 12 517 | 0.0724 | 694 |
| categorical_tabular | feedback | 0.7365 | 0.0021 | 32 380 | 0.1772 | 1633 |
| categorical_tabular | reviewed | 0.7365 | 0.0021 | 61 181 | 0.3256 | 2070 |
| pair_text | random | 0.6897 | 0.0154 | 0 | 0.0000 | 65 |
| pair_text | oneshot | **0.6950** | 0.0089 | 6 582 | 0.0341 | 197 |
| pair_text | feedback | 0.6940 | 0.0101 | 31 775 | 0.1781 | 726 |
| pair_text | reviewed | 0.6897 | 0.0154 | 87 938 | 0.5172 | 1986 |

## 【配对比较】

同任务、同 seed 的差值（仅用两侧都有效的配对，本批 3/3 全部可用）：

| 任务 | 对比 | 每 seed 差值 | 均值 | 样本 SD |
|---|---|---|---:|---:|
| categorical_tabular | oneshot − random | +0.0113 / +0.0177 / +0.0000 | **+0.0097** | 0.0090 |
| categorical_tabular | feedback − random | +0.0059 / +0.0108 / +0.0000 | **+0.0056** | 0.0054 |
| categorical_tabular | reviewed − random | +0.0059 / +0.0108 / +0.0000 | +0.0056 | 0.0054 |
| categorical_tabular | reviewed − feedback | 0.0000 / 0.0000 / 0.0000 | **0.0000** | 0.0000 |
| pair_text | oneshot − random | +0.0000 / +0.0160 / +0.0000 | **+0.0053** | 0.0092 |
| pair_text | feedback − random | +0.0000 / +0.0160 / −0.0032 | +0.0043 | 0.0103 |
| pair_text | reviewed − random | 0.0000 / 0.0000 / 0.0000 | **0.0000** | 0.0000 |
| pair_text | reviewed − feedback | 0.0000 / −0.0160 / +0.0032 | **−0.0043** | 0.0103 |

## 【诊断】

- **Main 批次**：LLM 调用 60 次，**传输失败 0、HTTP 非 2xx 0、`content` 非字符串 0、`finish_reason` 全为 `stop`（60/60）**；token 232 373，其中推理 token 167 414（**72.0%**）；`usage_complete` 全为真。
- **`content=null` 的新证据（来自 Pilot，已由 `finish_reason` 证实）**：5 个 Pilot 共 3 次 `finish_reason=length` + `content=null`，全部发生在"历史最长的那次调用"（feedback 的 trial3、reviewed 的 trial3 审查调用、one-shot 的初次计划）。**因此 `aa74003` 的 4096 不足；8192 也不足；16384 下 60 次调用零截断。**
- **传输失败全部是服务端/网络侧**：`ConnectionResetError 10054`（远程主机强迫关闭）、`Max retries exceeded`（连接层）、`ReadTimeout`。加入有界重试后，Main 批次 0 次失败。
- **训练超时 6 次**，全部在 `categorical_tabular` 的 feedback/reviewed：catboost depth6/8 × 600 iterations 单次训练接近或超过 120 s；同时 LLM 延迟会缩短第 3 个 trial 的剩余时间（Pilot 3 实测被截在 42.9 s）。
- **审查者一致率（新诊断）**：reviewed 共 18 次审查调用，**只改了 2 次提案（11%）**，且这两次都没有改变最终选中方案 → reviewed 的验证准确率与 feedback 完全一致。
- **搜索指标与官方验证存在不一致个例**：`pair_text` seed 22，oneshot 的内部 search_acc 最高（0.6970），但官方验证 0.6904 低于 seed 11 的 0.7053。内部搜索指标不能当作官方成绩。

## 【实验结论】

1. **有 LLM 搜索优于随机，但幅度很小。** 同 seed 配对：oneshot 比 random 高 **+0.97pp**（表格）/ **+0.53pp**（文本）；feedback 高 +0.56pp / +0.43pp。而 3 个 seed 的样本标准差是 0.005–0.015，**均值差在 1 个 SD 以内，不能声称统计显著**。
2. **one-shot 已经拿到这个搜索空间里几乎全部可得收益。** feedback 在两个任务上都略低于 oneshot（−0.41pp / −0.10pp，均在噪声内），却花 2.5–5 倍 token。
3. **审查（reviewed）不值得增加复杂度。** 相对 feedback：表格任务配对差 **0.0000**、文本任务 **−0.43pp**；代价是 **token 1.9–2.8 倍、耗时 1.3–2.7 倍、费用 1.8–2.9 倍**。原因已被诊断证实：审查者 11% 的改动率且未影响最终选择，即"审查没有提供新信息"。
4. **收益来自"一开始选对族"，而不是"更会搜索"。** oneshot 在表格任务 3/3 都落在 catboost depth6 家族、文本任务 3/3 都落在 word_char 家族；random 偶尔抽到 `extra_trees_leaf1` / `word_C1.5`（弱族）就掉 1–2pp。这符合"表示/族选择比逐轮微调更值钱"的预期。
5. **代价结构清晰**：random ≈ ¥0、oneshot ¥0.011–0.050/run、feedback ¥0.058–0.085/run、reviewed ¥0.086–0.189/run（表格任务更贵，因为推理更长）。
6. **失败在哪**：① API 侧连接重置（Pilot 阶段 5 次，靠有界重试解决）；② 推理长度方差极大（总 token 2.7k→9k），任何固定输出上限都有被吃穿的概率；③ 训练时间被 LLM 延迟挤占，导致 catboost 大配置训练超时（6 次）。
7. **证据不足 / 不可外推**：n=3；两个任务都是已知的开发任务，官方 validation 早期被看过，不是全新保留集；搜索空间只有 14 个候选；两个任务都不能支持"对所有未知任务/模态均有效"的结论；一次 6 次调用的策略在 1200 s 上限下仍有 8.3% 的 trial 被截断，说明"trial 匹配"是近似的。

## 【费用】

| 项 | 次数 | tokens | 费用 |
|---|---:|---:|---:|
| Main 批次 | 60 | 232 373 | **¥1.304480** |
| Pilot 1 | 3 | 5 167 | ¥0.026786 |
| Pilot 2 | 20 | 60 571 | ¥0.320695 |
| Pilot 3 | 20 | 59 898 | ¥0.321296 |
| Pilot 4（中止） | 4 | 17 148 | ¥0.097843 |
| Pilot 5 | 13 | 41 637 | ¥0.226874 |
| 预检探针（连通性 + max_tokens 接受度） | 5 | 约 260 | 约 ¥0.002 |
| **合计** | **125** | **约 417 000** | **约 ¥2.30** |

- Main 的 ¥10 批次上限**未被触及**（实际 ¥1.30，占 13%）。
- 用量完整性：Main 24/24 运行 `usage_complete=true`；Pilot 中有 3 个运行存在 `unresolved_call`（传输失败，用量未知），已在批次表中标注，未按 0 计。
- **定价状态**：使用 `superagent/pricing.py` 中"2026-11-13 平台截图"的暂估价（V4-Pro 输入 ¥2.25/M、输出 ¥6.75/M），**未经课程官方历史价目核验**；`price_status` 字段在每条 usage 记录中都标为 `provisional_platform_price_not_course_verified`。原始 `raw_usage` 已全部保留，可重新计价。
- 模型资格：`DeepSeek-V4-Pro` 的发布日期证据尚未记录在仓库中（`verified_release_before_cutoff=True` 是当前表里的声明，需用课程原始要求复核）。

## 【待办】

1. **token 匹配消融**：在同一任务/种子/训练次数下，给 feedback 与 reviewed 相同的"每运行 token 上限"，观察预算耗尽行为与准确率。仓库目前只有批次级人民币上限，**没有每运行 token 配额**——这是实现缺口，不是参数可调项。
2. **扩大样本**：n=3 → n≥5，并加入第三个任务（当前只有表格与文本对；缺图像/多分类）。
3. **把"LLM 时间"与"训练时间"分开记账**，消除"训练被 LLM 延迟挤占"造成的 trial 截断（当前 8.3%）。
4. **验证 16384 上限是否仍足够**：目前只在 60 次调用上验证过；若换成更长的历史或 14 个候选全量描述，仍需复查 `finish_reason`。
5. **未实现的对照**：本实验只测"搜索策略"，不含"任意生成算法代码"与"未见模态"的通用性；报告必须如实写出这一边界。

## 【附件清单】

- 批次级：`runs/agent_architecture/20261117T023216973488Z/{protocol.json, summaries.json, comparison.md, analysis.md}`
- 运行级（24 个目录各一份）：`summary.json`、`trials.json`、`selection.json`、`events.jsonl`、`llm_usage.jsonl`、`data_profile.json`、`split.json`、`validation_predictions.json`
- 校验：`python scripts/verify_batch_artifacts.py runs/agent_architecture/20261117T023216973488Z` → 全部齐备，`test_accessed` 全为 false
- Pilot 目录保留在 `runs/agent_architecture/` 下（含 3 个被中止/失败的批次），未删除
- 未附：模型权重 `*.joblib`（体积）、原始数据、API key（从未写入任何记录）
