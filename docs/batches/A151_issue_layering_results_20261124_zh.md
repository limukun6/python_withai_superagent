# A151：知识纠正与执行修复分层，零成本验收通过

2026-11-24。接续 `docs/DEEPSEEK_A_LINE_HANDOFF_20261124_ZH.md` 第 8 节第一项。本批 **0 次分类器拟合、0 次学校 API 调用、¥0**；0.985 秒。没有启动 A148，没有改共同默认方案。

## 瓶颈、结构与证据

| 瓶颈 | 本轮结构 | 实验 | 支持到哪一步 |
|---|---|---|---|
| A147 暴露：纠正一个已知定义也要占一次训练 | issue 按命题种类分层；知识类由已核复述关闭，执行类等执行证据，测量类等新测量 | 5 项程序 oracle + 6 条封存 A146 响应回放 | 机制层：3/3 受控纠错改为免 fit 关闭，候选选择不变；**没有外层提分证据** |
| 旧模型把所有矛盾都写成"需要匹配执行证据" | `resolution_channel` 与状态 `resolved_semantic` 分开记录 | `off` 路径逐字节保持旧行为 | 工程层可回退 |

## 改了什么

| 文件 | 改动 |
|---|---|
| `superagent/grounding_issues.py` | 新增 `claim_class`（knowledge/execution/measurement）、`CHANNEL`、`resolved_semantic`；`issue_for/address_issues/confirm_execution` 增加 `layering` 参数 |
| `superagent/grounded_proposal.py` | `issues_for/augment_context/assess` 透传 `layering`；开启时给模型追加一句规则说明 |
| `superagent/ab_flash.py` | `context_for/validate_proposal/FlashPicker` 透传 |
| `superagent/ab_run.py` | picker 构造与 `confirm_execution` 调用透传 |
| `superagent/cli.py` | 新增 `--issue-layering off|by-class`，默认 `off`；只允许配 `--proposal-contract grounded-v1` |
| `tests/test_issue_layering.py` | 新增 12 项测试 |

分类依据是命题形式而非数据集名：`value` / `compare` / `parameter_relation` 都引用已存在的事实，因此都是知识类；`capability` / `object_fit`（执行）与 `uplift` / `robustness` / `causal`（测量）是保留类别，目前没有生产者会发出。

**默认行为没变**：`--issue-layering off` 是默认值，旧 `test_grounded_proposal` 的 6 项测试一行未改、全部通过；`off` 时 issue 里不出现 `issue_class`/`resolution_channel`/`new_fit_required` 字段。

## 程序 oracle（5/5 通过）

| 用例 | 期望类别 | 期望关闭 | 实测 |
|---|---|---|---|
| 任务类型说错 | knowledge | resolved_semantic | 一致，`new_fit_required=false` |
| 保留执行类 | execution | 匹配证据才 resolved | 一致；失败试验下回到 open |
| 保留测量类 | measurement | 保持 addressed_pending_check | 一致；"完成+identity+fact 都确认"的行也关不掉 |
| 引用未知事实 | 不生成 issue | — | 无 issue，未知保持未知 |
| `layering=off` | 无新字段 | 旧 postcondition 原文一致 | 一致 |

## 封存 A146 响应回放（6 条，零 API）

| 任务/条件 | 封存动作 | 封存状态 | `by-class` 状态 | 候选是否变化 |
|---|---|---|---|---|
| coupon/natural | improve | 无 issue | 无 issue | 否 |
| coupon/controlled | improve | addressed_pending_check（`exact_bound_fact`） | **resolved_semantic**（knowledge，免 fit） | 否 |
| sms/natural | coverage | 无 issue | 无 issue | 否 |
| sms/controlled | diagnose | addressed_pending_check（`inverse_C_between_bound_configurations`） | **resolved_semantic**（knowledge，免 fit） | 否 |
| flowers/natural | coverage | 无 issue | 无 issue | 否 |
| flowers/controlled | coverage | addressed_pending_check（`exact_bound_fact`） | **resolved_semantic**（knowledge，免 fit） | 否 |

汇总：`oracle_passed=5/5`、`candidates_unchanged=6/6`、`closed_without_fit=3`、`knowledge_still_open=0`。

短信那条正是 A147 的起因：`inverse_C_between_bound_configurations` 属于知识类，现在只需一次已核复述即可关闭，不必再训练一个诊断候选。

## 一个必须写清的行为后果

分层**没有禁掉 `diagnose`**，但改变了它成立的条件：协议要求 `purpose=diagnose` 必须指向一个 open issue。知识类被免 fit 关闭后，如果一条轨迹上只存在知识类问题，那么"再诊断一次"就没有可引用的 open issue，该提案会被契约拒绝。

本批已把这个性质写成测试 `test_diagnose_still_needs_an_open_issue`。它意味着：**下一次真实小试里，诊断动作的可发起理由会变少**；这是预期的资源纠偏，不是静默删除动作。

## 不能说的话

- 不能说"分层提高了任务分数"：本批 0 拟合、无外层或正式测试评价。
- 不能说"消除了 A147 的负结果"：A147 原始记录未改写，`−3.1840pp` 仍有效。
- 不能说"知识类错误以后都能自动关闭"：只覆盖当前三种命题形式；因果与改进类仍在测量通道，且不会被无关的成功试验关闭。
- 回放不是新 LLM 实验：不计算自然错误率，也不产生新的线上节费结论。

## 资源与账本

- 实际分类器拟合 **0**；学校 API **0**；费用 **¥0**。
- 唯一真实账本 `runs/a129_B_ledger_return/private/ledger.sqlite` 未改动，SHA-256 仍 `5b1f5c58…`，387 请求 / known ¥7.0165495 / unknown 12 / pending 0。
- A 累计拟合仍 1644（本批 0）。A 持有空闲。
- 回归：15 个相关模块 **94 项通过**，0 失败。

## 复跑

```powershell
Set-Location -LiteralPath 'D:\codex\python_withai\runs\integration'
$py = 'D:\codex\python_withai\runs\a79_encoder_assets\venv\Scripts\python.exe'
& $py -m unittest discover -s tests -p "test_issue_layering.py" -q
& $py -m scripts.run_a151_issue_layering     # 输出目录已存在时会拒绝覆盖
```

证据：`research/a151_issue_layering/{protocol,results,completion}.json`，其中 `protocol.json` 记录 7 个源码文件的 SHA-256 与预登记期望。
