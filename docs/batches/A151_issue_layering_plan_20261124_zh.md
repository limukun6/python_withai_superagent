# A151 计划：把知识纠正与执行修复分层（先零成本验收）

2026-11-24 接续自 `DEEPSEEK_A_LINE_HANDOFF_20261124_ZH.md` 第 8 节第一项。批次号取 A151：A148 是已决定不启动的 44fit/12 调用扩大实验，A149（短程计划）与 A150（幂等续跑）按交接保留给后续。

## 1 要解决的问题

A147 的短信受控例里，模型按配置和定义就能核清一处 C 方向错误，却选了一个 `diagnose` 动作去重新训练。结果新动作在两份内部分区上都低于原 `improve`（−3.1840pp 均值）。

这不是"诊断动作无用"，而是**当前 issue 模型把一切矛盾都当成需要执行才能关闭**：

- `issue_for` 给每条矛盾的 `required_postcondition` 都写着"先有一次确认的同一命题，再有匹配的执行证据"。
- `confirm_execution` 只认"完成 + metric identity 确认 + configuration fact 确认"的试验行。
- 于是"任务类型说错了"这类纯知识错误，也必须占用一次训练名额才能关闭。

## 2 改动范围（已实现，默认关闭）

| 项 | 内容 |
|---|---|
| 分类 | `grounding_issues.claim_class` 给出 `knowledge` / `execution` / `measurement` |
| 关闭通道 | `knowledge` → `registry_check`；`execution` → `execution_evidence`；`measurement` → `measurement_evidence` |
| 新状态 | `resolved_semantic`（由已核对的复述关闭），与 `resolved`（由执行证据关闭）分开记录 |
| 开关 | `--issue-layering off|by-class`，默认 `off`；只允许配 `--proposal-contract grounded-v1` |
| 默认不变 | `off` 路径逐字节保持旧行为（旧 6 项测试未改、全过） |

分类依据是**命题种类**，不是数据集名：`value` / `compare` / `parameter_relation` 三种现存形式都引用已存在的事实，因此都属知识类；`capability` / `object_fit`（执行）与 `uplift` / `robustness` / `causal`（测量）是保留类别，目前没有生产者会发出。

## 3 本批预算

| 项 | 上限 |
|---|---|
| 实际分类器 fit | **0** |
| 学校 API 调用 | **0** |
| 费用 | **¥0** |

全部为离线单元验收 + 封存响应回放。真实小试不包含在本批；若后续需要，按交接建议另登记，上限 6fit/6 调用，且必须先满足第 5 节门槛。

## 4 验收目标（程序 oracle，不是人工金标准）

1. **知识纠正不再必需新 fit**：矛盾命题被正确复述后直接 `resolved_semantic`，且 `new_fit_required=false`。
2. **执行类仍待执行**：失败试验不关闭、匹配证据才关闭。
3. **测量类不被无关成功试验关闭**：一份"完成 + identity + fact 都确认"的试验行也关不掉测量类。
4. **unknown 不被误关**：引用未知事实的矛盾不生成 issue；未知事实保持 unknown。
5. **菜单与选择不漂移**：同一封存响应在 `off` 与 `by-class` 下选出的候选相同。
6. **回放不冒充新实验**：封存响应回放只验机制，不计算自然错误率、不产生新的线上节费声明。

## 5 停止规则

- 任一验收项失败：本批停在零成本阶段，不进入真实小试，保留失败原件。
- 若分层需要改动候选菜单、调度动作集合或任务名路由：停止并重新设计，不在本批夹带。
- 回放出现"知识类被重新打开"或"测量类被关闭"：视为实现错误，修版本号后重跑本批。

## 6 交付物

- 代码：`superagent/grounding_issues.py`、`grounded_proposal.py`、`ab_flash.py`、`ab_run.py`、`cli.py`
- 测试：`tests/test_issue_layering.py`（11 项）
- 运行：`scripts/run_a151_issue_layering.py`，输出 `research/a151_issue_layering/`
- 报告：`docs/batches/A151_issue_layering_results_20261124_zh.md`

## 7 明确不支持的说法

- 不支持"分层提高了任务分数"。本批 0 拟合、无外层评价。
- 不支持"消除了 A147 的负结果"。A147 原始数据不改写。
- 不支持"知识类错误从此都能自动关闭"。只覆盖当前协议的三种命题形式；因果与改进类仍在测量通道。
