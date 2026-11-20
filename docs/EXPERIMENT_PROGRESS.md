# 实验持续工作记录

当前总计划：nine_day_two_track_plan_20261120_zh.md。两位成员均用 Codex，A 管任务全流程/证据，B 管方法与搜索。

## 每次继续的约定

1. 先读 AGENTS.md、总计划和本记录，核对 Git 与已完成产物。
2. 每小批先写 docs/batches/<编号>_plan.md 与可复跑脚本，写明资源、对照、验收和输出；然后才运行。
3. 逐批落盘 protocol/results/completion；失败保留，不覆盖。API 失败停止，不自动重试。
4. 每次对话报告：瓶颈/结构、做了什么、结果支持范围、耗时/拟合/API、下一步。区分计划、运行中、完成。

## 当前进度：A 线三小时计划内批次已推进（2026-11-21）

- 用户明确此前做 B、现在切换 A；历史文件编号保留，我们当前只主改 A 的任务/证据/入口/交付，队友主改 B 方法与调度。
- 已发布共同基线 `0b112d823a1991686281fdc88602e758e29a4202`，远端分支 `codex/expanded-feedback`。
- 计划：`batches/A_next_3h_20261121_plan.md`；结果：`batches/A_window_20261121_results.md`。这是计划内批次实际成果，不宣称已经持续运行满三小时。
- A-05 开发18/24→24/24，预留10/12→12/12；A-06 4/12→12/12；A-07五真实路径完成、35fits/97.21秒；A-08边界28项通过，真实B调度尚未接入。
- 目标来源可追踪，完整指标/正类身份核对，导出前行/类/概率校验已实现；导出故障2/9→9/9，三个旧模型经最终导出器重载一致，无新增训练或外层重评分。
- 50项受影响回归通过，21fits；最终CLI五项通过，17fits/5模拟HTTP；本窗口累计73真实fits，真实API0，11条未知历史请求仍未对账。
- history-evidence-v3已更新并检查两组共享字段一致，旧A2状态不能冒称v3的新证据。图像仅sklearn digits，不能声称自然图像泛化。
- 所有小批计划/runner已保存，紧凑证据在research/a_window_20261121；给B的接入说明docs/A_TO_B_NEXT_HANDOFF_20261121.md。下一步接B适配器的匿名完整验收，再重新冻结A2状态并按账本条件启动真实模型实验。

## A2-02：v3真实历史状态重建完成

- 运行前协议batches/A2_02_plan.md，复用scripts/run_a2_state_study.py prepare；生产源码提交1d57fd4，source_unchanged=true。
- 四任务12次真实fit、12冻结状态、97.100秒、真实API0；两组共同字段逐项相同。原始目录runs/a2_02_v3_states，脱敏JSON在research/a2_02_v3_state_preparation。
- 仅完成prepare，真实ask及动作质量评价仍未执行。11条历史未知用量仍在，ready_to_call=false。
- 加上前述73fit，本轮累计85真实分类器fit，其中流程/准备47、回归38；5模拟HTTP、真实API0。准备批未打破三小时计划49次实验fit上限，回归另登记。
- 首轮代码/结果已推送1d57fd4；续跑说明A_LINE_RESUME_20261121.md。后续文档/证据提交不修改生产源码，保持这12个状态的生产指纹。

## A2-03：封存决策的公平动作评价器

- 计划/结果：batches/A2_03_plan.md、batches/A2_03_results.md。tools/a2_action_evaluation.py及run_a2_action_evaluation.py支持validate-only，真实完整24提案与fixture严格区分。
- 两固定前缀的确定性动作小试：4真实fit、8逻辑评价、4复用，16.643秒、API0；同选结果一致，不同选两划分差值方向相反。是评价器验收，不是LLM反馈收益。
- 13项输入拒绝检查通过；最终preflight通过，原始目录runs/a2_03_evaluator_smoke和a2_03_input_checks，脱敏证据research/a2_03_evaluator。
- 本轮新增4fits，此前85累计89，真实API仍0。未修改superagent生产源码，A2-02状态仍保持原指纹。真实提案、动作质量正式结果与B调度自主验收仍待后续。

## A1-01：任务语义与实际训练一致性

- 状态：已完成 baseline、修复、fixed 与回归，2026-11-20。
- 计划：batches/A1_01_plan.md。
- 脚本：scripts/run_a1_semantics_batch.py。
- 目标产物：主仓库 runs/a1_01_baseline 与 runs/a1_01_fixed。
- API：0；runner 完整分类器拟合：0，只拦截训练入口。
- 结果：baseline 1/9（1.248 秒），fixed 9/9（1.218 秒）；四个隐藏翻转 recipe 在 symmetric 下均被拒绝，preference 兼容，歧义概率不确认。不是性能或独立测试集准确率证据。
- 回归：91 项通过，runner 墙钟 42.418 秒；tests.log 内测试执行 40.742 秒。包含小模型拟合，未逐项计数，不将它们计为性能实验的零拟合。首次手工回归误填了不存在的 tests.test_recipe_models，命令失败；已改为实际存在的测试清单，完整重跑通过，未重试 API。
- 修改：task_semantics、worker、benchmark、confirmation、finalization 及任务语义测试；执行边界同时覆盖注册 recipe、搜索和 refit。未实现真正 symmetric 算法，新候选应与 B 协同。
- 交接：batches/A1_01_results.md；research/a1_semantics_guard 保存可共享证据。原始运行目录含 a1_01_regression。当前工作区未提交，protocol 文件哈希标识实际运行源码。
- 后续：A1-02 核对配置对象与证据引用；A2 付费实验尚未开始。

## A1-02：配置事实与测量引用

- 状态：已完成 baseline、修复、最终 fixed 与冻结源码回归，2026-11-20。
- 计划：batches/A1_02_plan.md；脚本：scripts/run_a1_evidence_batch.py。
- 资源：真实 API 0、机制实验完整拟合 0；回归夹具另计。
- 修改范围：estimator_facts、measurement_claims、prose_claims 与相关测试。
- 结果：baseline 6/19（1.198 秒），最终 fixed 19/19（1.190 秒）；合法配置/测量仍可确认，错 trial/split/metric 不再借用别处数值。未知、失败试验、否定/未来与无作用域数字不伪装成已确认事实。
- 回归：123 项通过，测试执行 40.905 秒、runner 墙钟 42.614 秒，source_unchanged=true。首轮回归期间实现改变，已保留但不作为冻结证据，最终完整重跑通过；真实 API 调用始终为 0。
- 交接：batches/A1_02_results.md；research/a1_evidence_binding 保存 baseline、fixed、regression 和 runner 快照。原始最终目录为主仓库 runs/a1_02_fixed_release 与 runs/a1_02_regression_final。
- 限制：这是已知故障开发回归，不是独立泛化准确率，也没有真实性能提升证据；没有更改 LLM 输出契约或接入下一请求。
- 下一步：A1-03 核对结果进入下一请求；本批不宣称已完成自主纠错闭环。

## A1-03：执行核对进入下一请求

- 状态：baseline、fixed 与最终回归均完成。
- 计划：batches/A1_03_plan.md；脚本：scripts/run_a1_loop_batch.py。
- 本批用受控缺失元数据和确定性模拟客户端走真实 run_one，不证明真实 LLM 的纠错或性能收益。
- 结果：baseline 4/9，11.321 秒；fixed 9/9，11.120 秒；各 6 次真实拟合、4 次模拟调用，合计 12 拟合/8 模拟调用/真实 API 0。fixed 两次请求的共同上下文逐项相同，仅 E 增加 checks/memory。
- 回归：129 项通过，测试执行 40.543 秒，runner 墙钟 42.244 秒，source_unchanged=true。首轮新增六个测试夹具遗漏 MetricSpec 的 labels 参数而失败，已修复并完整重跑；未重试任何 API。
- 修改：contracts 保留 fitted estimator 与配置核对，checked_history_policy 显式投影派生检查，hypothesis_memory 按 trial 保留核对，benchmark 不再要求 assessment 才审计，explanation_audit 包含 hypothesis。
- 证据：batches/A1_03_results.md；research/a1_feedback_loop 归档协议、状态、实际提示、核对、结果与回归日志；原始目录 runs/a1_03_baseline、a1_03_fixed、a1_03_regression_final。
- 限制：下一候选由模拟客户端预登记，两组候选相同，反馈只改变其说明；没有证明 LLM 会换动作或提高准确率。本批无复核与最终 refit，不是完整交付性能运行。
- 后续：A1-04 正式 CLI 的策略/账本接入；之后再计划真实 A2。

## A1-04：正式 CLI 与费用/失败链

- 状态：baseline、实现、最终入口验收和宽回归均完成。
- 计划：batches/A1_04_plan.md；脚本：scripts/run_a1_cli_batch.py；入口说明：history_planner_cli_zh.md。
- 真实 API 均为 0。模拟 HTTP 沿用真实 ExperimentLLM 的 SSE 解析、请求上限和 SQLite 记账，模拟费用不计实际用量。
- 首轮 fixed 已完成训练与故障检查，剩一项配置测试写成 self-review 而实际枚举为 self_review；修正后新目录完整重跑，旧失败保留。
- 最终结果：5 项入口测试全部通过，29.218 秒，17真实拟合/5模拟HTTP；累计含前轮为34真实拟合/10模拟HTTP；真实API始终0。两模式完成三划分确认/refit/导出，模型重载各15条预测一致。
- 故障：timeout与缺usage均费用未知并阻止后续任务调用；无效JSON保留已知模拟token费用；均首请求后停止、无随机恢复，保留anchor模型并写stopped outcome。
- 回归：141项通过，43.281秒，source_unchanged=true；首次回归默认编码读取JSON在中文路径出错，修为UTF-8后完整重跑。回归夹具拟合/模拟调用另计未逐项汇总。
- 交接：batches/A1_04_results.md；research/a1_cli_entry归档协议、实际请求日志/输出摘要和回归。原始成功目录runs/a1_04_fixed_final与runs/a1_04_regression_final；模拟账本不等于真实账本，不导入真实费用。
- 下一步：完成记录后制定 A2 真实调用协议；B 的动态调度接入口与强方法线仍另行推进。

## A2-01：真实任务冻结状态（付费调用尚未启动）

- 已按运行前计划 batches/A2_01_plan.md 完成prepare，脚本scripts/run_a2_state_study.py。
- 四个真实任务各3次搜索，共12次真实拟合、12个状态，50.225秒、API 0；SMS按macro_f1，其余accuracy。训练外层留出及正式validation/test不进入状态。
- 输出主仓库runs/a2_01_states；源码指纹冻结、R/E共享字段一致、source_unchanged=true。
- 用户确认11条历史未知费用未对账；本机没有已核验真实累计账本。真实ask暂停，不能用模拟账本或空账本绕过。ask脚本和24请求上限已写好，待实际历史对账后preflight。
- 当前协调任务：我们A，队友B；按AB_INTEGRATION_V1.md提供共同Git基线和输入/动作样例，优先接通匿名小任务再做大实验。

## A/B 交接与发布

- 已写docs/AB_INTEGRATION_V1.md，明确共同候选、实际11fit/3请求、完整匿名验收、文件归属和仍待接入的Scheduler。
- tools/export_b_handoff.py从已执行run导出任务、候选/历史、带split/row/hash的逐行预测与预算；examples/ab_v1有匿名真实训练产物导出的输入，以及explore/confirm/stop格式样例（样例动作不是算法收益证据）。
- 新增证据完整性测试1项通过：保留预测行序、预算和refit预留，篡改预测会被拒绝。原A1最终回归141项及CLI5项仍为对应源码哈希的记录。
- A2本地prepare摘要/状态归档research/a2_state_preparation；真实调用依然0。准备将本轮A线与计划/证据发布到codex/expanded-feedback，精确提交号以Git交接消息为准。
