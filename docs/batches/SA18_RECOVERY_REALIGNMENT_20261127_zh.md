# 主线校正：拒绝一个错误动作后，Agent应继续自主恢复

用户2026-11-27澄清：希望LLM介入使陌生任务持续推进，而不是把“正确阻断”作为研发目标。本说明优先于旧自动队列中偏重拒绝案例的顺序；网络/计费失败不重试、既有预算与数据隔离约定不因此改变。

**当前是源码审计、首项纠偏和结构计划，不宣称完整自主恢复已经实现；本次0fit/0API。** SA17的字段疑问机制可保留为恢复输入，但它的受控阻断结果不能充当完整Agent成功率。

首项纠偏先检验二分类ROC-AUC：当前MetricSpec要求带类别顺序的完整归一化概率矩阵，调换目标类别并选择其对应概率列应保持AUC相同。先用固定合成概率验证当前实现，0fit/API；若成立，解除仅因positive_label缺失而对内部AUC的误阻断，仍检查输出概率列的明确class_label，binary_f1的正类疑问仍保留。原SA17 U2结果是旧策略执行证据，不能继续说成“内部AUC一定需要额外正类定义”。完整恢复控制器仍另行实现，不能以这个单点修正代替。

## 实际缺口

1. semantic_training虽会把错误反馈给LLM，但主要仍是最多两次提案拒绝、两次字段工具。不能保证选择了有信息增量的恢复动作，随后抛UnsupportedFormat。
2. _train读题阶段捕获DECODE_ERRORS后直接return record_input_failure，转为类先验或无交付，离开主训练循环。没有跨阶段的RecoveryController继续尝试其他可行接入路线。
3. 字段工具覆盖有限，数组键/shape/dtype/图像结构尚不可查。LLM有时只能重复输出JSON，而不是先获得新证据。
4. 缺少统一的“已失败动作、失败原因、未检查证据、可用替代路线、剩余资源、已有可交付物”状态。B已有候选失败后换方法等逻辑，应复用，不另起一套算法选择器。

## 新结构与流程

```text
接收原始任务
  → 理解目标并提出可执行动作
  → 执行器验证／执行
  → 成功：进入下一阶段
  → 动作失败：保存 FailureRecord 和已有成果
      → RecoveryController把缺失事实、已有证据、可用工具/方法、失败历史和预算交给LLM
      → LLM选择查证／改适配／改表示／换可用方法／有标记的假设分支
      → 实际执行、验证任务含义和输出契约、继续推进
      → 预算不足或没有可执行动作：交付最佳有效成果与未决假设，不伪称完整完成
```

护栏的职责是拒绝不合法的**当前动作**、给出可操作的原因；控制器的职责是选择下一条路线。不能把放宽所有验证、重复同一次错误或生成形式上成功的错误文件当成恢复。

最小接口：

```text
RecoveryState:
  stage, goal, input_contract, observed_evidence, unresolved_fields
  failed_actions, failure_kind, installed_capabilities
  remaining_requests, remaining_fits, remaining_time, available_artifacts

RecoveryAction:
  inspect | repair_adapter | revise_interpretation | alternate_method
  branch_assumption | deliver_checkpoint
  parameters, evidence_refs, expected_check, estimated_cost

while resource_available:
    outcome = execute_and_verify(action)
    if outcome.succeeds:
        save_checkpoint(outcome)
        advance_stage()
    else:
        state.record_failure(outcome)   # 不丢模型，不把失败记成功
        options = available_recovery_actions(state)
        action = llm.choose_recovery(state, options)
        reject_repeated_action_without_new_evidence(action, state)
deliver_best_verified_artifacts_with_status()
```

## 按问题选择恢复动作

| 问题 | 应继续做什么 | 不能做什么 |
|---|---|---|
| 字段、数组键、形状或图像模式未知 | 查表头/结构、结合文档改适配、实际重放预测 | 没查证就结束，或盲目改维度凑格式 |
| LLM JSON格式不符 | 已计量响应作为错误证据，在共享预算内修正提案 | 把所有内容删到合法、丢掉目标或输出要求 |
| 任务解释有疑问 | 查其他文档/工具，比较有来源的解释；将可证明的非必需项保留为注记 | 仅凭有uncertainty就结束整个Agent |
| 当前模型/编码器不可用或训练失败 | 复用B的失败记录和可用方法目录，转可执行路线 | 让一条候选失败结束整个搜索，或谎称使用了未加载模型 |
| 输出预检失败 | 将具体ID/列/编码/概率来源问题交给LLM修正，再验证 | 已有模型被丢弃，或换成无关默认格式冒充完整交付 |
| 原材料真的没定义哪个目标 | 尽可能寻找依据；仍未确定时可在总预算内保留明确标记的分支交付 | 用不同目标的验证分数高低猜出题意，或把假设写成已知事实 |

模型无法凭空知道未提供的出题意图。多分支结果必须标记partial/assumption，不可算正确的单一任务交付。但这不要求Agent在第一次发现歧义时就结束；应先用完可行的自主恢复路线。

## 接下来按完整恢复验收，不按检查数量验收

第一批实现应同时包含：结构探查工具＋RecoveryState/动作记录＋主入口接管。工具通过若干测试只是前置，不再作为主交付。

开发矩阵预登记三个可恢复案例（记录表字段不一致、数组键未明、一个已选方法不可用）及一个确实信息不足的案例。各从同一正式命令启动，中途不手改文件或替LLM选择。可恢复案例必须经历实际失败、获得新证据/改路线、继续训练并由新进程交付；缺信息案例需继续查证并输出清楚标记的未决成果。

主指标：

- **无人干预恢复后完整交付率**，包括目标、指标、原ID/输出语义正确。
- **错误完成率**，防止只是“不阻断”却输出错任务结果。
- 人工介入次数、恢复动作的信息增量、重复无效动作次数。
- 真实API/fit/时间/费用；假设性部分交付与完整交付分开。

不能跨不同目标比较分数来证明恢复正确，不能把故意无法确定目标的案例放入“可恢复成功率”分母又称拒绝是成功。是否可恢复及oracle在执行前封存。

先0API源码盘点和状态/工具检查，再用小规模正式路径校准预算。继续沿用共享最多3次Flash/路径、实际fit记账和同一真实累计账本；若这些资源不足以验证完整恢复，先如实报告预算限制，不暗中扩大额度。任何付费批次必须另有可复跑脚本与冻结协议。
