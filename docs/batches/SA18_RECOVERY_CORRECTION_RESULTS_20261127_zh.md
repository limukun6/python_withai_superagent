# SA18：将主线纠正为自主恢复，并解除一条实际误阻断

用户澄清目标是LLM介入后持续推进。上一轮把“受控条件下按规则阻断”当作主要结果，不能回答未知任务能否自主完成。本批据此修订AGENTS、恢复架构计划和自动队列。

## 已做与未做

- **已审计**：semantic_training最多两次拒绝后抛UnsupportedFormat；_train随后直接return record_input_failure，离开完整训练流程。已有局部重提案不是跨阶段RecoveryController。
- **已修正一条过度限制**：当前MetricSpec使用带类顺序的完整归一化概率矩阵，内部二分类ROC-AUC在调换事件类并选择对应概率列时数值相同，不需要额外猜一个语义正类。task_questions因此不再仅因该疑问阻断该路线。
- **已复验**：SA17真实U2原回复未修改，现在可以编译并继续到预测预检；它原来的阻断只能作为旧规则执行证据，不能继续当成正确策略的普遍结论。没有新增LLM回答或重训。
- **尚未实现**：完整跨阶段恢复控制器、结构探查与适配/换路线的统一动作状态。计划见SA18_RECOVERY_REALIGNMENT_20261127_zh.md，不能把本批单点修正说成已经完成主目标。

## 证据

固定合成概率矩阵验证四种情况（部分正确、全部打平、全对、全反），内部两种类顺序的AUC分别约2/3、1/2、1、0，均数值相等。0fit/0API，不是分类效果实验。

SA17 U2原包及原回复SHA绑定重放，64训练行仍来自原字段，metric为ROC-AUC；执行器保留疑问，记录binary_auc_invariant_under_swapping_named_class_probabilities后继续。原回复文件SHA前后相同。最终证据research/sa18_auc_continuation_replay_02；首轮回放本身通过，但其“原回复未变”检查写成了恒等比较，已改为事前/事后SHA，并保留首轮记录，不将其作为完整性证据。

**294/294相关回归通过，25.203秒，0fit/0API。** 新增覆盖AUC继续、未指定输出类别仍拒绝猜测；binary_f1等依赖正类的未决问题仍保持状态，接下来应进入恢复查证，而非以其拒绝当成完整Agent成功。两个旧含fit夹具仍排除。

内部AUC能够继续不等于所有概率输出都已明确。如果用户要求一列“正类概率”却没定义是哪一类，输出语义仍需LLM查证或提出明确标记的假设；程序不能补一个类名冒充确定答案。目标真的未提供时，也不能跨不同目标比较验证分数来推测出题意图。

## 新验收目标

下一实现批次必须包含：FailureRecord/RecoveryState、可执行查证/适配/换方法动作、LLM选择下一动作、正式入口继续执行和交付。结构元数据工具是其中部件，不再单独靠若干检查替代完整闭环。

主指标改为无人干预恢复后正确完整交付率、错误完成率、人工介入与真实资源代价。单个错误动作应被拒绝并反馈；Agent要继续选择其他可行路线。预算用尽交付最佳有效成果，带假设的部分交付不能算完整成功。网络/计费不重复发原失败请求的约定保持。

本批只有源码审计/规则修正/合成评分探针/旧回复重放，**新增API0、fit0、费用0**。只读盘点核对账本仍485/known¥8.129441/unknown16/pending0；未冻结、无运行队列。

文件：scripts/sa18_recovery_inventory.py、sa18_auc_orientation_probe.py、sa18_auc_continuation_replay.py、run_sa18_recovery_checks.py；research/sa18_recovery_inventory_01、sa18_auc_orientation_probe_01、sa18_auc_continuation_replay_02、sa18_recovery_checks_01。旧SA17原实验JSON保留，只在其报告前置本次政策/数学纠正说明。
