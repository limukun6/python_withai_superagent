# v0.31：共同架构契约

v0.31修复v0.30在空历史时默认标注Accuracy的问题：初始记忆、路线和latest_result显式沿用TaskSpec，画像与任务契约冲突会被拒绝。

这一版先将可替换决策层接到相同执行和评估底座，支持任务指定Accuracy或Macro-F1，并增加从权威记录重建的假设记忆。不增加真实API调用，不替换已打包v0.28或待上传分支。

## 共同接口

| 契约 | 内容 | 使用位置 |
| --- | --- | --- |
| TaskSpec | 数据模态、训练类别域、主指标、输入契约 | 运行前固定，保存task_contract.json |
| ExperimentProposal | 候选及实际参数、理由、假设、过去实验引用、数值预期、回访条件 | 训练前写入experiment_proposals.json及冻结事件 |
| ExperimentResult | 对应提案、执行状态、真实指标、资源、预期核对 | 执行后写入experiment_results.json；失败无主分数 |
| DecisionState | 脱离原始数据的画像、可见历史、候选、剩余次数和记忆 | 决策实现只接收独立快照 |

`DecisionPolicy.propose(state)`返回经过校验的ExperimentProposal。现有本地/LLM策略通过ExistingPolicyAdapter接入，`benchmark.run_one(..., decision_policy=...)`可传入另一实现。控制器再次核对候选是否可用、参数是否相同及引用是否仅来自此前实验；决策层不能通过修改快照改变权威目录或日志。

本轮用两种脚本决策实现验证相同执行器无需改动。这是接口检查，不是Single/Multi-Agent性能比较。真正Planner+Reviewer的职责、触发条件和等预算比较尚未完成。

## 任务指标

在manifest中声明`"metric": "accuracy"`或`"metric": "macro_f1"`。未声明时沿用Accuracy。当前不支持ROC-AUC、Log Loss、其它F1聚合方式，未知名字会明确拒绝，不静默改成Accuracy。

Macro-F1对原训练数据中所有类别的逐类F1取等权平均，零分母贡献为0；确认划分即使缺一类也使用相同冻结类别域。未知类别、空向量、形状/长度错误会失败。定义采用[scikit-learn f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)的`average="macro", labels=training_classes, zero_division=0`，行为在当前安装的1.7.2版本上验证。

- `search_score`、`train_score`、`score_gap`属于声明的主指标。
- `search_accuracy`、`train_accuracy`、`generalization_gap`仍是真实Accuracy及其差值，不把F1写进旧字段。
- 搜索、路线优先级、候选生成、参数参照、确认选型、最终报告均使用主分数。确认不完整时保留主搜索结果，缺失F1不会借用Accuracy。
- `official_validation_score`和CLI的`validation_score`标注主指标；`official_validation_accuracy`/`validation_accuracy`继续保留真实准确率。
- 逐行“改对/改错”是正确性计数，描述Accuracy变化；主指标为Macro-F1时，它可以与主分数升降方向不同。反馈明确区分。

## 假设与路线记忆

`hypothesis_memory.json`从已保存提案和执行器的hypothesis_check重建，不让LLM改写旧测量。状态包括：

- measurement_supported：本次测量达到预先声明的数值预期。
- measurement_not_supported：本次测量未达到数值预期，保留原假设和结果。
- not_evaluable：执行失败或缺少可比较参照，不能按0判断。
- no_numerical_expectation：有文字假设，但没有可机械核对的阈值。

所有状态都不证明自然语言因果解释。旧路线最佳、失败、暂停和回访条件继续保留，不因为一次低分删除路线。记忆仅属于当前任务的一次运行，不跨任务沿用分数。无反馈策略不获得结果派生的记忆、状态或分数。

最终模型仍先确认、完整重训并冻结，再评分已有开发验证集。正式测试数据继续封存。
