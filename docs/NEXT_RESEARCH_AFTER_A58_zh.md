# 下一阶段：按真实瓶颈推进，双方都做实现与实验

先读batches/A37_A59_round_results_zh.md。已做的事实/诚实契约/partial交付保留为可开关结构；不要把它们叠加后就宣布最佳，也不要继续只堆检查文字。沿用七天研发、两天演讲的总体安排，这里是下一轮可交接工作块。

## A主线：目标与最终选模，不只反馈

1. **先补AUC全流程**：worker已有部分score支持，但AB仍明确拒绝AUC。补score矩阵/class_order/行ID/payload/label的指纹绑定、正确重算主指标、概率导出与partial模型。不能拿相同argmax标签就推断AUC相同。A负责数据/执行/导出，B同步比较与调度分支。先用同标签不同排名的人工案例，再用一个已见真实任务换明确AUC目标验证；不是新任务泛化证据。
2. **统一finalist复核**：初始赢家和后来挑战者采用同一最低证据门槛。预留预算给前两候选的共同复核，不把初始primary领先当可靠结论。与原动态、原固定在相同目录/实际11fit/时限下做先本地0API试验，再选最有信息量的真实Flash小批。保留反例，不能保证一定提高外层分数。
3. **补时间预留**：不仅预留refit的一个fit，也保留实际时间。未知路线成本标unknown，已见路线用本任务观测估计；估计不是硬保证，最终失败仍partial。记录训练、API、序列化分别耗时。

单fit方法下的统一决选原型：

```text
reserve = two_finalist_checks + final_refit  # 共3实际fit
while remaining_fit > reserve and time_is_feasible:
    action = propose_from_same_legal_catalog(task, facts, evidence, budget)
    execute_and_verify(action)
top_two = rank_on_same_primary_rows()
for candidate in top_two:
    retrain_and_evaluate(candidate, preregistered_confirmation_split)
winner = select_using_only_matched_split_evidence(top_two)
refit_and_export(winner)
on_failure: freeze_verified_checkpoint_as_partial_without_extra_fit_or_API()
```

比较两候选必须重训到各自复核训练分区，不能把在primary训练过的模型直接拿到有重叠训练行的另一划分上评分。不同划分不是新数据集；事后primary bootstrap不能替代训练重采样或外层验证。

## B主线：扩充真正可执行的强路线与成本调度

优先补当前full=baseline的偏好/SMS/数值表格能力，以及泛用图像路线；先核验课程合法性、依赖/权重与序列化交付，不把占位模型算已激活。每个方法交付准确family/参数、内部监督fit数、内存/CPU或GPU成本、概率支持、pair语义性质。双方目录/权重指纹一致后才比较决策器。

B自行跑方法库和预算实验，A自行跑目标/证据/决选/交付实验，不变成一个人实验一个人检查。A56有新接口和匿名实时形状样例；改共用控制循环或fit计数器前用独立提交交接，不覆盖对方checkout。多分类器方法不能直接塞进当前单fit固定分配；需要先定义按实际fit单位的可行共同预算。

## 实验顺序与预计耗时

| 工作块 | 实验 | 预计/上界与验收 |
|---|---|---|
| 目标接入 | AUC人工边界+一个真实已见任务完整CLI | 0API先行，约0.5–2小时开发/小试；按实际fit登记 |
| 最终复核 | 原动态/统一finalist/原固定，先两任务 | 每路径≤11fit；按当前观察约5–25分钟/小矩阵，重方法先小试 |
| B方法 | baseline/full共同清单、不同真实任务 | 先CPU/GPU各一小试重估；同任务机器/缓存/划分固定 |
| 冻结C1 | 五任务六组local/raw/E×固定/动态 | 最多330fit/60调用；当前方法中心估计1–2小时，串行外层上界30×660秒约5.5小时；新方法须重新校准 |
| 未见H | 先冻结全局策略/源/依赖/权重/预算，再各跑完整路径 | 原H1/H2仅此时开；结果不理想不能继续把它当未见调参集 |
| 交付 | 最终模型源码自包含包、干净环境、队友机器 | A54/A57已验证当前六包；最终新增方法仍要重验 |

先完成上述通用流程，不提前做答辩。正式报告仅使用最终提交源版本的完整自主运行结果；此前开发现象不冒充冻结结果。模型使用仍保留真实共享账本与11历史未知，不等待用户无法取得的账单才做研发，也不宣称预算已核验。

## 暂不优先扩大的方向

本轮v2数值预测全空，说明应把“需要回答的可观测问题”和“可未知的数值预测”分开，而不是重新强迫涨分。可以设计typed question/reference/same-split answer，但在AUC、统一决选和强动作库之前，不再大规模追加这类提示研究。

A55只证明一个任务的局部动作优势，A58未转成最终收益，因此若继续反馈研究，应检验何时启用证据整理、是否统一验证赢家，并做完整闭环。跨任务迁移优先于反复换同一任务seed追分。
