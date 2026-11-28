# Task1 在熟悉和陌生属性组合上的表现

课程指定验证的交付记录是 **1,561 / 2,030（76.90%）**，对应统一入口 `f214f8b33b7066fb4c670379fc6bbb4920553dab`。本轮使用课程训练包中的 8,117 行。正式入口负责独立选模、重训和导出，新增候选与采用规则在研究程序中接受对照。[交付成绩与指纹](../../research/task1_feedback_extension_1127/designated_release.json) · [本轮逐折汇总](../../research/task1_feedback_extension_1127/novelty_and_feedback_results.json)

## 根据反馈改答案能否提分

沿用已完成的五个普通外折，每行只计分一次。在每折的训练数据里取两份此前没有给 Agent 打过分的样本，先提名改动，再用第二份决定是否采用。每个候选评估某份样本时，拟合数据排除该份样本及与其完全相同的特征行。各折使用原 Agent 实际导出的模型和种子作参照。所有外折预测保存后才集中评分。[两种候选](../../research/task1_feedback_extension_1127/conditional_attribute_opportunity.json) · [模型分歧实验](../../research/task1_feedback_extension_1127/seed_disagreement_reading.json)

| 五折训练内对照 | 答对数 / 8,117 | 比原模型 |
| --- | ---: | ---: |
| 已冻结的 Agent | 6,144 | 基准 |
| 根据两份反馈，决定是否采用新的分类器或融合 | 6,144 | 0 |
| 根据两份反馈，决定是否按模型间的分歧改判 | 6,144 | 0 |
| 诊断用，固定给跨字段估计的树 25% 投票权 | 6,149 | +5 |
| 诊断用，每折都按模型分歧改判 | 6,153 | +9 |
| 诊断用，单独使用区分画像与情境作用的稀疏回归 | 6,012 | −132 |

两种自适应方案的五折都保留了原答案。强制按分歧改判能改对 103 道，也会改错 94 道，按完整特征行分组的净增益 95% 区间为 −0.22 至 +0.46 个百分点。固定给新树四分之一权重时，五折净变化为 −9、−6、+7、+11、+2 道。单看合计的正数不足以选这两个方案。

另用已保存的预测做事后回放，单独去掉两份反馈中的概率误差要求。其中一折对新融合的两份训练反馈分别多对 9 道、24 道，启用后在该折外层却少对 7 道。五折合计，放宽要求的候选决策得 6,137 道，放宽要求的分歧修正决策得 6,143 道。这解释了为什么这次保留原方案更有利。[单项规则回放](../../research/task1_feedback_extension_1127/accuracy_veto_ablation.json)

## 同一道题，训练时有没有见过相同的属性组合

普通五折已经把完全相同的特征行放在同一折，但一些记录虽然情境不同，其他一整组属性却相同。8,117 条评估记录都能在各自的训练集里找到这样的同组合样本。程序从特征推断出 567 种重复组合。另取其中 228 种，把每种组合的全部记录留在对应折的训练集之外。组合代表一组观测属性，数据没有提供可核实的个人身份。两折都给冻结的统一入口一份匿名训练包和一份只含待预测特征的包。[分组方案](../../research/task1_feedback_extension_1127/unseen_profile_stress.json)

| 同一批 3,263 道题 | 第一折答对 / 1,585 | 第二折答对 / 1,678 | 合计准确率 |
| --- | ---: | ---: | ---: |
| 原来的普通五折，训练中见过同组合 | 1,174 | 1,242 | 2,416 / 3,263（74.04%） |
| 按组合留出，训练中没见过同组合 | 1,105 | 1,200 | 2,305 / 3,263（70.64%） |
| 按组合留出，每折只猜训练集中最常见的类别 | 881 | 951 | 1,832 / 3,263（56.14%） |

逐行配对后，全新组合上多答对 189 道、少答对 300 道，净少 **111 道（3.40 个百分点）**。按 228 个组合重新抽样，差值的 95% 区间为 **−5.44 至 −1.56 个百分点**。两种划分使用的训练样本也随之变化。这个比较刻画了 Task1 在新组合上的表现，跨数据集迁移仍需独立证据。

两次完整运行分别实际训练 25 次、19 次，所有分类器调用均完成，两份导出均可在独立进程复现。第一折实际试过 TabICL，最终选择了两种树并保留原投票。第二折自主选择树与 TabICL 的组合，并对树用多个种子重训。该折三种子投票在这批陌生组合上比重训前少答对 4 道。两折共用同一入口及资源上限，每折的模型选择只使用该折的训练反馈。

## 补充模型能否帮助陌生组合

补充树使用单个属性、从训练中发现的字段关系与当次情境来预测。它不给整组画像分配专属特征，训练时也会降低高频组合的权重。程序只在训练中没见过的组合上考虑它。两折第一轮反馈都提名 100% 权重，训练侧分别多对 3 道和 11 道，第二轮却分别少对 10 道和 4 道。按既定规则，两折都保留原答案，合计 **2,305 道**。[模型与采用规则](../../research/task1_feedback_extension_1127/novel_profile_expert.json)

事先登记的固定四分之一权重对照答对 **2,313 道**，两折分别多对 2 道、6 道。逐行看，它改对 31 道、改错 23 道，净增 0.25 个百分点，按属性组合计的配对 95% 区间为 −0.15 至 +0.66 点。单用补充树也答对 2,313 道，但两折分别多对 10 道、少对 2 道，路径和稳定性均有差别。正常五折的 8,117 行都被程序识别为已知组合，按设计沿用原预测。

## 给第二种选择一次独立验证机会

初选的 100% 权重两次都未通过复核。我们给研究程序增加了一条通用决策规则。如果有较小权重在两份训练反馈上都达标，就让程序锁定其中一个，再用第三批完整留出的属性组合独立确认。两折都自主提名 25%。第一折在第三批少对 2 道，程序继续用原模型。第二折多对 3 道且概率误差更低，程序采用融合。最终得 **2,311 / 3,263（70.82%）**。它在两折分别持平、多对 6 道，净增 0.18 个百分点，配对 95% 区间为 −0.12 至 +0.54 点。事先定下的晋级条件是两折都正增长且合计至少提升 0.30 个百分点。[第三批复核规则](../../research/task1_feedback_extension_1127/novel_profile_runnerup.json) · [采用决定与结果](../../research/task1_feedback_extension_1127/novel_profile_decision_followup_results.json)

另一个独立对照复用上述已训练的成员，让程序检查原先的多种子平均对全新组合是否仍有帮助。一折本来就只用原投票。另一折两份新组合训练反馈都支持用重训前的概率，分别多对 6 道、12 道。程序只对新的组合改用它，最终 **2,309 / 3,263（70.76%）**，净增 4 道。只有一折具备修改条件。两种跟进方案单独评估，收益不能相加。[重训决策复核](../../research/task1_feedback_extension_1127/novel_seed_feedback.json)

## 运行异常和用量

首次并行运行期间机器重启，两折分别已有 20 次、19 次分类器调用，并留下 1,585 行、1,678 行临时预测。第一次串行恢复时，设置的 GPU 显存上限低于系统默认下限，导致 TabICL 两次都无法运行。这组正常交付了 2,300 / 3,263，记为资源受限对照。恢复原始显存设置后，先用已拟合的 TabICL 对 6 条未标注输入做离线推理，再用完全相同的包逐折从头运行。两折共 44 次拟合全部完成，实际试过 TabICL，取得上文的 2,305 道。[中断登记](../../research/task1_feedback_extension_1127/serial_stress_recovery.json) · [资源恢复与对照规则](../../research/task1_feedback_extension_1127/matched_catalogue_recovery.json)

本轮合计 **262 次本地拟合调用或决策器优化**。普通五折共 100 次，其中 40 次重建对照、30 次新分类器、30 次小型决策器。陌生组合实验中，首次中断耗用 39 次，资源受限的两次运行及其补充模型对照耗用 55 次。恢复资源后的两次正式入口及其补充模型对照耗用 60 次，第三批反馈增加 8 次。两棵与训练输入绑定的整训练集补充树直接复用。构造字段统计时的小型辅助估计另列。新增模型 API **0 次、0 token、¥0**。课程指定验证本轮未重新评分，正式测试与 H2 均保持封存。[训练与恢复明细](../../research/task1_feedback_extension_1127/novelty_and_feedback_results.json) · [末轮增量](../../research/task1_feedback_extension_1127/novel_profile_decision_followup_results.json)

复现时先按上次交接重建普通五折及其匿名训练包。本轮脚本默认从本机研究输出目录读取这些材料。普通入口来自冻结提交，`study_task1_conditional_opportunity` 和 `study_task1_disagreement_reading` 复建常规五折对照，`study_task1_unseen_profiles` 按整个属性组合分组，`study_task1_population_expert` 和 `study_task1_profile_runnerup` 评估新的采用规则。所需模型、课程数据和逐行预测保存在本机。[冻结入口与启动方式](TASK1_B_CONTEXT_EXTENSION_20261127_zh.md)

在仓库根目录和项目 Python 环境中运行下面的命令即可复现陌生组合实验。将 `TASK1_STUDY` 指向已具备上轮 `remaining_folds_source`、`release/inputs.json`、`seed_policy` 和 `seed_completion` 的新研究目录，`FROZEN_SOLVER` 指向 `f214f8b` 的检出目录，`TABICL_RUNTIME` 和 `TABLE_WEIGHT` 指向已校验的本机运行库及权重。两折依次运行。每个阶段在独立进程中读取上一步保存的结果。

```sh
export PYTHONPATH="$PWD:$TABICL_RUNTIME"
export SUPERAGENT_TABICL_V2_WEIGHTS="$TABLE_WEIGHT"
export HF_HUB_OFFLINE=1
python -m scripts.study_task1_unseen_profiles prepare --out "$TASK1_STUDY"
for fold in 0 3; do
  python -m scripts.study_task1_extensions entry \
    --out "$TASK1_STUDY/unseen_profile_stress_1128/$fold" --arm replicated \
    --source "$FROZEN_SOLVER" --feedback-mode seed-replication --cap 64 \
    --table-weight "$TABLE_WEIGHT" --optional-runtime "$TABICL_RUNTIME" || exit 1
done
for stage in seal grade; do
  python -m scripts.study_task1_unseen_profiles "$stage" --out "$TASK1_STUDY" || exit 1
done
for stage in build seal grade; do
  python -m scripts.study_task1_population_expert "$stage" --out "$TASK1_STUDY" || exit 1
done
for stage in nominate confirm grade; do
  python -m scripts.study_task1_profile_runnerup "$stage" --out "$TASK1_STUDY" || exit 1
done
```
