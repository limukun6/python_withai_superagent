# Task2 的一次完整自主运行：课程指定验证 689/940

本地 Agent 从空目录读取匿名的 6,608 对训练样本，在同一目录里自主探索词模型、六层 MiniLM、普通十二层、带措辞熟悉度特征的十二层和 E5。最后选择 E5 与带新特征的十二层等权组合，重训两者并导出模型。课程指定验证集答对 **689/940（73.30%）**；另一个已反复使用的训练内开发保留区答对 **1,237/1,693（73.07%）**。两个分数来自同一份冻结模型。训练包没有使用其余 1,693 对开发保留样本，正式课程测试集仍封存。[完整运行回执](../../research/b_joint_opening_1123/cold/results.json) · [课程验证逐行预测，仅含行号与模型答案](../../research/b_joint_opening_1123/cold/designated/predictions.csv)

## Agent 怎么选的

种子 18643 此前已用于内部研究，新运行用于复现已提出的策略。运行开始后全部动作由同一程序产生。首轮六层单模型答对 924/1,295。带新特征的十二层和 E5 单独答对 915 和 902，组合却答对 931。在另一份 1,314 对确认折，同一组合答对 943，两个成员及原先领先的六层分别为 927、932 和 935。Agent 因此重训两个成员。5 次探索、3 次确认、2 次完整重训，共 **10 次实际拟合、0 次模型 API 调用、0 个 API 输入或输出 token、¥0 新增 API 费用**。八个试验输出全部重现了此前冻结的逐行预测。模型、顺序和开发预测先冻结，再对课程指定验证评分。课程验证两次独立导出完全一致。[动作与确认比较](../../research/b_joint_opening_1123/cold/model_frozen.json) · [课程验证预测冻结记录](../../research/b_joint_opening_1123/cold/designated/prediction_seal.json)

**另一个种子作出了不同决定。** 在种子 40531 的训练划分上，带特征十二层答对 950/1,323，普通十二层为 913。允许额外探索未尝试编码器后，Agent 也实际试了 E5；该次仍交付原单模型。旧路径用 5 次拟合、新路径用 6 次，两条路径在开发保留区同为 **1,232/1,693（72.77%）**，逐行预测一致。这个未晋级的种子没有查看课程验证。二者的课程验证结果不能平均，也不能据此推断不同种子总有相同增益。[同种子真实对照](../../research/b_joint_opening_1123/coverage/results.json) · [改名、移除权重、缩减预算与损坏证据的决策检查](../../research/b_joint_opening_1123/feedback/results.json)

此前旧研究适配器在相同规模的课程验证中为 685/940（72.87%）。本次多答对 4 对，训练来源、代码及选择流程不同。C1 六组中五组的 668/940 是固定旧版的另一项对照。此次官方验证没有用于调权或选择新方案。H 迁移与最终全组统一提交仍按双方共同冻结的版本执行。[旧研究适配器记录](B_verified_preference_1123_zh.md) · [本次课程验证混淆矩阵与训练范围](../../research/b_joint_opening_1123/cold/designated/results.json)

## 团队整合与复现

独立模型包在项目目录外使用隔离 Python 启动，1,693 行开发预测和 940 行课程验证预测均与正式入口逐字节相同。包内含分类头、词频预处理和项目源码，另需通过指纹核验的 E5 与十二层权重。跨进程验收没有新增训练和 API 调用。[独立模型包回执](../../research/b_joint_opening_1123/cold/standalone_verified.json)

另附一个针对 A 侧提交 `08cba704a49f4915af1201fbe718c96b6d1cbb40` 的 Git 补丁，仅涉及 11 个生产文件和 4 个相关测试文件。实际在 A 的干净检出上应用过，49 项集成相关测试通过，用 A 接口加载 B 冻结模型，对两份特征包导出的全部 2,633 行都相同。A 的原生目录隔离、已修正的文字核对和十二层公开版本号 `a50ef00143b4d5391434df20ae11632588ac25be` 均保留。B 历史运行记录中的旧版本号保留原样；两边实际使用的配置、分词器和权重三份文件的大小与 SHA256 均已逐一核对。此次 A 侧检验只加载已训练模型。[可应用的 A117 补丁](../integration/A117_B_joint_encoder.patch) · [A 侧移植与资产回执](../../research/b_joint_opening_1123/integration.json) · [A 侧收到原 C1 与标注的记录](https://github.com/limukun6/python_withai_superagent/blob/08cba704a49f4915af1201fbe718c96b6d1cbb40/docs/batches/A117_B_C1_D3_received_20261123_zh.md)

从这个发布包运行 `python scripts/verify_b_joint_delivery.py` 可离线核对成绩、预测行序、动作、成本和补丁指纹。在上一批 15 次新拟合之后，联合目录研究本批再用 **22 次**：种子 40531 的原路径 5 次、新路径 6 次，旧种子补确认 1 次，完整从零复现 10 次。其它回放、导出和兼容性核对均为 0 次。团队此前的付费用量独立记账。[补确认和预算回执](../../research/b_joint_opening_1123/missing_confirmation/results.json)

使用已配置的本地模型权重，可从匿名训练包重放正式动作。课程验证评分由 `scripts/run_b_cold_coverage_validation.py` 在模型及预测冻结后执行。

```bash
python -m superagent run "$training_packet" --portfolio ab-v1 --planner local \
  --scheduler ab-adaptive-complement --preference-catalog deep-e5-familiarity \
  --cover-untried-encoder --seed 18643 --fit-limit 10 \
  --trial-timeout 300 --time-limit 1200 --run-dir runs/task2_joint
```
