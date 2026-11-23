# 正式入口从空目录自主选择按需推理

用同一份 6,608 行匿名标题训练包、同一种子和同一预算，打开 `--confidence-cascade` 后，Agent 自主探索并确认了词频十二层 MiniLM 与 E5 的组合。它先比较初筛预测，选词频十二层先作答，再用与初筛评估组互斥的 1,066 行确认 0.2 的阈值。确认折有 535 行可提前判断，和全量双模型都答对 754 行。这个开关只在原调度已选中两个合适的冻结编码器时生效。单模型获选或确认不通过时按原决策交付。[选择与真实运行记录](../../research/b_native_encoder_routing_1123/model_frozen.json)

**真实入口完成 9 个动作、10 次分类器拟合。** 与既有的完整双模型路径逐项核对，候选目录、动作、八次试验的评估数据指纹、概率与硬预测全部相同。两个新进程导出的开发预测也与原来逐行一致，答对 **1,237／1,693（73.07%）**。同一路由的逐段实测中，882 行只用第一个编码器。新版本未打开课程指定验证集或正式测试。已实测的 **689／940（73.30%）** 属于早先的完整双模型。[原生验收](../../research/b_native_encoder_routing_1123/results.json) · [四组实际使用量与得失](B_encoder_cascade_results_1123_zh.md) · [完整双模型的课程成绩](B_joint_coverage_results_1123_zh.md)

在本机的 A117 独立接入副本上，使用 A 的命令入口重新核验了全部 9 个动作、路由抉择和 1,693 行新进程预测。两个接入补丁链均从原始代码重新应用，所得生产文件与实际测试版本逐字节相同。真实任务本次新增 10 次分类器拟合；另有一个既有的工程回归测试确实在合成表格做了 10 次拟合，单独记录。B 接入侧 35 项、A 接入侧 12 项回归通过。本批新增模型 API 调用 0 次，新增费用 ¥0。[A 侧复核回执](../../research/b_native_encoder_routing_1123/a117_receipt.json) · [补丁链及内容指纹](../../research/b_native_encoder_routing_1123/patch_chain.json) · [工程测试计数口径](../../research/b_native_encoder_routing_1123/engineering_checks.json)

## 队友接入

- A 从提交 `08cba70` 开始，在仓库根目录依次应用已发布的 `A117_B_joint_encoder.patch`、`A117_B_single_conflict.patch` 和本次的 `A117_B_confidence_cascade.patch`。前两项已接好的副本直接用第三项。[A117 增量补丁](../integration/A117_B_confidence_cascade.patch)
- 本发布目录的独立副本先应用 `B_joint_single_conflict.patch`，再应用 `B_confidence_cascade.patch`。使用默认参数时，Agent 继续采用完整推理。选择按需推理时，在合适任务的 `superagent run` 命令上添加 `--confidence-cascade`。实验使用 `--scheduler ab-adaptive-complement --cover-untried-encoder --preference-catalog deep-e5-familiarity --planner local --fit-limit 10 --seed 18643`。[B 侧增量补丁](../integration/B_confidence_cascade.patch)

在发布目录执行 `python scripts/verify_b_native_encoder_routing.py`，可核对冻结顺序、物理拟合日志、跨进程预测、A117 接入与补丁指纹。
