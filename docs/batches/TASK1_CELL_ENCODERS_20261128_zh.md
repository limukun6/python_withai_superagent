# Task1 类别编码器与自主融合实验

Task1 的课程指定验证成绩仍为 **1,561 / 2,030（76.90%）**，对应可复现版本 `f214f8b`。本轮在训练集上完成了两项实验。统一入口自行比较编码器和融合权重，四次均保留已经交付的模型。用同一批训练反馈另测了按置信度分流的办法，也未达到进入下一阶段的条件。[评分与决策记录](../../research/task1_feedback_extension_1127/cell_encoder_results.json)

## Agent 如何选择

统一入口的可选模式为 `--feedback-reflection encoder-portfolio`。Agent 先按原流程探索，训练并保存自己的胜出模型，再判断任务类型、约束、训练数据中的类别数量和剩余拟合次数是否适合做编码器对照。调用方只提供训练包、无标签预测包、资源目录和预算。任务名、字段名、候选模型名及融合权重都不进入启动参数。[统一入口](../../superagent/autonomous_run.py) · [选模代码](../../superagent/cell_encoder_review.py)

这次可用库中有 MiniLM-L3、MiniLM-L6、MiniLM-L12、E5-small、BGE-small、MPNet 和 DistilRoBERTa。Agent 按本机模型文件的结构和内容指纹识别兼容的编码器。它还可以只从训练表里学习“哪些取值经常一起出现”。先对每列各个取值建坐标，再用相同的 CatBoost 分类器，连同原始类别一起拟合。导出文件保存每列的取值和坐标，遇到训练中未见的取值就使用该列的训练中心，同时保留原类别。更早的文本方法编码的是整块表格，这次测的是每个取值。[本地模型识别](../../superagent/local_sentence_backbone.py) · [类别表示](../../superagent/cell_encoders.py) · [MPNet 模型卡](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) · [DistilRoBERTa 模型卡](https://huggingface.co/sentence-transformers/all-distilroberta-v1)

在每次初选中，Agent 比较了 145 种预测方案，包含原模型、单个编码器、两个编码器等权组合，以及它们与原模型的不同混合比例。这些方案复用同一批逐行预测。初选固定编码器及比例后，要求两个互不重叠的训练内确认划分都比原模型好，加权平均至少提高 0.30 个百分点，同时类别概率的平方误差不能上升。复核时按原交付模型实际使用的随机种子重建对照。[预登记规则](../../research/task1_feedback_extension_1127/cell_encoder_tournament.json)

| 训练内外层划分 | Agent 初选 | 两个内层确认各净多答对几行 | 最终外层正确数，原版与新版相同 |
| --- | --- | --- | --- |
| 常规划分 0 | BGE-small 与 MPNet 各 12.5%，原模型 75% | −1、−2 | 1,253 / 1,623 |
| 常规划分 2 | E5-small 与 MPNet 各 50% | −12、−25 | 1,226 / 1,628 |
| 整组留出未知属性组合 0 | DistilRoBERTa 12.5%，原模型 87.5% | −4、−2 | 1,105 / 1,585 |
| 整组留出未知属性组合 3 | 原模型 100% | 初选即停止 | 1,200 / 1,678 |

常规划分合计 **2,479 / 3,251（76.25%）**。整组留出合计 **2,305 / 3,263（70.64%）**。两类划分的样本可能重合，分别计分。每一组的最终预测文件均与对应原版逐字节相同。“未知属性组合”指整种属性组合没有进入该组的训练。单独看 21 列，四组外层样本里的每个取值都在各自的训练列中出现过。这给出了一个更具体的提分方向，即学习熟悉取值组合在一起时如何改变预测。它仍需新的独立实验。[四组对照与特征覆盖记录](../../research/task1_feedback_extension_1127/cell_encoder_results.json)

## 只在原模型拿不准时听编码器

第二项研究复用了已经提名的编码器和逐行训练预测，新增拟合与 API 调用均为 0。Agent 可以学习只在原模型前两类概率接近、双方预测相反时融合，也可以要求编码器比原模型更有把握。只在第一批训练反馈上选择阈值和权重，随后原样应用于两批确认数据。一个划分从初选就选了原模型，两组没有选中分流规则。仅常规划分 2 选中了分流。两批分别改对 35、29 行，同时改错 47、54 行，共净少答对 37 行。这个结果说明编码器能补到一些错题，但这组置信度条件尚不能可靠地辨认它们。本项已完成训练内研究，未升级为统一入口的默认策略。[分流实验方案](../../research/task1_feedback_extension_1127/cell_disagreement_gate.json) · [分流研究代码](../../scripts/study_cell_disagreement.py)

## 复现与交接

- 运行代码提交为 `7ee35ec`，四次处理的求解器源码指纹均为 `3d3e4d87dab94bd5cad717f4e0f1561a51bbe7ac7da7a620a3c5eefafadbdfc9`。整组留出 0 从匿名原包执行完整的 `solve`。另外三组从过去已审计的原版选模和重训结果继续，继承原有逐行预测、已用时间和拟合账本。四组均在新进程中重载模型并核对完整预测，所有外层预测封存后再评分。这四组来自已有研发面板，不能据此推断陌生任务的成绩。[实验驱动](../../scripts/study_task1_cell_encoders.py)
- 四次运行的累计拟合分别为 35、27、36、47，各次上限均为 64。按实际发生的顺序，本轮新增 35、8、15、18 次，共 **76 次**。之前的 69 次复用拟合只计入相应运行限额。该研究线累计计数由 1,418 增至 1,494。新增模型 API 调用 **0 次，¥0**。本轮未读取课程指定验证标签、正式测试标签或 H2 数据。[独立拟合计数与封存指纹](../../research/task1_feedback_extension_1127/cell_encoder_results.json)
- 配置 `SUPERAGENT_CELL_ENCODER_LIBRARY` 指向本机通用编码器目录后，使用下面的统一入口。兼容的本地 Sentence Transformers 模块可以直接加入目录。公共资源由运行环境提供，目录中不会写入任务数据。

```bash
python -m unified_agent solve training.zip --prediction-input queries.zip \
  --run-dir run --output predictions.csv --fit-limit 64 --time-limit 1800 \
  --feedback-reflection encoder-portfolio
```
