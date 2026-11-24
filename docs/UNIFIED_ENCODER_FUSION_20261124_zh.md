# Task2 三种新编码器和自主融合实测

BGE-small-en-v1.5、all-mpnet-base-v2、RoBERTa-base 均完成了真实训练和逐行预测。统一版交付继续使用上次课程指定验证答对 **689/940（73.30%）** 的双模型。新实验全部使用训练集内部划分。[原交付](UNIFIED_AGENT_20261124_zh.md) · [本轮凭证](../research/unified_encoder_fusion_1124/)

## 三个模型答得怎么样

相同训练行上，三个新模型分别学习自己的分类头。BGE、MPNet 分别编码两条标题并计算向量差。RoBERTa 把两条标题一起读，再交换先后读一次，提取每一侧受另一侧影响的词向量。三者用相同的词法、风格特征和逻辑回归配置。词表和统计量按折训练，预训练权重固定。第一次五折共 6,608 行，属于 4,874 个互不跨越评价折的共享标题组。[方法与预算登记](batches/unified_encoder_fusion_plan_20261124_zh.md)

| 第一次五折 | 答对 / 6,608 | 准确率 | 比原双模型改对 / 改错 | 净变化 |
|---|---:|---:|---:|---:|
| 原双模型 | 4,722 | 71.46% | 0 / 0 | 0 |
| BGE-small-en-v1.5 单用 | 4,694 | 71.04% | 353 / 381 | −28 |
| all-mpnet-base-v2 单用 | 4,733 | 71.63% | 480 / 469 | +11 |
| RoBERTa-base 单用 | 4,657 | 70.48% | 466 / 531 | −65 |
| 每折在学习区选一对搭档，始终等权 | 4,758 | 72.00% | 320 / 284 | +36 |
| 让六个成员学习一组固定权重 | 4,749 | 71.87% | 96 / 69 | +27 |
| 让六个成员按题学习权重 | **4,773** | **72.23%** | **156 / 105** | **+51** |
| **Agent 自主选方案后的实际交付** | **4,688** | **70.94%** | **240 / 274** | **−34** |

每个备选的分数由同一批题上的实际预测算得。逐题融合的五折净变化为 **+15、+11、+7、+9、+9**，合计提升 0.77 个百分点，按标题组重抽样的 95% 区间为 **+0.29 至 +1.28 个百分点**。单用 MPNet 的增益区间为 −0.80 至 +1.11 个百分点。[五折结果](../research/unified_encoder_fusion_1124/result.json)

## Agent 是怎样决定的

由人预先指定要检验的三种新编码器和四种融合／选模办法。Agent 获得原双模型、上轮的 MiniLM-L6 及这三种新模型的训练外概率，在学习区比较每个模型与每一对搭档，拟合固定和逐题两套融合权重，再使用独立确认区决定是否采用。预测时没有正确答案或模型专属的获胜条件。五折中，它两次选 MPNet 单模型，一次选 BGE 与 MPNet 等权，两次保留原双模型。三次改选分别比对应的双模型少答对 10、2、22 道题。五个实际交付物都经新进程从无标签原始标题导出，并与事前封存的决定逐行核对。[决策与全部逐行输出](../research/unified_encoder_fusion_1124/)

逐题融合会看各成员有多确定、与原双模型意见差多远，以及两条标题的词汇重合和长度差。在五折汇总中，原成员一致的 5,537 题里，三种新模型平均占 **24.64%** 权重；原成员分歧的 1,071 题里升至 **29.55%**。其中 MPNet 平均占比从 **12.97%** 增到 **16.69%**。这是单独评估逐题融合规则时的权重。最终采用哪套规则仍由前述确认步骤决定。[融合实现](../tools/preference_portfolio.py)

## 交换训练反馈后再试一次

第一次实验后，新规则改用概率误差的改善和按标题组估计的不确定程度，允许把多个有支持的候选方案再次合并。保留相同的六个基础分类器，交换每折的规则学习题和外层评价题。每道题归属的基础模型折发生变化，因此本轮应对照本轮双模型。两轮共用这 6,608 道训练内题，不能作为两套独立数据估计总体效果。[迭代登记](batches/unified_encoder_feedback_rotation_plan_20261124_zh.md)

| 同一批轮换后的外层题 | 答对 / 6,608 | 相对同轮双模型 |
|---|---:|---:|
| 同轮原双模型 | 4,754 | 0 |
| 按确认区净增选择单一方案的原规则 | 4,728 | −26 |
| **综合多项概率反馈后，Agent 实际采用** | **4,753** | **−1** |
| 单独评估固定融合 | 4,772 | +18 |
| 单独评估逐题融合 | **4,783** | **+29** |

逐题融合三折净增、两折回落，差值依次为 +16、+7、−4、−3、+13 题。按组重抽样的 95% 差值区间为 +0.02 至 +0.93 个百分点。使用综合反馈的 Agent 四折保留原模型，一折采用融合。其差值区间为 −0.24 至 +0.20 个百分点，两轮的实际自主交付均未达到预登记的晋级标准。[反馈轮换结果](../research/unified_encoder_fusion_1124/rotation_result.json)

## 从匿名训练包冷启动

另用只有训练数据的匿名包实跑可选入口。输入中没有任务名、目标成绩或指定赢家。Agent 自动留下 1,693 行作审核，另用 3,973 行拟合六个基础分类头、1,356 行学习规则、1,279 行确认。它最终选择六模型固定权重，在确认区比同次训练的原双模型多对 5 题。在之后打开的 1,693 道训练内留出题上，双模型 **1,214/1,693**，自主融合 **1,212/1,693**，改对 19 道、改错 21 道。模型所见的基础训练行数与历史交付不同。这组结果用于检查实际交付和相同训练量下的差异。[冷启动协议](batches/unified_encoder_entry_plan_20261124_zh.md) · [留出结果与权重](../research/unified_encoder_fusion_1124/cold/audit_result.json)

| 冷启动实际使用的成员 | 权重 |
|---|---:|
| MiniLM-L12 加训练语料熟悉度 | 48.19% |
| E5-small-v2 | 24.89% |
| MiniLM-L6 | 5.22% |
| BGE-small-en-v1.5 | 6.13% |
| all-mpnet-base-v2 | 9.59% |
| RoBERTa-base | 5.98% |

两个独立进程分别使用特征缓存和直接从标题编码，1,693 行预测逐字节一致。另对三份独立分类头逐个关闭缓存，各用 32 对标题验证类别和左右互换，概率与缓存的最大差小于 0.00000035。[冷启动导出指纹](../research/unified_encoder_fusion_1124/cold/audit_seal.json) · [逐模型核对](../research/unified_encoder_fusion_1124/feature_reload.json)

## 复算与使用

下面的复算只解包课程原包中的 `train.csv`，重建数据划分、按已保存概率核对每个决定，并重新统计逐题得失。无需下载权重或重新训练。

```bash
python -m scripts.verify_encoder_portfolio --archive /path/to/Released_Task2-Headlines_CTR.zip
```

实际运行可选入口时，使用项目的编码器环境，权重根目录下存放项目已经固定版本的 MiniLM-L6、MiniLM-L12、E5-small、BGE-small、MPNet 和 RoBERTa 子目录。以下下载工具支持三个新增候选，下载后逐文件校验发布版本。设置 `--endpoint https://hf-mirror.com` 可从镜像获取相同的校验字节。训练与预测均使用本地模型。逐题融合是运行时可选方案；传统分类任务继续由统一入口处理。[BGE](https://huggingface.co/BAAI/bge-small-en-v1.5) · [MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) · [RoBERTa](https://huggingface.co/FacebookAI/roberta-base)

```bash
python -m scripts.fetch_portfolio_encoders --weights WEIGHTS --model bge-small-en-v1.5
python -m scripts.fetch_portfolio_encoders --weights WEIGHTS --model all-mpnet-base-v2
python -m scripts.fetch_portfolio_encoders --weights WEIGHTS --model roberta-base
python -m scripts.unified_encoder_portfolio train TRAIN.zip --weights WEIGHTS --run-dir runs/new-portfolio
python -m scripts.unified_encoder_portfolio predict FEATURES.zip --weights WEIGHTS --run-dir runs/new-portfolio --output delivery/predictions.csv
```

| 新增用量 | 训练内五折 | 轮换复验 | 匿名冷启动 | 合计 |
|---|---:|---:|---:|---:|
| 实际下游分类器拟合 | 15 | 0 | 6 | **21** |
| 连续融合权重拟合 | 10 | 10 | 2 | **22** |
| 用标签筛选最佳单模型或双模型 | 10 | 10 | 2 | **22** |
| 根据反馈计算方案间的权重 | 0 | 5 | 0 | **5** |

每条路径连同复用的分类头最多计 11 个学习步骤。模型 API 调用 **0 次**，计费输入和输出 token 均为 **0**，模型 API 费用 **¥0**。权重文件和语料缓存留在本机，仓库保存代码、固定版本、概率、脱敏逐行预测与校验指纹。
