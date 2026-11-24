# A/B 统一 Agent 与 Released Tasks 复现

从仓库根目录运行同一个命令。训练输入可以直接使用课程原始 ZIP；使用一份新的运行目录。代码从训练文件与任务说明识别输入结构、评价指标及成对文本的语义。

```bash
python -m unified_agent train TASK.zip --run-dir runs/my-task
python -m unified_agent predict FEATURES.zip --run-dir runs/my-task --output delivery/predictions.csv
```

## 两份课程指定验证成绩

| 原始训练输入 | Agent 最终选择 | 实际拟合（含重训） | 指定验证准确率 |
|---|---|---:|---:|
| Task1 优惠券，全部 8,117 行 | CatBoost，深度 8、L2 为 3 | 9 | **1,532/2,030（75.47%）** |
| Task2 标题，8,301 对中自动留出 1,693 对，训练 6,608 对 | 12 层 MiniLM 加训练语料熟悉度特征，与 E5-small 等权平均正类概率 | 10 | **689/940（73.30%）** |

以各自原始任务包从空目录启动，2,970 行验证预测与历史最高记录全部一致。两份交付模型各经两个独立进程输出并比对。Task2 自动留出的训练内核对为 1,237/1,693（73.07%）。[两个原始包的逐行核对](../research/unified_raw_1124/results.json)

## 在这个基线上继续做的 Task2 实验

1. 将全部 8,301 对训练标题直接交给相同的内层选模器，6 次拟合后选中 12 层 MiniLM 加语料特征。指定验证为 **684/940（72.77%）**，比保留分组核对的双模型少 5 题。训练规模与选中的模型同时变化，这个对照无法单独估计多用训练数据的影响。[全量训练记录](../research/unified_release_1124/titles_full/model_frozen.json) · [指定验证](../research/unified_release_1124/full_scores.json)
2. 用两个模型在 1,314 对确认标题上的训练外预测学习一个融合权重，MiniLM 的权重为 50.71%。在隔离的 1,693 对训练标题上，等权答对 1,237 题，改权重答对 1,236 题，改对 1 题、改错 2 题，保留等权。[训练内权重对照](../research/unified_release_1124/calibration/result.json)

3. 在最终原始包模型上，让两个编码器意见不同时由置信百分位较高的一方决定。训练内留出中两者有 283 题意见不同。新策略改对 10 题、改错 10 题，仍是 **1,237/1,693**，按标题组重抽样的差值区间为 −0.49 至 +0.53 个百分点。旧模型的 456 道错题中，319 道两者同时答错，另外 137 道出现在两者意见不同时。当前交付继续等权。[分歧实验与错误分布](../research/unified_raw_1124/confidence/result.json)

4. 让研究程序只看相同训练内划分的反馈，在剩余的三个已训练候选中自动挑第三个模型。它选择了 6 层 MiniLM 加词汇特征，训练内首轮比较净增 3 题，另一确认划分净增 7 题。在 1,693 对训练留出上，这个模型单独能答对双模型共同错的 319 题中的 40 题。直接加入等权投票实际救回其中 6 题，整体改对 33 题、改错 32 题，从 1,237 到 1,238。当三个成员都只用相同的首轮训练行拟合时，另一个对照从 1,234 到 1,240。两组配对区间均跨零，维持原双模型。这一轮复用已训练权重，未增加拟合或指定验证评分。[自动选出第三模型的依据](../research/unified_raw_1124/third/selection.json) · [共同错题与实际投票结果](../research/unified_raw_1124/third/result.json)

本轮共 45 次实际拟合。历史面板复现 19 次，全部标题对照 6 次，独立学习融合权重 1 次，最终从原始包验收 19 次，置信百分位和第三模型实验均为 0 次。模型 API 0 次，输入输出 token 均为 0，新增模型费用 ¥0。

## 运行环境

本次使用 Python 3.11.15、scikit-learn 1.7.2、CatBoost 1.2.10。依赖版本在 requirements-ab-encoder-tested.txt 及其引用的 requirements-tested.txt 中。模型目录中各有经哈希核对的 config.json、tokenizer.json、model.safetensors。训练与预测均可离线执行。

```bash
python -m pip install -r requirements-ab-encoder-tested.txt
export SUPERAGENT_MINILM_WEIGHTS="/path/to/pretrained_weights/all-MiniLM-L6-v2"
export SUPERAGENT_DEEP_MINILM_WEIGHTS="/path/to/pretrained_weights/all-MiniLM-L12-v2"
export SUPERAGENT_E5_SMALL_WEIGHTS="/path/to/pretrained_weights/e5-small-v2"
```

## A 和 B 怎样接到一起

共同执行器、动作校验、反馈记录、重训和预测导出来自队友分支的 8e5e98a4。B 的运行时方法发现和调度逻辑已进入相同的 superagent/。开放搜索可以从当前环境发现兼容的分类器。对于声明左右偏好的二分类，执行器可以对两个候选的训练内逐行预测做同划分比较。对于原生类别表，先以相同正则强度比较浅树与深树，再依训练反馈扩大搜索。

本版使用本地规划。人规定可执行能力、分组、探索规则和上限；运行开始后，Agent 根据训练反馈选择下一步、何时停止、重训哪个模型，以及是否保留两个模型。训练和预测均未传入任务编号、目标验证分数或指定赢家。安装了哪些模型、任务的输入结构与已实现的适配器会影响可选方法。

原始包入口只打开训练成员及其说明。对存在重复文本、且分组后类别样本足够的偏好任务，它根据共享文本形成连通分量，按既有五折规则保留一折。同一标题不能跨到留出侧。该折只供模型定稿之后核对，不参与内层选模。其他支持的分类任务将全部训练行交给同一个执行器。分组种子 41101 来自此前已登记的训练内实验。原始包入口不读取历史行号清单。

每条原生路径共用 10 次实际拟合上限，包含最终重训。课程指定验证在模型和预测冻结后评分。两个已发布任务的指定验证曾用于历史研发，结果属于开发验证。未见任务的效果需要独立测量。

## 复算

在仓库根目录运行以下命令，可从课程原包逐行复算指定验证成绩，并核对每次真实拟合、源码和预测指纹。还可在已有依赖和固定权重的机器上用上面的统一命令从头训练。

```bash
python -m scripts.verify_unified_release --packages /path/to/course-zips
python -m scripts.review_unified_delivery --packages /path/to/course-zips
```

原始包验收使用的源代码、动作、预算、预测和对照数据以去除机器路径的形式公开。原始课程数据与分类器权重按组内约定单独保存。[原包验收记录](../research/unified_raw_1124/) · [历史对照、校准与全量训练](../research/unified_release_1124/) · [最初实验协议](batches/unified_release_protocol_20261124_zh.md) · [原始包协议](batches/unified_raw_protocol_20261124_zh.md) · [分歧实验协议](batches/unified_confidence_protocol_20261124_zh.md)
