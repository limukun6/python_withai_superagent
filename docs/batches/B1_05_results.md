# B1-05 · B 的真实标题小试与对称文本交付

正式生产版本为 `b79a0ab`，基于 A 的 `5ed06c6`。两条路径运行同一个本地动态入口。开发实验共 14 次实际分类器训练，逐次与独立观察器一致，新增 API 请求 0 次，全批约 89 秒。两个模型全部封存后才读取外层开发标签。

| 数据 | 冻结模型的外层准确率 | 探索 / 复核 / 重训 | Agent 选中的方法 |
|---|---:|---:|---|
| Released Task2 原训练集按共享标题分组留出 | 1,195 / 1,672，71.47% | 9 / 0 / 1 | 冻结 MiniLM 加词法特征 |
| 匿名对称功能测试 | 48 / 48 | 3 / 0 / 1 | 字符 n-gram 的绝对差和逐元素积 |

Task2 初始可见 23 个合规候选。从共同的词法基线起步后，调度器在第 2 步选了本地 MiniLM，随后还尝试了其他 7 个配置。最终选中的模型通过新进程预测和目录搬迁核验。未测课程指定验证集和测试集。这里的 71.47% 只对应这 1,672 对开发数据。

新方法通过词或字符特征的绝对差与逐元素积，使两段文本对调后每一列特征相同。对称任务仍使用原标签。对称样本的 48/48 只用于功能验收。新进程导出模型在这 48 行上的正反序标签全相同，每一类的概率最大差值为 0。方向性关系仍进入原先的有序文本列流程。

H1 登记为 Google Research 的 PAWS-Wiki Labeled Final。官方训练分片文件 SHA-256 已核对，为 `8dc9ad3e…277cf`。49,401 对人类标注的输入按共享句子分组后封存两个外层及各自三个训练内划分。本轮在 H1 上训练 0 次、评分 0 次。[数据说明](https://github.com/google-research-datasets/paws) · [固定版本](https://huggingface.co/datasets/google-research-datasets/paws/tree/161ece9501cf0a11f3e48bd356eaa82de46d6a09/labeled_final)

队友可以用 `protocol.json` 中的数据、代码、模型以及环境摘要核对复现条件。位置索引以 `row_ids.json.gz` 交接，外层相对于原始训练行，内层相对于交给 Agent 的训练输入。H1 的索引使用数据集公开的源 ID。B 本机 scikit-learn 为 1.7.2，A-10 记录为 1.9.1。跨机成绩须先核对文件、划分、候选与依赖。

[本批协议](B1_05_plan.md) · [正式路径逐步结果](../../research/b1_05_pair_pilot/results.json) · [预算与交付核验](../../research/b1_05_pair_pilot/completion.json) · [H1 封存记录](../../research/b1_05_pair_pilot/h1/results.json)
