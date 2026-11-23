# 共同入口的 Released Task2 成绩和 H2 训练输入验收

共同入口在 8,301 对训练标题上从头选方法并完成重训。在模型封存后对课程指定的 940 对标题做预测，**答对 684 对（72.77%）**。旧版成绩 **689/940（73.30%）** 可继续按原出处引用。两次均把模型选型限定在训练数据内，使用同一份课程验证，正式测试留待最终评估。[本轮结果](evidence/results.json) · [旧版完整自主运行](../alternative-encoders-2026-11-23/docs/batches/B_joint_coverage_results_1123_zh.md)

| 运行 | 最终用于拟合的训练对数 | Agent 实际选择 | 指定验证 |
|---|---:|---|---:|
| B 历史运行 | 6,608 | 带语料词频的 MiniLM-L12 与 E5-small 等权投票 | 689/940 |
| 共同入口，生产提交 `87634a5`，核验快照 `a36296f` | 8,301 | 带语料词频的 MiniLM-L12 单模型 | 684/940 |

逐题比较有 38 题由错变对、43 题由对变错，相差 −0.53 个百分点。按 904 个共享标题组做 5,000 次配对重抽样，差值的 95% 区间为 −2.45 至 +1.37 个百分点。训练量、内部评估划分和选出的模型均有变化，单靠这两次成绩无法归因。核对归一化标题后，训练与课程验证之间共享标题的验证行数为 0。[分组计数](evidence/grouped_correctness.json) · [两份预测与导出摘要](evidence/exports.json)

## Agent 为什么停在单模型

可选方法和预算在启动前确定。Agent 先后试了词袋逻辑回归、MiniLM-L6、MiniLM-L12、带语料词频的 MiniLM-L12 和 E5-small。在 1,693 行训练内评估上，最佳单模型答对 1,220 题。它检查了十种等权组合，最高的 1,223 题来自该单模型和词袋模型。进入复核的原规则要求提高至少 0.2 个百分点，这次需要多答对 4 题。E5 与最佳单模型的组合答对 1,212 题。Agent 随后停止搜索，用全训练集重训单模型。共 5 次探索和 1 次重训，模型 API 调用、输入和输出 token、新增模型 API 费用均为 0。[完整动作与模型封存](evidence/models_frozen.json) · [各组合当时的反馈](evidence/selection.json) · [逐进程拟合记录](evidence/physical.jsonl)

随后单独验证了把临界组合再给一次机会的价值。从训练池排除首轮评估出现过的所有标题组，另外 4,971 对用于拟合，新留的两组也互不共享标题。采用同一对冻结模型，在 791 题上单模型与组合分别答对 567、564，在 846 题上分别答对 603、604。组合合计 **1,168/1,637，单模型 1,170/1,637**。两个复核区共享一对训练模型，均属开发证据。按预先写下的条件，本次保留原门槛。追加拟合 2 次、API 0 次，本项只使用课程训练包。第一次工作进程缺少共同目录配置，在拟合前退出。修复后逐项核对两个候选的实际参数才启动。[复核计划](overlay/docs/batches/B_near_threshold_plan_20261123_zh.md) · [复核结果](borderline/results.json) · [首次启动回执](borderline/preflight_stopped.json)

## 交付与 H2

正式 Task2 运行使用 91 个逐文件校验通过的生产源码文件。两个原生预测进程、一个隔离预测进程和一个从 ZIP 重新解压的隔离进程，在 940 行上产生字节完全一致的结果。独立包约 2.56 MB，另需清单所列的公开 MiniLM-L12 固定权重。Git 提供模型指纹、公开权重摘要和预测文件，模型 ZIP 留作私下交接。[预测包验收](evidence/portable_receipt.json) · [指定验证预测 CSV](evidence/validation_predictions.csv)

在同一版生产源码上，B 已经分别用正式读取器载入 EuroSAT 45101、45102 的训练包。每包 2,500 张、10 类、9 个候选，每包三个 2,000/500 内部划分，行序、标签及组边界都与 A 的登记一致。本机用登记的 MobileNetV3-Small 权重对一张训练图像完成离线前向，得到 576 维。H2 正式训练和外层评分尚待共同运行清单；这一步共 0 次拟合、0 次模型 API。[H2 训练验收记录](evidence/h2_training_readiness.json)

A 在 `3eebf55` 已发布 H1 完整结果，该提交和本轮的 91 个生产源码文件相同。H1 的正式运行表指定的是 61411 和 61421。H2 如需 Flash，按 A 最新交接由 A 在现行累计账本执行，或先把最新原账本和独占写入窗口交到 B。其最新回执为 363 条请求、12 条历史未知、0 条待处理。这里的两项 Task2 实验和 H2 训练验收都是本机离线计算。[A 的当前交接记录](https://github.com/limukun6/python_withai_superagent/blob/3eebf5519509bd407c7d5dcddf9526c14e0f47a3/docs/batches/A130_A132_H1_paid_results_20261123_zh.md)

## 如何复算

在本目录用 Python 3.10 或更高版本运行以下命令。第一条只需标准库。第二条用 NumPy 重算配对区间，并用课程原包的验证标签独立重评两份 CSV。

```sh
python verify.py
python verify.py --interval --course-archive /path/to/Released_Task2-Headlines_CTR.zip
```

从头复现实验时，单独检出 `a36296f550c85e49376caedd0149d21caf94dd63`，将本目录 `overlay/` 的内容按原目录复制到该工作树，安装此次运行依赖并准备 L6、L12、E5 三套登记过指纹的公开权重。下列第一条可用课程原包和仓库内的旧预测重建历史参照。它只读取 `val.csv` 生成无标签特征并存入参照目录。训练 CLI 只收到匿名训练包，评分程序在模型封存后读取参照文件。后两条从共同工作树根目录执行。[原始运行计划](overlay/docs/batches/B_shared_full_task2_plan_20261123_zh.md) · [生产源码快照](https://github.com/limukun6/python_withai_superagent/tree/a36296f550c85e49376caedd0149d21caf94dd63)

```sh
python /path/to/this-release/prepare_reference.py --course-archive /path/to/Released_Task2-Headlines_CTR.zip --out /path/to/reference
python -m scripts.run_b_shared_full_task2 --archive /path/to/Released_Task2-Headlines_CTR.zip --weights /path/to/pretrained_weights --previous /path/to/reference --out runs/shared_full_task2_replay
python -m scripts.study_b_borderline_stop --original runs/shared_full_task2_replay --weights /path/to/pretrained_weights --out runs/borderline_stop_replay
```
