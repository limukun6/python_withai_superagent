# Task2 已验证方法与队友接入说明

标准 SuperAgent 自主选用十二层 MiniLM 加词法特征，在训练集内部按共享标题分组预留的 1,693 对标题上答对 1,224 对，准确率 **72.30%**。两组分别从同一份匿名训练包的 6,608 行开始，使用相同的本地动态调度、种子 8123 和预算。模型封存后，每组都通过正式预测入口在两个独立进程导出结果，四份输出逐行相同。[配对成绩](../../research/b_accuracy_frontier_1123/pair_results.json) · [导出校验](../../research/b_accuracy_frontier_1123/pair_exports_frozen.json)

| 可用方法目录 | 最终自主选择 | 训练内保留区 | 实际监督拟合，含重训 | 模型 API |
|---|---|---:|---:|---:|
| 原目录加十二层，共 33 项 | 十二层 MiniLM 加词法与风格特征 | 1,224/1,693 | 10 | 0 |
| 再开放可学习的改写特征，共 34 项 | 同一十二层单模型 | 1,224/1,693 | 10 | 0 |

新候选额外学习标题改写片段的 384 个特征权重。调度器在第 4 次动作实际训练了它。初筛中改写模型比普通十二层少答对 2 对（919 对 921），第一次复核多答对 5 对（993 对 988），第二次复核打平（942 对 942）。依据事先约定的复核晋级规则，Agent 保留了普通十二层。验证支持将十二层加入共同目录；改写头可选，本次未带来额外的保留区提升。[逐次决策与分区证据](../../research/b_accuracy_frontier_1123/paired_feedback.json) · [改写特征消融](../../research/b_accuracy_frontier_1123/edit_ablation.json)

两组共使用 20/22 次实际拟合，模型 API 输入与输出均为 0 个 token，新增费用 ¥0。对照组完成后，验收脚本一度将账本的零用量初始化事件误判为调用。修复只涉及回执校验，保留该组已拟合模型并继续研究组。正式测试、H1/H2 均未参与，本轮没有启用课程指定验证评分。已有的 685/940（72.87%）属于旧研究适配器在课程指定验证集上的成绩。[预算与实际分类器调用](../../research/b_accuracy_frontier_1123/pair_usage.json) · [验收修订](../../research/b_accuracy_frontier_1123/pair_receipt_amendment.json)

## 从仓库运行

在仓库根目录或个人分支的解压包根目录，准备 Python 环境及已核验的本地六层、十二层 MiniLM 权重。将 `SUPERAGENT_MINILM_WEIGHTS`、`SUPERAGENT_DEEP_MINILM_WEIGHTS` 分别指向两个权重目录。库中包含资产的文件名、大小与 SHA-256。两个新增候选在资产及任务契约符合时自动进入共同目录。[编码器依赖](../../requirements-ab-encoder-tested.txt) · [实现和权重校验值](../../superagent/ab_sentence.py)

用课程发放的 Task2 原始 ZIP 在本机重建同样的分组。第一条命令仅读取课程包内的 `train.csv`，并校验匿名训练包、无标签特征、评分标签和分组四份产物的指纹。正式 `run` 的输入只使用匿名训练包。

```sh
python scripts/rebuild_b_preference_inputs.py --archive /path/to/Released_Task2-Headlines_CTR.zip --output runs/title_inputs
python -m superagent run runs/title_inputs/anonymous.zip --scheduler ab-v1 --portfolio ab-v1 --planner local --preference-catalog deep --fit-limit 11 --seed 8123 --time-limit 1200 --trial-timeout 300 --run-dir runs/title_deep
python -m superagent predict runs/title_inputs/features.zip --run-dir runs/title_deep --output runs/title_deep_export/rows.csv
```

用 `--preference-catalog deep-edit` 及不同的运行目录即可让同一调度器同时考虑改写候选。原目录可用 `--preference-catalog standard`。接入依据任务契约识别有方向的二分类文本对。候选、预算和选择规则在启动前登记，实际试什么、是否复核和最后交付哪个模型由 Agent 决定。[配对协议](B_shared_preference_pair_protocol_1123_zh.md) · [入口与候选交付验收](B_shared_preference_delivery_1123_zh.md)
