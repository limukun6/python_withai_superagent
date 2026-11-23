# B 线研发交付：数值表格与标题偏好

在 Dry Bean 的 A94 开发留出上，动态 Agent 用 7 次实际拟合选中新的数值核模型，答对 **2,005/2,178（92.06%）**。原动态路径答对 1,989 行（91.32%）。开放同一候选后，固定调度用了 11 次拟合，交付的预测与动态调度逐行一致。另一个二分类数据上，Agent 试过新方法后选回原逻辑回归。A 当前接入的团队快照继续按此前交接执行。[团队交接](../../docs/batches/B_TO_A_LATEST_INTEGRATION_REPLY_20261123_zh.md)

## 数值表格

新候选在原本标准化的数值之外增加 512 个随机傅里叶特征，使同一个逻辑回归能学到弯曲的类别边界。带宽按当前训练折的样本距离确定。原有线性与树模型继续参加竞争。由研发者设计候选及资源限制，运行中的选型、停止和最终重训由 Agent 完成。

| 比较 | 原方法 | 新方法 | 如何取得 |
| --- | ---: | ---: | --- |
| A94，新核对原逻辑回归 | 1,989/2,178 | **2,005/2,178** | 动态与固定入口均自主选型、重训并在新进程导出 |
| A94，新核对早期仅用核特征的模型 | 1,991/2,178 | **2,005/2,178** | 在同一份 8,710 行训练池上补训早期模型 |
| BN01，新核对早期核模型 | 2,008/2,174 | **2,017/2,174** | 在同一份 8,714 行训练池上补训新模型 |
| Wisconsin 二分类开发任务，默认目录对扩展目录 | 111/114 | 111/114 | 两套目录均自主选择较强正则的线性模型，AUC 均为 0.98755 |

在 A94 上，换掉逻辑回归后改对 34 行、改错 18 行，净增 0.73 个百分点。按完整特征组做配对重抽样，95% 区间为 +0.09 到 +1.38 个百分点。动态路径做了 6 次探索和 1 次重训。固定路径做了 4 次探索、6 次复核和 1 次重训。后者验证了更多内部分区，两者只贡献一份外层预测效果。两个历史对照分别复核了全部 10 和 11 个调度动作。[动态结果](evidence/adaptive/results.json) · [固定结果](evidence/fixed/results.json) · [调度预登记](protocols/B_numeric_schedule_comparison_1123_zh.md)

两代核方法的配对比较是在知晓各自历史成绩后登记的。A94 净多 14 行，BN01 净多 9 行。两份开发集共享 402 个完全相同的特征组；后者的配对 95% 区间为 −0.18 到 +1.01 个百分点。这两次结果都保留了原始训练规模。BN01 的 2,017 行来自研发补训；A94 的 2,005 行另有正式入口自主运行证据。[两代核方法逐套配对](evidence/generations/results.json)

此前的内部筛选中，11 近邻初筛多对 4 行，换组后少对 12 行和 9 行。新核初筛多对 14/1,737，换到此前三份筛选未评分的组又多对 7/855 和 5/964，达到预定门槛。Wisconsin 以 ROC AUC 作为实际选型目标，引入核候选后从 7 次拟合增至 8 次，两个最终模型及 114 行概率输出相同。[近邻和二次判别](evidence/geometry/results.json) · [核的训练内筛选](evidence/kernel/results.json) · [AUC 对照](evidence/transfer/results.json)

## Task2

在 5,017 行实测比较上，用“甲胜乙、乙胜丙”生成 411 对间接比较，各给实测比较的一半权重。只用训练组生成弱标签；两个评估组均按实际观测的胜负打分。仍用 E5 与词频 MiniLM-L12 投票，只重新训练 E5 分类头。A 组从 578 升到 581/814，B 组维持 563/777。合计由 1,141 升到 1,144/1,591，低于净增 8 对的门槛。训练图中仅有 1 条直接观测可独立核对一条两步关系。保留原 Task2 主线；课程指定验证的已测成绩为 **689/940（73.30%）**，本批只打开训练内开发行。[规则](protocols/B_transitive_preferences_1123_zh.md) · [本批配对结果](evidence/transitive/results.json) · [689/940 的原运行](../alternative-encoders-2026-11-23/docs/batches/B_joint_coverage_results_1123_zh.md)

## 复算和使用

在本目录执行以下命令，可用 NumPy 从匿名分组记录重算配对正确数与区间，复核拟合次数和跨进程导出哈希。

```bash
python verify.py
```

代码包内有实际 B 版本的完整生产 Python 源码、研究运行器、依赖清单和各批计划。在已安装依赖的环境中，从解压目录运行统一入口。数值核可用于声明为数值表格且内存估算可容纳额外特征的分类任务；移除开关可运行旧目录。[代码包](runtime-source.zip)

```bash
unzip runtime-source.zip -d numeric-runtime
cd numeric-runtime
python -m superagent run "$training_packet" --portfolio ab-v1 --scheduler ab-v1 --planner local --numeric-kernel --seed 707 --fit-limit 11 --trial-timeout 300 --time-limit 1200 --run-dir runs/new_numeric
python -m superagent predict "$features_packet" --run-dir runs/new_numeric --output runs/new_numeric_export/rows.csv
```

从 A 的 `c8a21d6` 接这项可选能力时，可在单独工作树应用 A 补丁。A 副本已实际重载 B 的冻结模型并导出 2,178 行，逐行与 B 相同；A 侧本次验收复用了冻结权重。B 补丁适用于 `c376578`。包内 `historical/task2_source_delta.patch` 可在副本中还原本次标题研究的生产文件版本。[A 补丁](integration/A_c8a21d6_numeric_kernel.patch) · [B 补丁](integration/B_numeric_kernel.patch) · [独立导出与版本指纹](evidence/integration.json)

恢复研发后共 **39 次本地监督拟合、0 次模型 API 请求、¥0 新增 API 费用**。其中 Bean 内部核筛选 3 次、正式动态与固定路径 7＋11 次、两代核方法补训 2 次、Wisconsin 两条路径 7＋8 次、Task2 弱标签 1 次。暂停前已完成的近邻和二次判别另有 4 次。A94 评分首次核对时混淆了原数据行号和导出文件的位置，程序在计算正确数之前中止。修正时复用已封存的模型和文件，核对 A94 文件指纹、每行特征、原始行号和导出位置。课程正式测试和 EuroSAT H2 本批均未评分。
