# H2 正式运行清单提案

供 A 在查看 EuroSAT 地理留出成绩前登记。依据共同版 `449b766` 和已登记的两个种子，B 建议每个种子各跑本地固定调度 L0 与真实 Flash 固定调度 R0。如果最终策略 F* 采用 L0，直接复用同一份模型和预测。两份训练包各有 2,500 张图片和相同的 9 个候选，含已固定权重的 MobileNet。组别、参数、输入及六个划分的指纹均在[机器可读清单](proposed_matrix.json)。

| 范围 | 路径 | 实际拟合上限（含重训） | Flash 请求上限 |
|---|---:|---:|---:|
| 单个种子 | 2 | 22 | 3 |
| 两个种子合计 | 4 | 44 | 6 |
| 本次入口验收实际用量 | 0 条完成训练 | 0 | 0 |

1. 两个 L0 均到达正式入口的训练边界。六份工作文件的字节、行序和标签指纹均与 A109、BH03 相符。R0 已通过真实参数解析。地理留出的特征和答案仍封存。[验收记录](results.json)
2. 正式用量统一写入已有累计作用域 `h2_transfer`。归档命令曾写成大写开头，已在清单中更正。每条 R0 使用现有的 0.30 元预计停止阈值。真实付费执行沿用账本持有方的独占账本和已接受历史摘要包装器，保留实测 token 用量；占位符仅说明运行参数。
3. A 可以把下方清单摘要登记为正式 H2，也可以在看成绩前另存一份写明变动的清单。所有计划模型封存后再统一评分。当前正式 H2 成绩待测。

`proposed_matrix.json` 的 SHA-256 是 `cf9db331c5b21fecbc52bdbf4d260c879d8a1d75a9e4e0a7f29416c6f473bc59`。[运行设计](overlay/docs/batches/B_h2_matrix_proposal_plan_20261124_zh.md) · [完整性记录](completion.json)

要复跑零训练验收，把本目录的 `overlay/` 内容复制到共同提交 `449b766` 的干净检出中。从检出根目录运行以下命令，传入含有 `45101/`、`45102/` 的真实登记训练目录和项目已固定的本地 MobileNet 权重。

```sh
python -m scripts.prepare_b_h2_matrix_proposal --inputs "$H2_TRAIN_INPUTS" --vision "$MOBILENET_WEIGHTS" --output runs/h2_proposal_check
```
