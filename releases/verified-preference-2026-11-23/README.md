# 可运行的 Task2 偏好分类方法

本目录包含与已验收整合版本一致的 87 个 SuperAgent 运行模块、六层与十二层编码器的依赖清单、匿名训练输入重建脚本及配对实验记录。进入本目录即可按报告中的命令运行。代码在训练前校验本地编码器权重；运行数据包和编码器权重由使用者提供。[接入方式、权重路径与完整命令](docs/batches/B_verified_preference_1123_zh.md) · [固定源码版本](https://github.com/limukun6/python_withai_superagent/tree/c6584776be99a1eaee59b881047566219651edbd)

| 标准入口选项 | 训练内保留区正确数 | 实际拟合 | 模型 API |
|---|---:|---:|---:|
| `--preference-catalog deep` | 1,224 / 1,693 | 10 | 0 |
| `--preference-catalog deep-edit` | 1,224 / 1,693 | 10 | 0 |

两组最终都选普通十二层，保留区准确率均为 **72.30%**。课程指定验证集上的历史 72.87% 属于旧适配器，使用另一份 940 行数据。[逐步选型](research/b_accuracy_frontier_1123/paired_feedback.json) · [两组实际用量](research/b_accuracy_frontier_1123/pair_usage.json) · [文件指纹清单](SOURCE.json)

发布时从本目录独立运行了 8 项测试，并用课程原包重建了训练输入与三份内部划分。再用原已训练模型从 `superagent predict` 导出 1,693 行结果，与配对实验封存的 CSV 逐字节一致。[预测文件 SHA-256](research/b_accuracy_frontier_1123/pair_exports_frozen.json)
