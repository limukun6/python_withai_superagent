access.jsonl 只记录训练包与验证包允许的成员。validation/ 中保留实验原始选型元数据（路径已脱敏）。solutions/ 中的 adaptive 选型说明补齐已有选择的 selected_experiment 名称，使标准预测入口可以导出清单；不更改模型字节、训练、预测、分数或策略。模型与逐行结果摘要见 scores.json。
