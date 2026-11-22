# A61：AUC目标接入前的零拟合审计

本批不训练、不访问外层验证或H/test、不调用API。运行前保存本计划和脚本。它回答“现有真实模型能否产生正确列序的概率，以及标签相同为何不能证明AUC相同”，不回答“AUC自主搜索是否有效”。AUC正式AB入口仍拒绝运行，不提前删除保护。

使用A47五个已见任务的第一份primary模型及其原训练内search分区，每任务一次概率推理。核验模型/数据SHA、主指标身份、原primary硬预测可重现。计算一个明确的诊断ROC-AUC，反转概率列并同步反转类名后结果必须相同；只换类名而不换数值会有不同含义，不当作等价操作。汇总仅保存统计/指纹，不发布数据或概率矩阵。

人工反例：相同全部硬标签，概率排名不同，AUC不同；二分类正类顺序；多分类列序重排；NaN/缺列/非法概率/缺类必须拒绝。人工分数与真实模型诊断分开统计。既有模型按原目标选出，不能把诊断AUC当成AUC目标的最终成绩。

运行：
```powershell
& 'D:\codex\python_withai\.venv\Scripts\python.exe' -m scripts.run_a61_auc_readiness --reference 'D:\codex\python_withai\runs\a47_catalog_Q' --output 'D:\codex\python_withai\runs\a61_auc_readiness'
```

预期5个概率推理、约5–30秒，实际fit/API均0；已加载类的fit和网络显式禁用。失败保留，不重跑训练。后续接入必须补齐score矩阵与class_order的行序/标签/payload/配置指纹、scheduler重算、worker/状态/交付/partial/接口导出，并用AUC目标重新自主训练；A61不代替它。
