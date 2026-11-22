# A38：真实模型搬迁后的零训练交付核验

三小时窗口内独立工程验收。只在A32/A34全部模型冻结并completed后运行，不并发新训练。四任务预先固定取routes模式：Coupon/偏好对R0_routes，SMS/照片R1_routes，不按分数挑模型。

将每个已冻结模型的task_spec.json、final_selection.json及指定model文件复制到新目录；原文件不移动不改。用新进程正式predict入口，在无网络、禁止任何受支持分类器fit的保护下，对原同一无标签features.zip预测。与原CSV逐字节一致即通过，另记模型SHA/行数；不打开holdout标签，不重新评分，不将搬迁当新性能样本。最多4次推理、0监督fit、0API，预计10–30秒。

scripts/run_a38_portability.py与scripts/a38_predict_guard.py。若某模型依赖原目录、隐式网络或隐式重训，停止并保留错误，不自动修复重训；工程问题与算法分数分开。全部原数据/模型/查询包只在忽略runs目录，Git只放代码与脱敏验收记录。
