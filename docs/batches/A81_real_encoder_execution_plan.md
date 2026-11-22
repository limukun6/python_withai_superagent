# A81：真实冻结编码器执行与交付（资产ready后才启动）

A累计起点1239，A79须ready且四文件完整SHA匹配；使用其独立Python3.12覆盖环境，所有模型执行禁网，不调用学校API，不读外层val/H/test。A80外部资产交付契约3检查和旧词法包兼容已过，不能代替本批真权重。

仅从A47登记的SMS训练包以seed40811固定分层抽256行，合成64对明确preference输入、32张uint8合成RGB图像。依次实际worker调用MiniLM plain、MiniLM nonlinear、nonlinear preference、MobileNetV3-small线性头，各最多1fit；验证真实384/576维特征、监督fit计数、概率/交换性质。不调参，不以合成图像或训练内分数报告自然图像/未见迁移。

再将256行SMS匿名任务从统一CLI运行local ab-v1/full、seed707、fit-limit5（含refit）、trial180秒、path600秒，默认本地策略，无人工选择下一方法。目录需实际含两条MiniLM路线，是否被选及是否成为赢家分别记录；不因为未选就改策略。记录source/菜单/依赖与动作。随后正式新进程和独立源码包对31条训练来源无标签输入预测，CSV逐字节一致。若赢家为编码器，源码包须列完整外部资源及实际torch等版本；另测试缺资产拒绝（0额外fit）。如赢家是词法，编码器交付只标worker重载验证，不能说CLI赢家用了它。

四个指定动作是执行/性质探针，不是自主效果实验；第五条才是自主入口验收。全批最多9fit/0API，预计3–10分钟，硬总上界约25分钟；首失败停、不自动重试。保存protocol/results/completion、真实fit日志、原始异常。模型/权重/抽样数据只放runs，Git仅脱敏摘要；未冻结Q/C1/H。
