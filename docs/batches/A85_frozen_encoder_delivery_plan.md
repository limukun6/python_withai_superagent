# A85：复用A81冻结模型，真权重跨进程交付与损坏拒绝

A81四个worker已训练成功，正式路径在训练前因研究fixture清单错误而停，A84单独恢复。此批只复用A81的nonlinear单文本与vision两个真实冻结worker模型，0新fit/API；不是新增自主选模或性能验证。

按原payload的kind/标签/查询与原source指纹建立明确标注frozen_worker_probe的预测适配目录；不伪造为完整自主完成或full-refit。打包真实已训练pipeline、实际任务规格与来源，delivery_status=worker_probe；源码包v2记录外部权重/依赖。两个模型各在正式predict新进程与换cwd的python -I包执行，8条无标签CSV一致。然后缺权重、同大小一字节改动各启动新进程，必须在输出前拒绝；原权重不改，损坏副本只放新runs。

前提当前source与A81训练协议一致，否则停止。每进程180秒，预计1–3分钟，0fit/0API，H/test不读，不评估自然图像泛化。模型/权重/二进制不提交，保存来源哈希和只读核验结果。
