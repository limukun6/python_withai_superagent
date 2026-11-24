# 把已冻结的两个模型原样送到另一部电影评论语料

锁定 B-TEXT-KERNEL-05 电影来源留出时由各自 Agent 交付的两份模型：原模型 46a85166d1a25f0abad0bc36caaed9a67d371ea5ea3a6e31863043c0d3c6d428，非线性模型 8ae5466eb0fbefa09320a74e13453eda3fbe10cc6d6362342bec5c41a05ce464。两者当时只用商品和餐馆评论训练。此前 UCI 电影样本上的改进促成这次复查。新语料固定为 Pang 与 Lee 的 Cornell sentence polarity v1.0。作者按 Rotten Tomatoes 正／负评价标识整理 5,331 条正例、5,331 条负例。用全体原始片段，按 LF 分行；UTF-8 严格解码失败时，以 Latin-1 逐字节保留原文本。排除与此前 UCI 全部 3,000 条已见文本标准化后完全相同的行，保留其余所有行与重复组；登记排除数。把样本按文本 SHA-256 排序，同值用发布文件位置打破并列，预测包只有文本，没有来源或标签。

两份冻结模型分别由各自当时的代码版本在离线环境和新进程预测。两份输出都落盘并绑定模型与行序后才与标签比较。主指标 macro-F1，同时报告准确率、改对改错、正负类召回；按标准化文本组作 4,000 次成对重采样，种子 112204。该间隔只表述本语料的样本不确定性，不假定同电影或影评人片段独立。此项是冻结模型跨语料推理能力检验，没有在 Cornell 上运行新的自主优化流程。不开课程官方验证／测试／H，不读密钥，新增监督训练、无监督训练与模型 API 均为零。[作者发布页](https://www.cs.cornell.edu/people/pabo/movie-review-data/) · [原始标记说明](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.README.1.0.txt)
