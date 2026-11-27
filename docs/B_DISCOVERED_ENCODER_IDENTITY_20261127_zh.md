# 新编码器当选后，案例投票如何用上同一套向量

Agent 能从资源目录发现适用的开源句子编码器，用它训练分类器，也能根据训练反馈给它派生一个相似案例投票模型。原实现给新分类器加载了正确的权重，但派生时继承了内置 MiniLM 的模型标识，导致投票模型可能用上另一套向量。

现在从已训练分类器保存的编码器标识和文件指纹生成投票模型。在三份以前真正训练过的分类器上逐一对照，相同文本的原向量和派生向量匹配数由 0/3 变为 3/3。新增的 Arctic 资源也通过了向量对照。随后经正式工作进程训练一个派生模型，使用 623 条训练、145 条内部验证。存下的向量和随包导出的资源都与原分类器一致。这一步有 1 次本地拟合，模型 API 0 次。[旧版证据](../research/paired_encoder_search_1127/local_adviser_before_results.json) · [修正版证据](../research/paired_encoder_search_1127/local_adviser_after_results.json)

此前 710/940、702/940 及电影台词 354/450 的相似案例投票均来自内置 GTE-base，701/940 的模型只使用两个编码器。新代码在原通用资源目录下从课程原包冷启动，允许 22 次拟合、1,800 秒。Agent 实际用 20 次拟合，自选 BGE-small、GTE-base 和基于 GTE-base 的相似案例投票，三者权重分别是 37.5%、37.5%、25%。无方法或路径提示。新进程重载产生的 940 行预测与已评分的旧交付逐字节相同，因此可确认同样的 **710/940（75.53%）**。本次仅打开课程原包中的 README 和训练文件；另给无标签的预测输入。没有重读指定验证答案。这个数针对原资源目录，额外安装资源可以改变自主选择。[复跑协议](../research/paired_encoder_search_1127/corrected_delivery_protocol.json) · [真实动作与训练反馈](../research/paired_encoder_search_1127/corrected_delivery_trace.json) · [逐字节比较](../research/paired_encoder_search_1127/corrected_delivery_comparison.json)

同一工作进程对同一批不可变文本先后调用预测和概率输出时，现在也会复用 GTE 的编码结果。在 96 条及 192 条的两组对调顺序测量中，实际编码均从 3 批减为 2 批，经过时间分别减少约 20% 和 22%。预测和概率逐字节相同。这些比例只描述所测的工作进程序列。[时间与等价性记录](../research/paired_encoder_search_1127/frozen_reuse_results.json)
