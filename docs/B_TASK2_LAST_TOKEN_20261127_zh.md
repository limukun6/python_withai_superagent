# 千问本地嵌入模型的自主对照

在同一训练集内部，按标题关联隔离抽出 768 个标题对供 Agent 自己做实验，另留 256 对检验。两边均可用 28 次拟合、1,800 秒，只在其中一个资源目录里额外放入通用 Qwen3-Embedding-0.6B。冻结模型在本机提取特征，通用分类头在训练数据上学习。Agent 从普通入口运行，并自行决定是否使用它。

| 条件 | 最初候选数 | 实际拟合（含重训） | 最终自选模型 | 另留 256 对 |
| --- | ---: | ---: | --- | ---: |
| 原资源目录 | 71 | 26 | ELECTRA 二次特征头 | 159 / 256 |
| 增加千问嵌入 | 72 | 26 | ELECTRA 二次特征头 | 159 / 256 |

Agent 实际训练了千问候选，首份内部验证答对 94/145，与该轮的 MPNet 和 DistilRoBERTa 持平。两条路线独立重载后，留出的 256 个预测全部一致。按事前条件，子样本组必须优于对照，才开展这一配置的全量追加试验。这些结果只反映此子样本与分类头。

本机独立计时使用同一批 12 个输入，CPU 3.28 秒、Apple GPU 0.30 秒，逐向量最小余弦相似度 0.9998。Agent 因此自动选 GPU。预计算 1,343 条训练文本耗时 27.18 秒，处理 27,420 个本地词元。两组共 52 次分类器拟合，模型 API 请求 0 次，API 费用 ¥0。全过程使用训练数据，正式测试和 H2 封存。

[发布者与固定权重](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3) · [事前对照方案](../research/paired_encoder_search_1127/last_token_subset_protocol.json) · [两组成绩和决策](../research/paired_encoder_search_1127/last_token_subset_results.json) · [设备计时](../research/paired_encoder_search_1127/last_token_device.json)
