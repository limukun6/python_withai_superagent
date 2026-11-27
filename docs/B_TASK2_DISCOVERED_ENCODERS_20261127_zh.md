# Agent 从资源目录发现新编码器

将带标准 Sentence Transformers 模块说明的通用编码器放进资源目录后，普通入口可以读取模型结构、汇聚方式、长度和归一化设置，生成可训练的文本分类或文本对偏好候选。课程任务可以禁止预训练资源。当前识别单层平均、首词元、最大池化及可选归一化；其他模块组合由既有方法接手。研究人员准备了模型下载和环境。运行中的 Agent 自己决定试哪些候选、何时复核、是否融合和最终选择。

| 同一套通用调度 | 实际选择 | 实际拟合（含重训） | 独立评价 |
| --- | --- | ---: | --- |
| Task2，原候选目录，普通默认 | GTE-base | 10 | 700 / 940，74.47% |
| Task2，原候选目录，预算 28 | GTE-small 与 GTE-base | 25 | 693 / 940，73.72% |
| Task2，增加可自动发现资源，预算 28 | MPNet 与 GTE-small 二次特征头等权 | 28 | 693 / 940，73.72% |
| 匿名短信，原目录，预算 18 | 带非线性词汇特征的 MiniLM | 7 | 153 / 160，macro-F1 0.8930 |
| 匿名短信，增加可自动发现资源，预算 18 | DistilRoBERTa | 9 | 152 / 160，macro-F1 0.8748 |

Task2 的首次菜单含 71 个选择，已安装的 MPNet、DistilRoBERTa、paraphrase-MiniLM-L3 都由 Agent 实际训练。MPNet 进入后续两份独立划分的复核，并最终入选。正式运行封存并独立重载后，才打开课程指定验证标签一次。与原预算 28 的融合相比，新组合改对 35 道，也改错 35 道。与默认交付相比，新组合改对 47 道，改错 54 道。27 步决策分别更换三套匿名编号并逆序呈现，81 次选择都与原记录一致。

短信使用公开 UCI 数据中原先划给开发训练的样本，取 640 条作新任务输入，另留 160 条作评价。Agent 按题目给定的 macro-F1 独立选择了另一种编码器。旧目录对照是在看到新目录短信分数后才追加，代码和资源均沿用冻结版本。这两个任务使我们观察到了跨任务改选和新模型的可执行性。陌生任务总体能提高多少仍待独立验收。

本轮两个新目录正式运行及短信旧目录对照分别拟合 28、9、7 次。另外特意指定最小权重做了 1 次迁移验收，用新进程和改名目录逐条复现 160 条预测与概率。Task2 的离线交付包在搬迁后重载，940 条完全一致。模型 API 调用 0 次。正式测试集和 EuroSAT H2 继续封存。

[事前实验方案](../research/paired_encoder_search_1127/discovery_protocol.json) · [训练、决策及课程验证](../research/paired_encoder_search_1127/discovery_score.json) · [匿名复核](../research/paired_encoder_search_1127/discovery_decisions.json) · [可搬迁交付回执](../research/paired_encoder_search_1127/discovery_portable.json) · [短信结果](../research/paired_encoder_search_1127/discovery_sms.json) · [短信对照](../research/paired_encoder_search_1127/discovery_sms_control.json) · [预训练资源来源](../research/paired_encoder_search_1127/discovery_assets.json)
