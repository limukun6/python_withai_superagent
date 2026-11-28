# 一个免费轻量编码器进入候选后，Agent 怎样选择

这次加入 Potion-base-8M。发布者用通用 BGE 模型蒸馏出一张词向量表，用 CPU 就能把一段文本变成 256 维向量。下载的模型、分词器和配置共 30,922,337 字节，使用 MIT 许可。接入按资源声明的静态编码和归一化操作工作，同样支持其他采用该格式的资源。用 96 条训练文本和 9 条边界输入与 Sentence Transformers 3.4.1 对比，最大绝对误差为 5.96 × 10⁻⁸。[发布者说明](https://huggingface.co/minishlab/potion-base-8M) · [固定版本及文件指纹](../research/paired_encoder_search_1127/static_install_receipt.json) · [对照回执](../research/paired_encoder_search_1127/static_reference_parity.json)

**普通入口把它列入候选，最后选择了 MPNet 与 BGE-small。** 在 6,590 条训练题和相同的 32 次上限下，首轮候选从 72 个增至 73 个。Agent 实际做了 23 次拟合，本轮没有安排 Potion 训练。新进程在 1,711 条留出上重现 1,285 条正确，逐行预测与加入资源前一致。[候选与排序](../research/paired_encoder_search_1127/static_menu_trace.json) · [实际试验记录](../research/paired_encoder_search_1127/static_menu_choice.json) · [逐行对照](../research/paired_encoder_search_1127/static_menu_results.json)

为了区分“没有试到”和“试了也没有改善”，又给它一次与旧方法相同的训练机会。所有模型在同样的 2,685 行上训练。在另一组 1,319 题上，原组合答对 931 条，加入 25% 的 Potion 后为 916 条，加入 50% 为 907 条，Potion 单用为 874 条。自动选择保留原组合，按协议结束这一组。新模型留在通用可用目录中，由后续任务的反馈决定是否使用。[候选得失和选择](../research/paired_encoder_search_1127/static_audition_nomination.json)

普通运行和额外一次对照共用 24 次本地拟合，模型 API 调用 0 次。以上分数来自课程训练集内部。当前已核实的指定验证交付仍为 710/940（75.53%）。[可提交版](https://github.com/limukun6/python_withai_superagent/pull/4)
