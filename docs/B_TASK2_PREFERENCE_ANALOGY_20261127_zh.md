# Agent 能否从相似的训练例子学会什么时候应该融合

利用已选模型的文本向量，找出训练中改法相似、上下文也接近的例子。每个相连标题组至多投一票。Agent 在一个学习划分选择相似案例应占的比重，然后在不同的标题组上计分。

在两次重新分组的五折训练内实验上，这种选择分别比对应 GTE-base 多答对 38 和 37 行，每次覆盖相同 8,301 对。十折中八折改善，一折持平，一折下降。顾问单独使用时分别为 5,713 和 5,680，对照为 6,051 和 6,039。这一步包含 10 次利用标签建立索引和 20 次融合比例选择。

从课程原包启动后，真正的统一入口自行选择 GTE-base，随后试验从其表示派生的顾问。首份训练内留出有 1,693 对。按顾问比重 0%、25%、50%、100%，依次答对 1,228、1,227、1,225、1,143。按事前决策条件，Agent 保留 0% 的原模型，共用 11 次拟合，指定验证 700/940。独立的 16 次同预算上限对照选择 BGE-small 与 GTE-base，701/940。较少训练机会的默认基线也是 700/940。

两条路线封存并在新进程重载后才计分。45 次原版和 30 次新版匿名动作回放全部一致。此次本地运行模型 API 0 次，正式测试、H2 均保持封存。首轮近似并列能否获益于额外证据，另行登记研究。

[十折成绩](../research/paired_encoder_search_1127/preference_neighbors_results.json) · [正式运行的事前规则](../research/paired_encoder_search_1127/analogy_raw_protocol.json) · [Agent 的实际选择](../research/paired_encoder_search_1127/analogy_raw_decision.json) · [两组指定验证](../research/paired_encoder_search_1127/analogy_raw_score.json) · [匿名和反事实回放](../research/paired_encoder_search_1127/analogy_raw_replays.json)
