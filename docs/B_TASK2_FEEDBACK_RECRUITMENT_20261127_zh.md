# 扩大预算后，Agent 会不会重新考虑一个落选模型

课程指定验证成绩已核实的交付仍在草稿 PR #4，710/940（75.53%），运行源码指纹为 `c7982b0159ad`。本研发分支考察一个不同的问题。多给几次训练机会，能否让 Agent 自己发现，一个单独得分稍低的模型能帮当前模型改对一些题。[可提交版](https://github.com/limukun6/python_withai_superagent/pull/4)

### 一次真正从头开始的试验

从课程训练集重新按共享标题分组，6,590 条交给统一入口，1,711 条单独留出。两边获得同一通用模型目录、32 次拟合上限和 1,800 秒。运行只收到匿名任务、无标签预测输入和预算。先封存两边的模型与预测，由独立进程重载后才对这 1,711 条评分。[分组与预算](../research/paired_encoder_search_1127/recruitment_inputs.json) · [封存和评分](../research/paired_encoder_search_1127/recruitment_comparison_results.json)

| 运行 | 实际拟合 | Agent 最后选择 | 留出答对 | 准确率 |
| --- | ---: | --- | ---: | ---: |
| 原调度 | 28 | MPNet | 1,268/1,711 | 74.11% |
| 新调度 | 23 | MPNet 和 BGE-small 等权 | 1,285/1,711 | 75.10% |

新版先保留已训练的 MPNet，再回看其余实际跑过的方法。第一次训练反馈中，加入 50% 的 BGE-small 得分最高。锁定这个选择后，它购买两份此前未评分的训练划分，分别净增 7/1,306、持平 0/1,273，达到事先设置的平均提升门槛，才全量重训新成员。最终比自己保留的 MPNet 改对 67 条、改错 50 条。按标题组重采样的 95% 区间为 −0.18 至 +2.24 个百分点，仍需更多独立任务判断收益能否推广。[所有候选和两次复核](../research/paired_encoder_search_1127/recruitment_comparison_decision.json)

用同一入口跑公开 Wine 三分类，指定指标为宏平均 F1。Agent 只用 5 次拟合，保留梯度提升树，36 条留出均答对，宏平均 F1 为 1.00。这个小例子验证了它能切换指标和方法，也能主动保留单模型。[三分类决策](../research/paired_encoder_search_1127/recruitment_table_decision.json) · [三分类结果](../research/paired_encoder_search_1127/recruitment_table_results.json)

第二批另取 1,645 条，与前一批留出没有共同标题组。Agent 用 24 次拟合，自选 MPNet 和 GTE-base 各半，又提出给相似案例投票 50% 权重。训练内的两次检验分别多答对 14/1,376、少答对 4/1,334，于是保留双模型。新一批答对 1,219/1,645（74.10%），与这次保存的原组合完全相同。两批共 3,356 条，相对各自的原组合合计净增 17 条（+0.51 个百分点）。严格要求“两批都提升”的条件没有达到。[第二批真实决策](../research/paired_encoder_search_1127/recruitment_replication_decision.json) · [两批汇总](../research/paired_encoder_search_1127/recruitment_replication_panels.json)

### 回到完整课程原包

固定原交付的通用模型目录，只给 32 次拟合和 1,800 秒上限。Agent 用课程原包的 8,301 条训练题，从头训练 20 次后已得到原三成员组合，随后主动提出给 MPNet 50% 票权。在两份此前未评分的训练划分上分别净增 14/1,616、29/1,696，遂采用新组合。最终权重为 MPNet 50%、BGE-small 18.75%、GTE-base 18.75%、GTE-base 的相似案例投票 12.5%。全程共 29 次拟合，约 10 分 45 秒，模型 API 0 次。[自主决策与新划分](../research/paired_encoder_search_1127/recruitment_raw_decision.json) · [用量](../research/paired_encoder_search_1127/recruitment_raw_outcome.json)

对无标签的 940 行输入，模型在新进程逐行重现。相对可提交版改变了 65 行；本轮按事前约定封存指定验证答案，因此这个四成员模型的指定验证准确率仍未知。保留可提交版的 710/940。[文件读取与重载](../research/paired_encoder_search_1127/recruitment_raw_sealed.json) · [无标签比对](../research/paired_encoder_search_1127/recruitment_raw_changes.json)

### 决策依据和两条未晋级的路线

- 在已结束的标题和电影台词运行上，分别匿名改名并打乱方法五次，同一份反馈的十次选择都一致。用原始反馈，标题选 MPNet 50%，台词选非线性 MiniLM 25%。把候选预测改成与原方案相同后，两次都决定保留原方案。把可用预算降到少一次拟合后，两次也都保留原方案。改写反馈的实验只用于检验决策机制。[回放记录](../research/paired_encoder_search_1127/recruitment_replay_results.json)
- 让 Agent 根据每道题上不同模型的信心决定听谁，在 8,301 道训练内留出上，等权融合答对 6,127 条，逐题决策答对 6,133 条。五折中两折提高，按预先规则结束。实际训练 15 个小型判断器，复用全部底层模型。[逐折结果](../research/paired_encoder_search_1127/disagreement_router_folds.json)
- 尝试让一个模型学习两个模型给出的概率。七个实测方法中，程序选 BGE-small 作为单模型，选 MPNet 和相似案例投票提供概率。三种训练方式使用相同的 4,003 条真实训练数据。在另留的 1,309 题上，只用真实标签答对 911 条，用均匀概率平滑答对 909 条，用两个模型的判断辅助答对 905 条。三次拟合均完成，按事先规则结束。本研究借用知识蒸馏的想法。[实验记录](../research/paired_encoder_search_1127/soft_teacher_results.json) · [知识蒸馏论文](https://arxiv.org/abs/1503.02531)

上述训练和决策均在本地完成，模型 API 调用和费用均为零。课程正式测试和封存迁移题 H2 均未打开。

### 有多少错题可以通过“换一个成员回答”解决

在 8,301 条课程训练题上，四成员组合两次五折分别答错 2,174、2,162 条。其中 973、978 条是四个成员全错，另有 1,201、1,184 条至少一个成员判断正确，但融合后答错。两次分组均四个全错的有 749 条，两次均被融合压过正确答案的有 719 条。这些是事后计数，未送入 Agent。哪一个成员答对，实际预测时仍要靠可用信息判断。[按成员和错误类型统计](../research/paired_encoder_search_1127/routing_headroom_results.json)

在意见相反时，用同一折内真正留出的相似文本学该听谁。保持相似关系，两次分别比原融合多答对 3 条和 1 条。打乱文本与历史反馈的对应关系，分别少答对 12 条和 11 条。正确配对有帮助，但其净值未达到采用门槛。另一次使用单次运行中的三个训练划分，积累 721 条分歧、663 个独立标题组。调整后的规则在后两份内部检验分别少答对 4 条和 1 条，程序保留原投票。单独记录的候选在 1,711 题也少答对 2 条。第二次运行本来就拒绝额外成员，按条件跳过该项。[相似案例与打乱对照](../research/paired_encoder_search_1127/routing_neighbors_results.json) · [合并反馈的实际收益](../research/paired_encoder_search_1127/pooled_disagreement_first.json)

另给全新分类器一次训练机会。按内部成绩选中 BGE-small 和 MPNet，将两种向量及文本外形统计交给浅层梯度提升树。在 1,318 条训练留出中，原四成员全错的 185 题里，它单独改对了 33 题。但给它 25% 权重，整体改对 27 条、改错 33 条。因此反馈选择的权重为零，原模型继续交付。[方法、选中的表示和逐题得失](../research/paired_encoder_search_1127/semantic_tree_results.json)

全量训练生成的四模型包换目录后，在断网条件下重现 940 行逐行预测。压缩包为 `Released-Task2-research-unscored-7c7ab56f00d5.zip`，容量 739,262,062 字节。它与已有成绩的交付在 65 行上判断不同，这一版的课程指定验证分数仍未知。[权重、压缩包及独立预测指纹](../research/paired_encoder_search_1127/recruitment_raw_portable.json)

最后，用三个较早训练划分的真实错题教一个小模型在原预测上作修正，两个独立运行各训练一次。第一批两次内部检查分别净增 3 条、净减 17 条，第二批分别净减 13 条、4 条，Agent 两次均拒绝修正。封存后单独给被否决的修正规则评分，第一批净增 6/1,711，第二批净减 14/1,645。两批原预测均成功保留。[第一批](../research/paired_encoder_search_1127/feedback_residual_first_results.json) · [第二批](../research/paired_encoder_search_1127/feedback_residual_second_results.json)
