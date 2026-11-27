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

### 决策依据和两条未晋级的路线

- 在已结束的标题和电影台词运行上，分别匿名改名并打乱方法五次，同一份反馈的十次选择都一致。用原始反馈，标题选 MPNet 50%，台词选非线性 MiniLM 25%。把候选预测改成与原方案相同后，两次都决定保留原方案。把可用预算降到少一次拟合后，两次也都保留原方案。改写反馈的实验只用于检验决策机制。[回放记录](../research/paired_encoder_search_1127/recruitment_replay_results.json)
- 让 Agent 根据每道题上不同模型的信心决定听谁，在 8,301 道训练内留出上，等权融合答对 6,127 条，逐题决策答对 6,133 条。五折中两折提高，按预先规则结束。实际训练 15 个小型判断器，复用全部底层模型。[逐折结果](../research/paired_encoder_search_1127/disagreement_router_folds.json)
- 尝试让一个模型学习两个模型给出的概率。七个实测方法中，程序选 BGE-small 作为单模型，选 MPNet 和相似案例投票提供概率。三种训练方式使用相同的 4,003 条真实训练数据。在另留的 1,309 题上，只用真实标签答对 911 条，用均匀概率平滑答对 909 条，用两个模型的判断辅助答对 905 条。三次拟合均完成，按事先规则结束。本研究借用知识蒸馏的想法。[实验记录](../research/paired_encoder_search_1127/soft_teacher_results.json) · [知识蒸馏论文](https://arxiv.org/abs/1503.02531)

上述训练和决策均在本地完成，模型 API 调用和费用均为零。课程正式测试和封存迁移题 H2 均未打开。
