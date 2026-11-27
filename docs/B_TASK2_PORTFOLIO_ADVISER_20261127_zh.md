# Task2 交接：Agent 自主组合两个编码器和相似案例投票

课程指定验证集共 940 条，最新统一入口从原包运行，答对 **710 条（75.53%）**。启动时只给数据、通用资源目录、22 次拟合和 1,800 秒上限。两条路径均完成全量重训，另起进程重现了全部预测。拟合次数包含重训，两组模型 API 调用均为 0。

| 运行 | 实际拟合 | Agent 自选结果 | 指定验证 |
| --- | ---: | --- | ---: |
| 原调度，22 次上限 | 21 | 本地发现的 DistilRoBERTa | 682/940（72.55%） |
| 新调度，22 次上限 | 20 | BGE-small 37.5%、GTE-base 37.5%、相似案例投票 25% | **710/940（75.53%）** |
| 上一版，16 次上限 | 16 | GTE-base 75%、相似案例投票 25% | 702/940（74.68%） |
| 原调度，16 次上限 | 16 | BGE-small、GTE-base 各 50% | 701/940（74.57%） |

新版对同预算原调度改对 72 条、改错 44 条，净增 28 条。对上一版 702 改对 30 条、改错 22 条，净增 8 条。两组 22 次路线同时改变了搜索阶段能用的训练次数和是否允许顾问加入，28 条是整条运行路径的差距。课程指定验证已用于前期开发。本轮封存预测后统一次计分，后续训练研究使用训练集内部划分。课程正式测试和 EuroSAT H2 均未打开。[两组启动规则](../research/paired_encoder_search_1127/portfolio_adviser_raw_protocol.json) · [评分及配对数量](../research/paired_encoder_search_1127/portfolio_adviser_score.json)

### Agent 为什么接受了第三票

它先在 16 次拟合内选出一个可交付的双编码器。两个成员都能提供表示，Agent 按各自首轮内部验证表现选中 GTE-base，以它的向量寻找相似训练案例。每个独立训练组最多投一票。第三票复用 GTE-base 的表示，整份模型共用两套预训练权重。

首份留出集上，加入 25% 顾问后，1,693 道中改对 35 道、改错 36 道。因为两者接近，又确实能纠正一部分原有错误，Agent 在读取后两份反馈前锁定了 25%。在按共享文本分开的两批题目中，加入它分别多答对 8/1,645 和 14/1,651。三个批次净增 21/4,989，达到预先设定的接纳门槛。人没有指定入选模型、投票来源或比例。开发者提供的是通用方法、可用权重和准入规则。[三批训练内反馈和最终权重](../research/paired_encoder_search_1127/portfolio_adviser_decision.json)

把两个编码器的首轮反馈互换，Agent 改用 BGE-small 提供相似案例。给成员改名并调换展示顺序，选择保持一致。把顾问的预测换成和现有模型相同或方向相反的意见，顾问权重都变为 0%。新版普通调度在三套匿名名称下共重放 45 个动作，均还原实际选择。[决策回放](../research/paired_encoder_search_1127/portfolio_adviser_audit.json)

### 重跑与交付

```sh
python -m unified_agent solve "$TASK_PACKAGE" --weights-dir "$WEIGHTS_DIR" \
  --fit-limit 22 --time-limit 1800 --run-dir "$RUN_DIR" \
  --prediction-input "$FEATURE_PACKAGE" --output "$PREDICTIONS"
```

使用交付环境中相同的资源目录可以重跑上述预算。仓库包含运行实现、评分与重载回执。含权重和已拟合索引的本地交付包为 `Released-Task2-validation-95c3d7f5c681.zip`，331,792,378 字节。整包移到新目录、禁网、新进程加载后逐字节重现 940 行预测。[包及模型 SHA-256](../research/paired_encoder_search_1127/portfolio_adviser_portable.json) · [上一版的复现说明](B_TASK2_NEAR_TIE_20261127_zh.md)

### 从相似案例学习权重的试验

另一条路线让分类器学习哪些类似的比较更有参考价值。它用冻结的编码器计算两句话的改动方向和上下文，再从训练组抽取代表样本，学习这些样本对预测的影响。两种向量来源都取自 Agent 实际选出的成员。首轮 1,693 条中，GTE-base 方案多答对 12 条，BGE-small 方案多答对 6 条。程序锁定前者和 50% 权重。换到 1,645 条复核数据，原组合答对 1,196 条，加新模型后只答对 1,175 条，触发事先写好的淘汰规则。正式模型继续使用原权重。

共发生 4 次本地分类器拟合。其中 3 次产生可评分预测，首次拟合因保存校验值时的数组维度错误，需要补跑一次。这 4 次都已计数，模型 API 为 0。此轮只比较训练内表现。[预设试验与方法来源](../research/paired_encoder_search_1127/preference_kernel_protocol.json) · [完整结果](../research/paired_encoder_search_1127/preference_kernel_results.json) · [首次拟合记录](../research/paired_encoder_search_1127/preference_kernel_first_attempt.json)
