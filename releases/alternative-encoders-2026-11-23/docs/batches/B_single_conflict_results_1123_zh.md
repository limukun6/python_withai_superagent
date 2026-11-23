# Agent 学会在新证据足够时换模型，也能坚持原选择

旧动态策略在台词开局 E5 领先 2 对、首次确认普通十二层领先 4 对时，直接交付了开局领先者。为这个具体分歧增加可选参数 `--resolve-single-conflict`：仅当组合未晋级、两个内折支持不同单模型、预算还容得下成对复核时，才启用第三划分。接替者必须在新评价的共享文本组上占优，还须在三次观测按不重复原行计分时领先至少 0.2 个百分点。研究代码与原主线使用同一目录。[预登记的判定规则](B_single_conflict_1123_zh.md)

| 从训练中整部留出的影片 | 旧动态 | 新动态 | 第三份内部证据如何影响选择 |
| --- | ---: | ---: | --- |
| 原诊断划分，136 部，450 对 | E5，335/450，8 次拟合 | 普通十二层，344/450，10 次拟合 | 在此前未评的内部文本组上，十二层净胜 14 对。三份证据共覆盖 756 个不同原行，十二层平均领先 1.57 个百分点。 |
| 另一划分，95 部，276 对 | 普通十二层，211/276，8 次拟合 | 普通十二层，211/276，10 次拟合 | 六层挑战开局领先的十二层，在此前未评的内部文本组上净少对 8 对。Agent 保留十二层，两个独立运行的外层预测逐行一致。 |

第一次的留出表现已经参与发现原策略的问题。新策略在这组净增 9 对，按影片分组的 95% 区间为 −0.93 至 +5.00 个百分点。第二次在差异规则固定后，从原来的 1,744 对开发台词中单独留出 95 部影片，剩下 1,468 对用于重新训练，两组均得 **76.45%**。该次额外两次确认提高了决策所用证据量，外层准确率持平。两次合起来说明同一规则在真实运行中既能改选也能保留。外推到其他任务仍要看同预算比较。[诊断回执](../../research/b_single_conflict_1123/diagnostic/results.json) · [复核登记](B_single_conflict_replication_1123_zh.md) · [复核决策和成绩](../../research/b_single_conflict_1123/replication/results.json)

对 Task2 已存的两条真实路径进行逐步回放：种子 18643 的 9 个动作仍选择 E5＋带词频特征的十二层，种子 40531 的 6 个动作仍选择单独的词频十二层。此回放新增拟合 0 次，不会产生一项新的课程验证成绩。此项研究实际新增 **28 次本地拟合**，诊断路径 10 次，复核中新旧两条各 10 和 8 次，模型 API 0 次。[两条标题路径的动作核对](../../research/b_single_conflict_1123/diagnostic/title_replay.json)

我们在队友 A117 基线的本地独立副本上先应用联合编码器补丁，再接入本选项，通过 26 项相关测试，重现诊断路径的全部 10 个动作和 450 行预测。运行 `python scripts/verify_b_single_conflict.py` 可核对两份互斥的影片留出声明、训练回执、两类决策和跨版本导出。[A117 接入回执](../../research/b_single_conflict_1123/a117/receipt.json)

**复跑入口。** 在 HJL 分支仓库根目录执行下面两行可在独立检出中完成语境检查与接入，然后在发布子目录运行 `python -m unittest tests.test_ab_single_conflict`。A117 请在已接入联合编码器的根目录使用 A117 版增量补丁。两个变体均通过 `git apply --check`。[HJL 发布包增量补丁](../integration/B_joint_single_conflict.patch) · [A117 联合编码器版增量补丁](../integration/A117_B_single_conflict.patch) · [A117 所需的前置补丁](../integration/A117_B_joint_encoder.patch)

```sh
git apply --check --directory=releases/alternative-encoders-2026-11-23 releases/alternative-encoders-2026-11-23/docs/integration/B_joint_single_conflict.patch
git apply --directory=releases/alternative-encoders-2026-11-23 releases/alternative-encoders-2026-11-23/docs/integration/B_joint_single_conflict.patch
```
