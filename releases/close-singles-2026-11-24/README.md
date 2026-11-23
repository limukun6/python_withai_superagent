# Task2 的 Agent 什么时候值得追加单模型复核

**保留共同默认调度。** 增加一个按需开启的选项 `--confirm-close-singles`。原调度即将停在单模型、多个方法在训练内分数接近、而且它们各自能改对对方的错题时，Agent 可以花两次拟合把领先者与挑战者放在同一批新标题组上比较。它用观察到的预测选挑战者，保留最终重训预算。已有值得复核的组合时，沿用原动作。[事前方案](overlay/docs/batches/B_autonomous_close_singles_plan_20261124_zh.md)

| 实验 | Agent 实际做了什么 | 结果 | 当日新增拟合 |
|---|---|---|---:|
| 共同版与可选版从头运行 | 两组都从 35 个候选中做了 5 次探索、2 次确认和 2 个组件重训，都选了 E5-small 与 MiniLM-L6 等权投票 | 共同版 676/937（72.15%），可选版 678/937（72.36%） | 9 + 9 |
| 完整 8,301 行训练历史续跑 | 复用旧版的 5 次初探。Agent 自己选 E5-small 挑战带语料词频特征的 MiniLM-L12，追加两次匹配确认，再全量重训胜者 | 1,358 对首轮未见标题组中，MiniLM-L12 答对 962，E5 答对 947。保留 MiniLM-L12 | 3 |
| 分类头重训稳定性 | 两个独立进程，各用同样的训练数据和冻结编码器比较两种求解容差 | 同容差跨进程的系数和预测完全相同。收紧容差后 937 对的投票类别也都相同 | 4 |

共新增 **25 次本地分类器拟合，学校模型 API 0 次，新增模型 API 费用 ¥0**。另有 5 次历史拟合用于续跑。个人分支的复现包包含两版可选代码、动作、拟合回执和交付预测。

## 如何读这三个结果

第一项在原 8,301 对课程训练标题中另留 937 对，涉及 702 个标题组。它们与本轮训练的 7,364 对以及方案列明的先前计分分区均无共享标题组。两组模型先冻结，再用独立进程各导出两遍。两组实际八个动作完全一样，七次探索和确认的硬预测、概率也逐项相同。留出净增 2 对，净差 +0.21 个百分点，按标题组配对抽样的 95% 区间为 0.00 至 +0.54 个百分点。本轮没有达到事前写定的调度晋级门槛。[从头对照](evidence/cold/results.json) · [逐组正确数](evidence/cold/groups.json)

在同一份特征矩阵上互换两个冻结分类头，937 行逐一复现了两份交付预测。两道题的变化全部跟随 MiniLM-L6 的逻辑回归头。E5 的两个分类头系数相同。原生全量重训的 MiniLM-L6 分别在第 33、32 次迭代结束。另一项受控重训中，默认容差两次均为 33 次，收紧到 `1e-6` 后两次均为 63 次。调严格没有改变投票类别；原来两个进程为何在不同迭代结束，仍未证实。[冻结分类头互换](evidence/cold/refit_audit.json) · [独立重训](evidence/repeatability/results.json)

第二项专门选择了已知新旧调度会分歧的一条历史。E5 在首轮改对了领先者的 122 道错题，因此获得确认机会。在与首轮无共享标题组的确认行上，E5 净少答对 15 对，Agent 保留原路线。该模型已在全部 8,301 对上重训，用 32 行训练内无标签特征做两次独立交付检查，输出一致。此次没有开课程指定验证或正式测试。此前共同版在指定验证的可引用成绩为 684/940（72.77%），B 历史最高为 689/940（73.30%）。[续跑动作与证据](evidence/resumed/model_frozen.json) · [旧成绩出处](../shared-task2-2026-11-23/README.md)

电影台词历史上，原调度已经复核一个组合，新增选项保留全部八个动作。两项历史重放均使用当时可见的候选和观测，电影任务保留历史目录里的两项 MiniLM-L12 版本信息。完整标题旧路径共六个动作，在第六步第一次分歧。阈值、时间或历史用量不满足追加条件时，接入版保留原路线。[历史重放](evidence/replay/after_guard.json) · [三条真实路径共 24 个动作的兼容验收](evidence/replay/observed_paths.json)

## 两个模型意见不同时，按题挑选能否再提高成绩

在双方预测冲突时，比较“这个模型对这道题的信心，在它自己的历史预测中排第几”。同意时沿用共同答案，百分位平手时沿用原交付策略。两个模型仍由当时的 Agent 选出。三个面板的预测先一起封存，再按事前规则检查；新增拟合和 API 均为零。[事前规则](overlay/docs/batches/B_disagreement_ranks_plan_20261124_zh.md)

| 已有开发面板 | 原交付正确 | 等权投票正确 | 按置信百分位正确 | 相对原交付的改对／改错 |
|---|---:|---:|---:|---:|
| 标题，937 对 | 676 | 676 | 680 | 6／2 |
| 全部标题训练包的确认区，1,358 对 | 962 | 968 | 968 | 58／52 |
| 电影台词确认区，223 对 | 166 | 166 | 166 | 3／3 |

937 对上达到 **72.57%**，比原交付增加 0.43 个百分点，标题组配对区间为 −0.11 至 +1.04 个百分点。两份标题面板共享 133 个标题组，各自计分。1,358 对上的等权投票也达到同样正确数，电影台词净变化为零。本次属于已见开发数据上的事后比较，未达到启动新自主对照的门槛。供队友作为后续研究选项，默认策略沿用原版。[三项结果与区间](evidence/rank-routing/results.json) · [冻结预测摘要](evidence/rank-routing/predictions_frozen.json)

## 下载后复核或接入

只核对数字及文件摘要可在本目录运行 `python verify.py`。安装 NumPy 后可重新计算置信区间。以下命令还会把两份补丁依次应用到临时副本，验证对照和两个实验版本的全部生产源码摘要。传入的本地 Git 仓库需含共同提交 `449b766`。

```sh
python verify.py --interval --base-repo "$SHARED_REPO"
```

要整合，在共同版 `449b766eb4a29af2f4020092ab6b3eab425d25be` 的干净检出依次应用 [01-measured.patch](01-measured.patch) 和 [02-budget.patch](02-budget.patch)，复制 [overlay](overlay/) 到检出根目录。第一份是双组从头运行的生产源码；第二份复用公共用量和时限检查，是完整标题续跑使用的源码。两个开关版本在已观察的动作上已核对一致。合入代码后默认关闭。可选模式限本地有向二元文本偏好、accuracy 和 `ab-adaptive-complement`。

```sh
git apply "$RELEASE/01-measured.patch"
git apply "$RELEASE/02-budget.patch"
cp -R "$RELEASE/overlay/." .
python -m unittest -q tests.test_ab_close_singles tests.test_ab_adaptive_opening tests.test_ab_encoder_coverage tests.test_ab_complement tests.test_ab_single_conflict tests.test_ab_encoder_routing tests.test_a137_numeric_kernel
```

复现双组从头实验需要两个干净检出。对照保持共同提交，实验组只应用第一份补丁，并把 `overlay/` 复制到实验组。使用含校验过的本地 MiniLM-L6、MiniLM-L12、E5-small 权重的目录。可从[原训练包转换函数](../shared-task2-2026-11-23/overlay/scripts/run_b_shared_full_task2.py)的 `release_rows(archive, 'train', ('headline_A','headline_B','label_pairwise'))` 和 `train_packet` 重建 8,301 行匿名训练包，运行脚本会先验证其 SHA-256。在实验组根目录运行以下命令，每次使用新的输出目录。

```sh
python -m scripts.run_b_autonomous_close_singles all --training "$TRAIN_ONLY_ZIP" --weights "$FROZEN_WEIGHTS" --control "$CONTROL_CHECKOUT" --changed . --out runs/close_singles
python -m scripts.audit_b_close_singles_refits --run runs/close_singles --weights "$FROZEN_WEIGHTS" --output runs/close_singles/refit_audit.json
```

复现已观察分歧的续跑需保有旧版[完整标题运行](../shared-task2-2026-11-23/)的本地模型与原始历史。在应用两份补丁的检出中，以其中的 `shared_full_task2_01` 作为 `ORIGINAL_RUN`，使用本包的 [before_guard.json](evidence/replay/before_guard.json) 和相同的冻结编码器。

```sh
python -m scripts.run_b_close_singles_continuation --original "$ORIGINAL_RUN" --replay "$RELEASE/evidence/replay/before_guard.json" --weights "$FROZEN_WEIGHTS" --output runs/full_continuation
python -m scripts.check_b_minilm_refit_repeatability --run "$MEASURED_COLD_RUN" --weights "$FROZEN_WEIGHTS" --output runs/refit_repeatability
```

持有前述三项本地完整历史时，可运行 `python -m scripts.check_b_disagreement_ranks --cold "$MEASURED_COLD_RUN" --full runs/full_continuation --original "$ORIGINAL_RUN" --movie "$MOVIE_RUN" --weights "$FROZEN_WEIGHTS" --output runs/disagreement_ranks` 复算逐题规则，其中 `MOVIE_RUN` 是原电影台词 `quote_adaptation_01` 目录。
