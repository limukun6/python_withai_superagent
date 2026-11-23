# B 线补交 · 两项六组对照、48 个案例和 24 份盲标

队友可以直接使用本分支根目录的研究文件。六组使用同一份 C1 生产代码 `c64bee45c834878121f62c6198cd484a53a0b616`，输入与分区沿用 A94。以下是原批次的成绩。此次交接复用冻结记录，新增拟合与模型 API 请求均为 0。

| 组别 | Task2 课程指定验证（940 对） | Dry Bean A94 开发留出（2,178 行） |
| --- | ---: | ---: |
| L0 本地固定 | 668 · 71.06% | 1,989 · 91.32% |
| L1 本地动态 | 668 · 71.06% | 1,989 · 91.32% |
| R0 原始历史固定 | 668 · 71.06% | 1,986 · 91.18%（提前结束） |
| R1 原始历史动态 | 668 · 71.06% | 1,989 · 91.32% |
| E0 核对历史固定 | 655 · 69.68% | 1,989 · 91.32% |
| E1 核对历史动态 | 668 · 71.06% | 1,988 · 91.28%（提前结束） |

Task2 的五个 668 分组逐行相同，E0 相对 L0 多答对 72 对、多答错 85 对。Bean 四个完整组逐行相同。R0 的 Flash 提议改写了任务类型，控制器在 1 次拟合后结束。E1 的第三次 Flash 流式连接中断，留下经过 5 次拟合的检查点。全部 12 份终局模型均通过新进程预测核对，完整调度路径有 10 条。历史共 112 次拟合、22 次 Flash 请求，21 次有计量，输入 136,278、输出 20,779 token，按课程更新后的空闲时段价已知部分为 ¥0.219394。E1 的一笔缺少最终用量，完整总费用未知。[Task2 原始回执](../../research/b_c1_validation_1123/preference/) · [Bean 原始回执](../../research/b_c1_validation_1123/bean/) · [资源汇总](../../research/b_c1_validation_1123/resource_summary.json) · [累计账本交接摘要](../../research/b_c1_joint_delivery_1123/budget_handoff_receipt.json)

## 可直接整合的独立评估材料

| 用途 | 冻结材料 | 实测结果 |
| --- | --- | --- |
| A1 检查器 | [48 个含预期和依据的案例](../../research/b_integrated_offline_1123/a1_cases.json) · [A67 逐例结果](../../research/b_integrated_offline_1123/a1_check/results.json) | 八类各 6 例。45/48 与预期一致。三处差异均为检查器给出“未知”，错误确认 0 次。 |
| A2／D3 解释质量 | [原始遮蔽输入](../../research/a70_independent_review/blinded_reviews.json) · [24 份逐条标注及依据](../../research/b_integrated_offline_1123/a2_annotated_reviews.json) | 事实错误为“有”6 份、“无”8 份、证据不足 10 份。范围或指代错误 1 份。12 对相同历史状态中有 10 对选择相同。 |

48 例的预期先在 `6f29f84` 冻结，24 份标注在 `020ba75` 冻结。B 侧一位 Codex 标注者依据任务契约和当时历史独立给出判断，记录显示标注时未查看检查器输出、实验条件键或后续动作结果。人工双标与独立盲法审计尚无记录。A67 通用脚本在其回执中写入固定的 A 侧夹具说明；该次实际输入可用 48 个 B 侧 ID 和冻结哈希核对。[冻结方案](B_integrated_offline_1123_plan.md) · [独立汇总及输入指纹](../../research/b_integrated_offline_1123/independent_summary/)

在仓库根目录运行 `python3 scripts/verify_b_c1_d3_handoff.py`，可核对 12 组模型和预测回执、48 个冻结预期与完整混淆矩阵，以及 24 份标注对原文的保留情况。发布前另逐组读取本机冻结预测，按对应留出标签重新计分。Git 交付包含汇总、指纹及去标识盲标材料，原始数据、模型权重和 SQLite 通过原有渠道保管。

## 四个案例的构造与三处判断分歧

原夹具的切片固定为 20 行。四个原例给出的准确率 72%、62.5%、84%、48% 分别对应 14.4、12.5、16.8、9.6 行答对，均无法在 20 行等权样本中观测。原 48 例的冻结记录仍为 45/48。只看可实现的 44 个原例，检查器与预期一致 41 次。事后构造的 4 个补例都改用了整数行数，逐例均与预期一致。两次观测的来源与原始 ID 均可复核。[四个补例与原例映射](../../research/b_integrated_offline_1123/a1_slice_supplement.json) · [四个逐例输出](../../research/b_integrated_offline_1123/a1_slice_sensitivity/results.json) · [构造核验](../../research/b_integrated_offline_1123/realizable_cases/results.json)

三处分歧均在 44 个可实现的原例中。`B_current_kernel` 的事实是 RBF SVM，解释却声称本轮用了逻辑回归。检查器把该陈述当作比较背景。`B_denial_matches` 与 `B_denial_conflicts` 都是“本轮没用逻辑回归”，前者实际使用 Extra Trees，后者实际使用逻辑回归。检查器对两句否定陈述均返回“未知”。这是 B 对现有原件的复核，A 或另一位审阅者可用原始事实做独立裁决。24 份解释的空白盲标表已附在同一分支，供独立复标时直接使用。[三处分歧及检查器判据](../../research/b_integrated_offline_1123/a1_check/results.json) · [24 份空白盲标表](../../research/a70_independent_review/blinded_reviews.json)
