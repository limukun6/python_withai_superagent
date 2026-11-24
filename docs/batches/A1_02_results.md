# A1-02 结果：实际分类器与测量引用绑定

2026-11-20。计划与脚本先于 baseline 保存，随后修复、重跑。属于已知故障机制验证，不是封存案例集或真实任务性能结论。

## 瓶颈 → 结构

1. 原实际对象映射将 RidgeClassifier 报为 logistic，所有非线性 SVC 核报为 rbf，还将 LinearSVR 回归器归入分类器。现在 Ridge 使用自己的身份，SVC 记录原始 kernel 并区分 linear/rbf/poly/sigmoid/precomputed，未知 callable kernel 不猜测；回归器不再冒充分类器。
2. 原百分数只需匹配任意历史测量就能被确认。新增 measurement_claims，以 `(trial, split, metric, value)` 核验唯一已完成试验的指定测量，返回 evidence_id；失败、缺失或重复身份保留未知，错值记为矛盾。
3. 自由文本仅在受限的明确陈述语法下绑定测量；无法绑定的百分数、未来预测、否定和复杂指代保留无法核验。试验号确认只说明身份存在，不认证整句推理。清理重复 check_prose 定义。

## 实验 → 证据

| 固定开发案例 | 修复前达标 | 修复后达标 |
|---|---:|---:|
| 真实类身份/核函数，含合法对照 | 3/7 | 7/7 |
| 自由文本中的引用、指标、否定、失败与未来语态 | 3/12 | 12/12 |
| 合计 | 6/19 | 19/19 |

例如 trial1 搜索 60%，trial2 训练 95%：`Trial 1 search accuracy was 95%.` 现在为矛盾；`Trial 1 search accuracy was 60%.` 确认且带来源。`search accuracy 70%` 即使历史中存在 70%，缺少唯一引用也不确认。训练/搜索与 accuracy/macro_f1 的合法例仍能确认，因此并非拒绝一切引用来达到安全结果。

增加结构化核验的边界测试，覆盖缺失指标、失败试验、试验 ID 不连续、重复身份、NaN、错误 schema、主指标身份、容差单位；配置测试覆盖 callable kernel 与回归器。相关回归最终状态见 research/a1_evidence_binding/regression/completion.json 与 tests.log。

baseline 机制探针约 1.198 秒，最终 fixed 约 1.190 秒，不含 Python 启动。两者 API 均 0、完整分类器拟合均 0。类身份检查只构造对象，不进行训练；回归另含小模型夹具，未逐项计数，不能把回归也称为零拟合。

最终冻结源码回归 123 项通过，测试执行 40.905 秒，runner 墙钟 42.614 秒，source_unchanged=true；脚本与源码哈希可从归档协议复核。

## 限制与下一批

19 项均为已知故障的开发回归，不能推断新文本的检查准确率。自然语言覆盖故意收窄，需要同时报告 unknown 与覆盖率；没有解决一般否定逻辑、因果归因或统计显著性。现有日志上的旧“确认数”不能直接与新检查器比较，需要标记版本后另行重算。

本批没有将结构化测量 claim 加入 LLM 输出契约，没有把 actual estimator/evidence 投影到下一请求。A1-03 要先写计划与模拟客户端 runner，再完整走“提案→执行观测→核对→下一请求”，接入 hypothesis 字段与不带 assessment 的提案审计。之后才开始真实付费 A2。

原始目录：主仓库 runs/a1_02_baseline、a1_02_fixed_release、a1_02_regression_final。中间 fixed/fixed_final 及首次 regression 均保留；首次 regression 期间实现发生修改，不作为冻结源码回归证据。最终 regression 检查 source_unchanged，失败会明确退出。研究目录仅归档 baseline、最终 fixed 与最终 regression。

当前改动尚未提交，实际源码由 protocol 中内容哈希标识。对话额度不足时从 docs/EXPERIMENT_PROGRESS.md 接手，不覆盖已完成目录。
