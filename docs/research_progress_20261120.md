# 通用反馈结构与近期实验入口

本次同步包含候选事实核验、目标分组假说检查、失败记忆和完整循环对照。当前代码位于 codex/expanded-feedback 分支。

## 近期报告

- [A/B 接口与共同验收约定](AB_INTEGRATION_V1.md)为队友接入入口，匿名逐行预测示例位于 examples/ab_v1，导出工具为 tools/export_b_handoff.py。
- [持续实验进度与续跑入口](EXPERIMENT_PROGRESS.md)：A1-01 至 A1-04 已完成；正式 raw/checked CLI、累计账本与失败停止已验证，5项入口验收和141项回归通过。每批先落计划和代码再运行，下一步按协议准备真实 A2。
- [raw/checked 正式入口说明](history_planner_cli_zh.md)包含离线验收命令、真实运行模板与故障语义。
- [当前九天计划：完整系统、通用性与效果](nine_day_two_track_plan_20261120_zh.md)为后续执行依据；两人均使用 Codex，分别负责任务全流程/证据闭环与强方法/预算搜索，新增真实图像和方法能力对照。配套预算见 [实验清单](nine_day_experiment_budget_20261120.json)。
- [DeepSeek 更新复核](deepseek_review_20261120.md)列出已复现的语义过滤、证据接入和错误确认问题。
- [通用完整循环对照](checked_loop_pilot_results_20261120.md)说明表格与文本三组比较及当前证据范围。
- [候选事实与目标检查](targeted_feedback_progress_20261120.md)记录结构改动、局部编辑动作与真实 Flash 试跑。
- [详细错误反馈对照](flash_error_feedback_v1_results_zh.md)记录四次 Flash 决策。
- [有界组合实验](composition_study_v1_results_zh.md)记录本地组合器与固定动作对照。
- [后续实验计划](下一阶段执行计划.md)以通用任务接口、失败后调整和跨任务迁移为优先方向。

## 代码与结果

superagent/targeted_feedback.py 提供分类器事实和目标分组检查。superagent/checked_history_policy.py 提供共享的原始历史、检查记忆两种策略。检查接入现有 contracts、benchmark、hypothesis_agent 和 hypothesis_memory。

research 下五个实验目录保存 protocol、results、completion 和原始 runner_snapshot.py，便于对照源码哈希与汇总结果。原始快照保留实验时的本机路径约定，用于追溯。汇总中的模型绝对路径也是历史定位信息，模型文件及逐行运行数据保存在实验机。

scripts 下提供可从当前仓库根目录执行的脚本，自动识别普通检出与原集成工作区。实验包放入 data/released_task1.zip 和 data/released_task2.zip。带模型决策的脚本从仓库根目录 api_key.txt 读取密钥。输出目录须使用新名称，已有目录会触发保护。

例如，完整循环先导实验可从仓库根目录运行。

```powershell
python scripts/run_checked_loop_pilot.py
```

该命令最多发送 12 次 Flash 请求并执行 66 次拟合，会产生 API 用量。部分脚本会校验前序实验目录或源码身份，应按报告中的依赖顺序执行。历史协议描述当时运行的代码版本，当前代码包含后续扩展，严格复现历史版本需要对应源码。

相关测试可通过以下命令运行。

```powershell
python -m unittest tests.test_targeted_feedback tests.test_hypothesis_agent tests.test_architecture_contracts -q
```

数据包、API 密钥、模型、特征缓存和完整运行目录沿用仓库忽略规则。发布摘要记录课程折算费用，学校账户实际扣费需单独核对。
