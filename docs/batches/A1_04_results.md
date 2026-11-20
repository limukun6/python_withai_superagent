# A1-04：raw/checked 已接入正式 CLI、累计账本与失败停止

## 瓶颈与结构

此前 CheckedHistoryPolicy 只能由研究脚本手工构造，缺少标准入口身份和统一任务账本。本次增加 `--planner raw-history` / `--planner checked-history`，由正式 CLI 构造相同 ExperimentLLM、流式传输、TaskBudgetAccount 和对应策略。复用搜索、三划分确认、完整 refit 与模型导出。

manifest 记录 planner 身份、context_version=history-evidence-v2、最大请求数和失败策略；不兼容的审阅架构、研究生成与资源 allocator 组合在加载数据前拒绝。B 的动态调度未在本批接入，不把两组固定复核入口说成六组全部就绪。

请求或提案错误立即抛出并保存 stopped outcome，不调用下一请求或随机恢复。已完成 trial 模型单独列为保留产物，未完成 refit 不冒充最终可部署模型。缺 usage 或传输故障使总费用保持 null，并保留任务账本的未决阻断；无效 JSON 且 token 已知时费用照记，当前运行依然停止。

## 实验与结果

build_parser 的正式 run handler，使用真实三分类数值任务、真实训练、确认和 refit。仅 HTTP Session 响应被替换，ExperimentLLM 的 SSE 解析、usage 核验、请求上限、费用计算与 SQLite 累计均真实执行。此处 token 与金额为模拟，不能计作学校账户消费。

| 验收 | 结果 |
|---|---|
| 旧 CLI 是否支持两模式 | 不支持，baseline 5 个测试未通过，0 拟合/请求 |
| raw 与 checked 正式训练路径 | 都完成：各2搜索＋4确认＋1refit，模型导出成功 |
| 两次运行累计同一任务账本 | 正确，各一次模拟 HTTP，累计折算 0.00036 元（仅模拟） |
| raw/E 信息边界与 manifest | raw 无 checks，E 有 checks；各最大1请求，明确失败停止 |
| timeout | 第1请求后停止，未知费用，任务后续请求被账本阻止 |
| 缺 usage | 第1请求后停止，完整费用 null，账本保留未决 |
| JSON 无效但 usage 已知 | 第1请求后停止，0.00018 元模拟折算保留，不随机恢复 |
| 不兼容架构/研究/allocator | 在读取任务前拒绝 |
| 独立进程重载 | 两组各15条验证预测与导出记录逐行一致，无新增拟合 |

最终入口验收 5 项全部通过，包含以上多个子断言；机器墙钟 29.218 秒，17 次真实分类器拟合、5 次模拟 HTTP、真实 API=0。另有一次初轮验收因测试写错枚举 self-review（实际 self_review）而失败，训练与故障部分已执行17拟合/5模拟HTTP，保留记录；修正后新目录完整重跑。因此本批 CLI 验收累计 **34 次真实拟合、10 次模拟 HTTP、0 次真实 API**。

旧 CLI/账本与 A1 回归最终 **141 项通过**，墙钟 43.281 秒，source_unchanged=true。首次回归发现旧测试默认编码读取 UTF-8 JSON，在中文路径中产生解析错误；显式 UTF-8 后新目录完整重跑。回归自身的模型夹具与模拟请求未逐项汇总，不包含在上述34/10的 CLI 实验计数内；真实 API 仍为0。

## 产物、交接与边界

原始：主仓库 runs/a1_04_baseline、a1_04_fixed、a1_04_fixed_final、a1_04_regression、a1_04_regression_final。失败记录不覆盖。

归档：research/a1_cli_entry，含源码指纹、日志、manifest/outcome、决策与账本摘要、模型重载核对；不归档任何 credential、训练包、模型或 SQLite。命令说明见 docs/history_planner_cli_zh.md。

这证明正式入口能完成两策略的模型开发和可追踪的费用/失败处理，不证明真实 Flash 响应质量、模型资格/价格依据或性能提升。原定报价规则沿用当前仓库实现，本批不核实外部价格。代码尚在工作区，源码身份以内容哈希为准。

下一步：按 A2 制定真实任务状态清单、冻结协议和付费 runner，先检查现有真实任务累计账本，再做真实 Flash 同历史决策对照；不能用本批模拟账本替代课程真实账本。B 的强方法、图像与调度线并行按共同计划推进。
