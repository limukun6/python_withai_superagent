# 单 Agent 原型接入与运行

本原型基于团队提交 `68e6772daa9ced52849c16ab694fefb88029a860`。源码包包含原项目及本地新增实现；补丁文件只包含本轮修改。先在独立分支检查补丁，与组员合并时保留共同的执行工具和数据合同。

## 变更位置

| 文件 | 作用 |
| --- | --- |
| `superagent/hypothesis_agent.py` | 状态、方案校验、预期核对与单 Agent 接口 |
| `superagent/benchmark.py` | 接入两种新策略、共用首轮、保存实验记忆和冻结产物 |
| `superagent/experiment_llm.py` | 每次运行的等额配额及实际用量超估后的停止保护 |
| `superagent/predict.py` | 读取 benchmark 的冻结模型包，兼容旧裸模型 |
| `tests/test_hypothesis_agent.py` | 新策略、反馈隔离、产物衔接和配额检查 |
| `examples/offline_demo.py` | 无密钥、无真实 API 的训练与预测演练 |

旧 `run` 入口保留旧规划逻辑，新策略通过 `python -m superagent.benchmark` 运行。统一预测使用 `python -m superagent predict`。

## 安装与离线验证

沿用项目 `requirements.txt` 安装依赖。以下命令从项目根目录运行，Windows 与 macOS 均可使用当前虚拟环境的 `python`：

```text
python -m unittest discover -s tests -v
python examples/offline_demo.py --output-dir runs/demo01
```

每次演练使用新的输出目录，防止覆盖既有证据。离线示例生成合成表格数据，并使用明确标注的脚本决策器。两个策略各做三次真实 Logistic Regression 训练，随后各启动新进程加载模型并预测。示例里的测试标签是无法解析的哨兵数据，预测不应加载它们。

示例输出 `demo_summary.json`、每个策略的 `trials.json`、事件与用量记录、冻结模型和预测 CSV。合成分数和脚本决策不能写入正式架构性能报告。

## 真实 Pilot

先核对接口、允许的模型版本与费用口径，再使用自己本地的密钥文件执行开发演练。不要把密钥写入代码、日志或提交版本。下面是一项任务、一个种子、三种策略的初始配置；额度只作为开发估算上限，不保证足以完成所有调用。

```text
python -m superagent.benchmark "data/released_task1.zip" --strategies feedback reflective reflective_no_feedback --seeds 11 --max-trials 3 --anchor-first --per-run-budget-rmb 0.2 --batch-budget-rmb 1 --api-key-file "api_key.txt" --model DeepSeek-V4-Pro
```

三次 LLM 运行各预留相同的 0.2 元课程核算额度，合计不超过批次上限。扩展任务或种子前要相应增加预分配总额。DeepSeek 使用更新后的官方空闲时段价格，不受调用时间影响；所有输入 Token 均按未命中缓存价核算。配置和来源见[计费说明](deepseek_pricing.md)。调用前仍须估算 Token，用量以响应中的实际总数核算。

输出目录中的协议文件保存参数和源码哈希。比较时核对 `valid_policy_run`、用量完整性、fallback、失败次数、真实 Token 与预期核对记录。预算不足、接口故障和策略决策错误应分开解释。保留所有运行，不通过反复重跑挑选某次有利结果。

## 预测合同

```text
python -m superagent predict "data/released_task1.zip" --run-dir "runs/benchmarks/BATCH/task0_categorical_tabular_reflective_11" --output "runs/final/predictions.csv"
```

`final_selection.json` 声明 `model_format: candidate_bundle_v1`，指向 `selected_model.joblib`。模型包包含 `kind`、`candidate` 和 `model`；预测器检查它们与任务合同一致后加载。旧的无 `model_format` 裸模型继续兼容。

当前 CSV 是继承的 `row_id,prediction` 格式，ID 从 0 开始。正式任务若规定其他 ID 或列名，应在公共输出适配层实现并由 Evaluator 验收。

## 后续接入顺序

先完成一个真实 Pilot，确认新策略能够提交合法方案并回顾实际结果；再补跨进程恢复账本、平台实际扣费对账、外部 Evaluator 和任务合同扩展。最终版本冻结后重新生成报告所用实验。所有 LLM 解释均是简短实验记录，不要求完整内部推理文本。
