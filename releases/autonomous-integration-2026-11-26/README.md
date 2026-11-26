# SuperAgent 最新研发交接（11 月 26 日）

此目录提供可独立运行的最新源码。完整研发分支 `codex/task-driven-planning-1125` 包含同一份入口、所有测试、实验脚本和复核证据。队友在研发分支根目录开发，可以直接沿用团队的测试和验收程序。[完整源码](https://github.com/limukun6/python_withai_superagent/tree/f03323574614e8876cde5078c7aa31f093b62a36) · [两代整合版核验与 Task1 开工说明](https://github.com/limukun6/python_withai_superagent/blob/f03323574614e8876cde5078c7aa31f093b62a36/docs/TASK1_DEVELOPMENT_HANDOFF_20261126_zh.md)

本轮两道课程任务用修复后的默认入口从头自主训练。Task1 指定验证答对 **1,533/2,030（75.52%）**，Task2 答对 **684/940（72.77%）**。每题 10 次拟合，模型 API 调用合计 0 次。个人包独立重载后，2,970 行预测全部重现。正式测试保留。[本轮验收证据](verification/task-contract.json) · [源码对应关系](verification/source.json) · [个人包独立重载](verification/personal-package-reload.json) · [前一版两代对照](verification/two-integrations.json)

此版能核对无表头数组的列名、保存标签含义、按题面点名的类别评分，并在训练前执行方法禁令。遇到含糊的字段说明时，可选的校园 DeepSeek 能辅助定位原文。程序保留证据来源及不确定状态。[任务契约修复说明](https://github.com/limukun6/python_withai_superagent/blob/f03323574614e8876cde5078c7aa31f093b62a36/docs/TASK_CONTRACT_EVIDENCE_20261126_zh.md)

## 运行 Task1

从此目录使用 Python 3.11。下面的 `TASK.zip` 是原始任务包，`FEATURES.zip` 是只含待预测特征的包，请选择新的运行目录。

```sh
python -m pip install -r requirements-tested.txt
python -m unified_agent solve TASK.zip --prediction-input FEATURES.zip --run-dir runs/task1-start --output runs/task1-start/predictions.csv
```

当时由 Agent 自动选出的候选是 `supported_field_interactions`。它从本轮训练数据找出有用的类别字段组合，再拟合 CatBoost。要评估新表格方法，可将其加入通用候选目录，在相同训练预算下交给原调度器比较。[候选入口](superagent/ab_methods.py) · [字段交互](superagent/field_interactions.py) · [调度和内部选模](superagent/experiment_policy.py)

## 复现编码器环境与可选功能

复现 Task2 的候选范围时，安装 `requirements-ab-encoder-tested.txt`，运行 `python prepare_encoders.py weights`。若已有权重，可用 `python prepare_encoders.py weights --check-only` 核对版本和文件摘要。从此目录运行时，设置下面四个变量。Task1 的表格基准可离线独立运行。[登记的权重版本](verification/weights.json)

```sh
export SUPERAGENT_MINILM_WEIGHTS="$PWD/weights/all-MiniLM-L6-v2"
export SUPERAGENT_DEEP_MINILM_WEIGHTS="$PWD/weights/all-MiniLM-L12-v2"
export SUPERAGENT_E5_SMALL_WEIGHTS="$PWD/weights/e5-small-v2"
export SUPERAGENT_BGE_SMALL_WEIGHTS="$PWD/weights/bge-small-en-v1.5"
```

同一个入口提供 `--task-analysis flash`，可启用校园 DeepSeek 开题分析。默认模式自行根据训练反馈做决策，上表的两份成绩均来自默认模式。开启付费功能时请按团队同一任务累计账本配置。[开题分析使用说明](https://github.com/limukun6/python_withai_superagent/blob/f03323574614e8876cde5078c7aa31f093b62a36/docs/TASK_OPENING_V1_1126_zh.md)
