# 原始历史 / 检查记忆的正式入口

这两个模式已接入同一个 `superagent run`：

- `--planner raw-history`：原始历史、实际 estimator 观测与候选元数据。
- `--planner checked-history`：相同信息，增加配置/解释/假说检查与带 trial 作用域的记忆。

两者使用 single 架构、expanded 目录、catalog anchor、refinement/allocation off。`--error-feedback` 可给两组都启用错误切片。动态探索/确认调度仍由 B 的 Scheduler 接口接入；当前入口使用原有三划分固定复核，不宣称六组整合矩阵已经全部完成。

## 本机离线验收

从主仓库执行，无真实 API，模拟 HTTP 和实际训练：

```powershell
& .\.venv\Scripts\python.exe .\runs\integration\scripts\run_a1_cli_batch.py --phase fixed --output .\runs\a1_04_new_verification
```

这将训练 17 次分类器，使用独立模拟账本。不可将模拟 token 和账本金额计入真实任务费用。

## 真实运行模板（会调用 API）

进入包含本实现的 checkout；本机当前是 `D:\codex\python_withai\runs\integration`。以下仅是入口说明，不是已执行实验。任务累计账本必须先按现有历史初始化或导入，沿用稳定 task-budget-id；不要用 `--new-project` 重置已有任务费用。

```powershell
& D:\codex\python_withai\.venv\Scripts\python.exe -m superagent run TASK.zip --planner checked-history --api-key-file D:\codex\python_withai\api_key.txt --budget-ledger EXISTING_LEDGER.sqlite --task-budget-id EXISTING_TASK_ID --trials 4 --time-limit 1200 --trial-timeout 90 --error-feedback --run-dir NEW_RUN_DIRECTORY
```

raw 对照只改变 `--planner raw-history`，并使用新的运行目录。默认采用既定校园端点和 DeepSeek-V4.1-Flash，两组同样流式读取并复用 ExperimentLLM 的账本和请求上限。真正配对实验还需冻结行 ID、候选、源码和执行顺序，不能仅运行两条命令就声称公平比较。正式 A2 需要另行保存计划。

## 成功与失败的交付语义

成功运行：`run_manifest.json` 标记实际 planner、上下文版本、最大请求数与停止策略；`run_outcome.json` 记录完成状态、拟合、调用、累计任务账本，模型通过搜索、复核、refit 后导出。

规划请求或提案失败：当前运行立即结束并抛出原始异常，不随机补方案、不自动重试；`run_outcome.json` 记录 stopped、已完成搜索和保留的 trial 模型。trial 模型是部分产物，不能冒充完整预测解决方案。API 报错后不要自动循环重启命令。

缺 usage / 模糊传输错误：完成费用为 null，任务账本仍阻止继续调用，待核对后再处理。已知 token 但 JSON 无效：照实保存已知费用，仍停止当前运行。实际平台扣款继续记为未知，由账单核对。该功能不改变既有定价来源，也不代表已验证课程价格资格。
