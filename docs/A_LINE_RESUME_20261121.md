# A 线接跑入口

最新A-11：正式入口固定/动态三任务六路径对照完成；先读batches/A11_fixed_dynamic_results.md与EXPERIMENT_PROGRESS顶部。已见任务外层预测均相同，动态27vs固定33fits，定位省训练；不要重复该批追涨分。下一工作为动态Flash raw/checked回调/账本失败链，先落计划与runner、无真实请求验证。真实调用仍需核验11条历史未知账本。自然图像与合法对称文本/H资格尚未完成。

最新更新 A-09：已接收 B a787a28 并完成正式本地动态 CLI 验收。先读 A_TO_B_BRIDGE_ACCEPTED_20261121.md 与 batches/A09_B_bridge_cli_results.md。下文等待 B 的事项已完成；下一步改为真实任务小试/环境对齐，再接动态 Flash 回调。旧 A2 状态与当前源码不再同源。最新实验进度以 EXPERIMENT_PROGRESS.md 顶部为准。

先读EXPERIMENT_PROGRESS.md、batches/A_window_20261121_results.md、A_TO_B_NEXT_HANDOFF_20261121.md。代码在codex/expanded-feedback，首轮实现提交1d57fd4。主仓库main不是这轮实验源码，运行时工作目录必须是runs/integration。

## 已执行的批次

主仓库runs/a05_*、a06_*、a07_*、a08_*、a_window_*保存原始产物。research/a_window_20261121保存紧凑证据；状态prepared不代表已经调用API。完整二进制模型和数据只在本机，fresh clone不能凭归档JSON恢复模型。

继续前检查completion与protocol：source_unchanged、实际fit数、失败状态；不要覆盖已完成目录。复跑必须新目录，并在项目总资源里累计，不能将已有案例重放当新独立样本。Git可能规范化CRLF，历史源码/文件hash是运行当时原始字节；fresh clone如严格校验不匹配，应新建协议重新冻结，不篡改旧hash。

## 无需再次训练的检查

```powershell
Set-Location "D:\codex\python_withai\runs\integration"
D:\codex\python_withai\.venv\Scripts\python.exe scripts/run_a08_ab_acceptance_batch.py --output D:\codex\python_withai\runs\a08_new_check
D:\codex\python_withai\.venv\Scripts\python.exe scripts/run_a05_task_contract_batch.py --source-root . --output D:\codex\python_withai\runs\a05_new_check --cases-file research/a05_task_contract/cases.json --subset all
```

两命令均0实际fit/API；第二条是旧案例回归，不能叫未见泛化测试。

## 下一项真正需要的实验

1. B提供候选能力目录、next_action适配器与提交号后，A接scheduler_bridge到统一循环，先做匿名任务完整验收，真实fit含refit≤11、本地先跑。没有B模块时不要用fixture结论冒充。
2. 新v3状态由A2-02协议生成在runs/a2_02_v3_states，检查completion后使用。真实ask需已核验的累计任务/共享账本；目前11未知请求未对账，不直接运行ask、不新建真实空账本。检查源码hash与状态context_hash，任一变化就重新制定版本协议。
3. 有效raw/checked决策封存后，再提前写动作评价计划和runner，在共同两份训练内划分测动作；相同动作只训练一次、如实标明复用，不按外层成绩挑决策，不自动扩充重复请求。

本轮尚不能支持“真实LLM已纠错”“Agent精度提高”“B动态调度已接通”“自然图像泛化通过”。本轮已实证的是任务与结果契约检查、正常任务兼容性和跨类型模型交付。
