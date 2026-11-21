# 正式六组运行入口与接续（2026-11-21）

工作目录必须为D:/codex/python_withai/runs/integration；解释器D:/codex/python_withai/.venv/Scripts/python.exe。当前分支codex/expanded-feedback。模式由两参数组合，不需要研究脚本手工挑下一算法。

| 计划组 | planner | scheduler | 真实请求上限 |
|---|---|---|---:|
| L0 | local | ab-fixed | 0 |
| L1 | local | ab-v1 | 0 |
| R0 | raw-history | ab-fixed | 3 |
| E0 | checked-history | ab-fixed | 3 |
| R1 | raw-history | ab-v1 | 3 |
| E1 | checked-history | ab-v1 | 3 |

六组共同--portfolio ab-v1 --fit-limit 11，不传--trials或--policy。固定为4搜索+前3各2复核+1refit，动态在同上限内决定预算去向；本地和Flash共享候选目录。所有结果都执行基本正确性核验，E开关只改变发给规划模型的派生checks/memory，不能关闭数据隔离或基本合法性。

本地示例（新目录）：

```powershell
Set-Location "D:\codex\python_withai\runs\integration"
& "D:\codex\python_withai\.venv\Scripts\python.exe" -m superagent run "TASK.zip" --planner local --scheduler ab-v1 --portfolio ab-v1 --fit-limit 11 --run-dir "NEW_RUN"
```

Flash在上例选择raw-history/checked-history，并提供--api-key-file、--budget-ledger、--task-budget-id，默认校园Flash/SSE、3请求上限；当前全项目任何unknown/pending都会在读取key/请求前拒绝，空/new_project账本也拒绝。真实11条历史未知尚未对账；不要把research归档里的模拟用量当真实账本。请求失败终止、无重试；到达3次上限后额外探索走预登记本地规则，不是失败兜底，日志planner_routing记录来源。

真实小试已有scripts/run_a16_live_flash_pilot.py和batches/A16_live_flash_pilot_plan.md。默认只校验；实际账本恢复后再明确--execute。该runner不会新建账本；目前只有blocked preflight反例，无真实运行成绩。

## 本机已准备的数据

- 照片开发：D:/codex/python_withai/runs/a13_data_registration_network/photo_dev/train.zip；真实照片开发路径D:/codex/python_withai/runs/a13_photo_pilot，9fits/78/80，已作为开发任务。
- H1：D:/codex/python_withai/runs/a13_holdout_packages/H1_task.zip，2000行PAWS同义判定，equivalent标签；输入解析可用，B合法方法0，不能启动性能实验。
- H2：D:/codex/python_withai/runs/a13_holdout_packages/H2_task.zip，120张Rice照片；8个初始合法候选；未训练/未评分，保持封存。
- H两份实际61411/61421划分登记在a13_holdout_packages/registration.json及来源注册文件。部署策略冻结前不运行H成绩，不改名为开发后又当未见。

这些数据/模型只在本机，不随Git克隆。另机按预登记prepare脚本取得同摘要数据，再核对实际行ID；不得仅凭相同seed宣称公平。

## 给B的最小接续

取包含本文的最新codex/expanded-feedback提交。A维护ab_run.py/ab_flash.py/schema.py/cli.py，B继续ab_catalog/ab_methods/ab_scheduler。B需要给H1加入真正交换不变的表征/分类器，明确candidate_swap_rule=preserves、能力/参数/拟合数；同时通过ab_catalog当前对symmetric的过滤和worker语义检查。不能只加元数据flag或复用翻标签增强。H1任务包已可直接加载，A不用再重写数据格式。

当前六组接入完成、模拟HTTP流程验证通过，本地有真实任务证据；完整六组真实Flash效果尚未完成。下一研发日先核验真实累计用量及B对称方法，再按总计划冻结Q/C1矩阵。真实H仍按D5冻结后的D6窗口评价。
