# 正式六组运行入口与接续（2026-11-21）

A26–A27最新：A已按用户明确授权补齐4个交换不变词法双文本候选，H1从0到4，不再等B补同一基线；H仍未训练评分。正式匿名CLI5fit、24条搬迁/交换预测一致、概率交换差0。真实Flash另试1次仍DNS失败，已停止；ledger现在172行（170历史+2未发出尝试）、11历史未知。先按CAMPUS_FLASH_A27_FAILURE_AND_TEST_20261121.md做只读网络检查与单次连通，不重复失败批次。完整结果batches/A26_A27_results.md。

A20–A25更新：五任务L0/L1十路径开发对照已完成（102fits/478.379秒），五对同预测、动态少8fit；并非最终冻结六组C1。结果batches/A20_A25_results.md。旧偏好CSV预测契约已补。用户明确不等待平台账单；新增显式研究策略见下文，原11历史未知保持。真实Flash尝试在首请求DNS解析失败，0模型服务请求发出；只读校园IP443也超时，没有重试。当前真正缺口是运行环境学校网络可达性。

A17–A19更新：新增--catalog-scope baseline用于同版Q目录对照，六组C1始终使用full（默认）。照片Q两组均97.5%，不宣称分数提升。保守README声明读取已恢复课程Task1指标和Task2偏好语义；不改B算法。最新五任务30配置位于D:/codex/python_withai/runs/a18_c1_readiness_current，尚未执行C1。费用任务ID沿用task1/task2/dry_bean/sms_spam，照片olivetti_faces；A16默认sms_spam。实际历史11未知仍未对账。

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

Flash在上例选择raw-history/checked-history，并提供--api-key-file、--budget-ledger、--task-budget-id，默认校园Flash/SSE、3请求上限。默认strict对项目任何unknown/pending仍在读取key/请求前拒绝，空/new_project账本也拒绝。显式incremental-research只允许导入历史unknown继续，新增unknown/pending依然停；保留完整费用null和课程预算未核验标记。不要把模拟用量当真实账本。请求失败终止、无重试；到达3次上限后额外探索走预登记本地规则，不是失败兜底，日志planner_routing记录来源。

真实小试scripts/run_a16_live_flash_pilot.py默认只校验，不新建账本。A23已有真实连接尝试但无模型响应；原始失败目录不覆盖。原始A16计划后续修订见batches/A23_incremental_flash_research_plan.md与A20_A25_results.md。本机真实历史账本已建于D:/codex/python_withai/runs/a23_real_local_history/ledger.sqlite；170历史+1未发出尝试、11未知，不能重新导入或新建账本清掉历史。

后续学校网络恢复后，先用新目录校验（0训练/请求），通过再按当批预登记明确加--execute。以下不是后台作业，没有自动重试：

```powershell
Set-Location "D:\codex\python_withai\runs\integration"
& "D:\codex\python_withai\.venv\Scripts\python.exe" -m scripts.run_a16_live_flash_pilot --prepared "D:\codex\python_withai\runs\a07_prepared" --output "D:\codex\python_withai\runs\flash_next_preflight" --ledger "D:\codex\python_withai\runs\a23_real_local_history\ledger.sqlite" --api-key-file "D:\codex\python_withai\api_key.txt" --history-policy incremental-research
```

执行必须另选从未存在的output目录。两臂合计最多22fit/6请求；每臂0.30元、合计0.60元是估计预留/usage结算停止阈值，不是扣费硬上界。无可靠历史总上界，`valid_for_budgeted_final_report=false`，不能冒充课程预算已核验。转交另一机器要同时交付实际累计账本和所需准备数据（私下传递，不进Git）；不要用仓库内原始history_audit重新建账本忽略后续调用。

## 本机已准备的数据

- 照片开发：D:/codex/python_withai/runs/a13_data_registration_network/photo_dev/train.zip；真实照片开发路径D:/codex/python_withai/runs/a13_photo_pilot，9fits/78/80，已作为开发任务。
- H1：D:/codex/python_withai/runs/a13_holdout_packages/H1_task.zip，2000行PAWS同义判定，equivalent标签；A26已有4合法对称词法候选，尚未训练/评分，仍需统一冻结后按H协议启动。
- H2：D:/codex/python_withai/runs/a13_holdout_packages/H2_task.zip，120张Rice照片；8个初始合法候选；未训练/未评分，保持封存。
- H两份实际61411/61421划分登记在a13_holdout_packages/registration.json及来源注册文件。部署策略冻结前不运行H成绩，不改名为开发后又当未见。

这些数据/模型只在本机，不随Git克隆。另机按预登记prepare脚本取得同摘要数据，再核对实际行ID；不得仅凭相同seed宣称公平。

## 给B的最小接续

取包含本文的最新codex/expanded-feedback提交。A维护ab_run.py/ab_flash.py/schema.py/cli.py；用户授权A补方法后，symmetric_text.py和注册/语义守卫已更新，B接续前先读A_TO_B_A26_SYMMETRIC_20261121.md，避免覆盖或重复实现。B继续方法效果与调度实验；未来更强对称候选仍须真实交换不变、每次fit声明准确，通过worker守卫，不能仅加flag。H1数据格式无需重写。

当前六组接入完成、模拟HTTP流程验证通过，五任务本地开发对照完成；完整六组真实Flash效果尚未完成。接续优先在可达学校网络执行新小试、接B对称方法与剩余Q，再按总计划冻结Q/C1矩阵。历史未知继续披露，不等待平台记录才做研究。真实H仍按D5冻结后的D6窗口评价。
