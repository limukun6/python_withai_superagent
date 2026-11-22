# 当前三小时窗口接手

用户要求连续推进三小时，实际窗口为北京时间2026-11-21 23:24:19至2026-11-22 02:24:19。不要把先前A29的41分钟计入新窗口，不要为凑时长重复训练/请求。主助手继续推进直到窗口任务收尾；不得只留下后台任务就声称全部完成。

## 正在运行的两个进程

1. `scripts.run_a29_choice_matrix execute --study route-coverage`，输出主仓库runs/a32_coverage_matrix。20路径，每路径11fit/3请求，生产源码为45ebb5d加入候选路线门槛后的版本，候选/种子/提示/控制规则已冻结。当前工具会话86103仅供同一Codex线程继续poll，它不是Windows进程PID。
2. `scripts.run_three_hour_queue`，输出runs/three_hour_20261121_2324/queue。当前工具会话84470同样不是PID。队列等待A32 completed后串行执行已登记的A34、A35和条件A36，按剩余窗口阈值启动，任何阶段失败停后续，不重试。它会自己生成A32只读summary。

用scripts.status_a29_choice_matrix读取主批进度；completion为null不保证主进程仍活着，必要时查Python命令行。不要启动第二套队列或覆盖任何输出。主训练队列只能一个。

## 冻结文件，不能边跑边改

- superagent全部生产Python；scripts/run_a29_choice_matrix.py；scripts/a29_worker_observer/sitecustomize.py。
- 队列另校验scripts/run_a34_transfer_coverage.py、scripts/run_a35_fixed_state_study.py、scripts/summarize_a29_choice_matrix.py的SHA。改这些会让队列停止；确需修复应先保留原阶段错误、制定接续，不重复已经收费的请求。
- 可以写独立分析/图/文档、tools内原型，不修改以上文件。A37训练profile只是独立0API原型，尚未接正式Flash。

## 阶段与资源

计划batches/A31_A35_three_hour_window.md及A31/A32/A34_A36/A35各子计划。A31已有124项通过、7监督fit；原12固定提案10个会被新菜单挡住，仅回放。另24个原R/E上下文在新off路径重建一致，0fit/API。A32最多220fit/48请求；后续队列最多再48新请求。按source实际完成与usage计，不把上限当用量。

新窗口起点A窗口597fit，ledger203行/known课程计价5.3767495元/原11unknown/pending0。A31后604，之后按实际completion累计。历史40fit回溯更正不是当时的物理观察。真实ledger=主仓库runs/a23_real_local_history/ledger.sqlite，不进Git；不能重建空账本/用旧快照/并发分叉。所有计费研究仍标总课程预算未核验，平台实际扣费null。

## 已完成旧批可用事实

A29两任务六组12路径128fit/24请求/2459.935秒/¥.205921，A30再6fit/0API。具体表在batches/A29_A30_results.md，全部原源为202a250，不能与新A32源混称同版。偏好对同一checked完整messages两次返回不同首选；一个新划分动作优势未转成端到端优势，不能只看一次LLM选择。

## 独立工作与收尾

- A37 tools/a37_profile_contract.py复用既有profile，五开发任务数据事实/改名不变/raw-E共用验收通过，0fit/API；原型没有接入正式context，不能宣称收益。最新runs/a37_training_profile_probe_final。
- A38计划与runner已写，等A32和A34都completed再执行四模型搬迁预测（0fit/API/不读标签）。脚本scripts/run_a38_portability.py，来源预定R0_routes/R1_routes，不按分数挑。
- A32 summary输出research/a32_coverage_matrix由队列生成；不要提前自己生成同名目录。A34/A35/A36完成后用独立只读脚本归档JSON，模型/数据/payload/SQLite/key不提交。图像只允许统计图，不提交原照片。
- 汇报统一“瓶颈→结构→实验→证据边界”，给实际耗时、fit/API/token费用、未启动阶段与原因。原11历史未知保持；H1/H2和正式test继续封存。最后更新EXPERIMENT_PROGRESS/A_LINE_RESUME/交接并推送。
