# A-23：保留未知历史、限额继续真实Flash研究（2026-11-21）

用户明确无法取得学校平台记录且要求“不用等，继续推进”。本批按此更新执行策略，不再把人工平台对账设为所有研发的前置条件。历史费用仍unknown，完整课程预算仍未核验，不将研究模式称为已核验正式实验。

## 历史与新增分开记账

从本机真实benchmarks/agent_architecture/v2_pilot/sse_live_validation及local_followup源日志重建非空历史audit。按请求事件指纹去重；无request事件的明确nonstream_comparison用原usage事件身份保留。所有已知token按现有冻结价格重算并保存原计费字段来源，11未知保持null；不改原日志，不引入任何A12模拟记录。导入同一持久SQLite，不能创建空账本重置费用。

默认strict入口保持拒绝未知；新增显式--history-policy incremental-research，仅允许继续历史导入的unknown（historical: ID）。任何新增失败unknown或pending均立即挡住下一请求，防止失败后不断累计。known task+shared依然不得超20；新增每臂0.30元、最多3请求、11fits，整批0.60元/6请求/22fits。但unknown历史没有可靠上界，因此总费用是否≤20仍unverified，结果valid_for_budgeted_final_report=false，known新增费用与完整成本null分开报告。绝不把unknown清成0，绝不称平台已对账。

## 先离线检查，再真实小试

实现/测试在A20源冻结模型运行结束后接入正式CLI，避免破坏本地C1源身份。先验证原strict仍拒绝、历史unknown在显式研究模式保留、共享已知费计入、非历史新unknown/pending停止、来源/新增状态均落盘；这些0fit/0HTTP。

真实小试在SMS已见开发任务，动态raw-history→checked-history，两臂同full目录/707/808/909/11fit、600秒/worker60秒，最多3Flash，APItimeout60读取上限。任一请求/训练失败停止整批，不自动重试，不继续另一臂。两模型全封存后才评分；外层留出不进决策。若真实Flash只改措辞不改动作，也保留。相同状态采样和真实任务泛化仍是其他实验。

run_a16_live_flash_pilot.py增加显式history-policy，默认strict。先本机审计和无调用验证完成，执行研究模式由本次用户继续要求授权；日志/protocol明确unknown历史和课程预算未核验。原始账本/密钥/模型/SQLite不提交，仅脱敏汇总。运行前保存输入/代码/候选/划分/预算/runner摘要。
