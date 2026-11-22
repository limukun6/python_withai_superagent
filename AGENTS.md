# A/B 共同开发约定

- 当前状态只维护在 [docs/CURRENT_AGENT_STATE.md](docs/CURRENT_AGENT_STATE.md)，每次接手先读该文件，再核对实际completion/fit日志/真ledger；历史状态在EXPERIMENT_PROGRESS，不能把旧运行快照当当前指令或重复计费。
- 稳定约定保留如下；AGENTS不再滚动叠加各批次进度。清理前原文完整保存在research/a92_continuation_state/project_conventions_before_cleanup.txt，仅作历史审计。

- 开始前读 docs/AB_INTEGRATION_V1.md、docs/EXPERIMENT_PROGRESS.md 和 docs/nine_day_two_track_plan_20261120_zh.md。
- A负责统一入口、任务/证据契约和模型导出；B负责方法目录与Scheduler。修改共用控制循环前明确文件归属，用独立checkout与Git提交交接。
- 用户2026-11-21明确授权A补对称文本方法；A26已在symmetric_text及方法注册/语义守卫完成，不再等B重复该基线。H1已有4合法候选，但H原包/模型/成绩仍封存；交换性质与合成验收不代表真实迁移效果。B继续方法效果与调度，不覆盖新模块。
- 每个实验小批次先保存计划、可复跑代码和预算，再执行；输出protocol/results/completion，不覆盖旧运行。每次向用户报告实际进度、资源与证据边界。
- 本地与Flash使用同一可执行候选全集；每路径上限11次实际分类器拟合（含refit）、3次Flash；多组件/OOF内部拟合必须计数。大批前先做匿名小任务完整接入验收。
- 真实API从忽略的api_key.txt读取密钥，不写进代码、日志、提交或输出。请求失败不自动重试；费用未知保持unknown/null，不创建空账本掩盖历史。
- 2026-11-21用户明确“我没法调出学校平台的记录，不用等，继续推进”。11条历史未知保持unknown，不再等待人工平台记录才做研发。默认strict仍拒绝未知；显式--history-policy incremental-research可用已导入真实非空历史账本继续小额研究，结果必须标记课程总预算未核验、不能作为预算合规正式结果。任何新增unknown/pending立即停止，不自动重试；不得使用模拟/空账本代替真实历史。
- llm-budget-rmb是按估计预留和实际usage结算的停止阈值，不是保证平台最终扣费的硬上界。旧历史无可靠上界。DNS未发出请求的例外只能凭原请求绑定的传输证据追加审计，不可把一般超时或无usage记作零。
- 数据包、模型、训练缓存、credential、SQLite不提交。只提交可审查的代码与脱敏协议/证据。
- 正式测试留到冻结版本后的最终评价；任务身份不作算法路由规则。受限合成/故障注入测试与真实性能结论分别报告。

- 执行沙箱会拒绝校园网络（DNS/10013），实际Python解析还可能得到198.18.*代理虚拟IP。已核验的显式进程包装器scripts.campus_direct仅把官方域名解析到AGENTS既有10.121.10.250，HTTPS Host/SNI/证书不变，不改系统DNS/代理，不自动失败切换。真实任务必须用获准访问网络的执行权限，并按批协议显式启用，不能在沙箱中误报校园网断了。模型列表探测a50_models_direct已HTTP200/19模型；原失败请求不能自动重试。
