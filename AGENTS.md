# A/B 共同开发约定

- 最新A67–A72本轮完成，无训练/API进程。A累计1214拟合尝试（含A67回溯15和A70超时1）；A69 B1四任务两内部seed16路径168fit/879.761秒，0涨7平1负；A70真实24回答成功451.177秒/¥.165155，原60秒评价停，A71复用冻结回答统一180秒下20fit/101.039秒完成12状态动作比较，0涨10平2负。原A70失败保留，不能重复ask。A72修review引用范围，旧结果保留原source；当前40相关检查及48开发例通过。先读docs/batches/A70_A72_feedback_and_review_results_zh.md、D3_ACCEPTANCE_CHECKLIST_20261122_zh.md。剩独立48例和24语义标注见D3_REMAINING_INDEPENDENT_REVIEW_TO_B_zh.md。真ledger299/known6.1417335/旧11unknown/pending0；H/test未开。

- 最新接手读docs/CURRENT_D3_A67_A70.md：A67/A68比较scope修复及开发48例完成，A69四任务B1与A70四任务A2队列已启动，先查actual completion，禁止改生产/活动runner，不能重复计费。A69前累计1025（含回溯15 Dummy fit+A68实测3），不再用下方1007当现值。A67独立B侧48仍缺，H/test仍封存。

- 最新A66已补A1八格实际训练输入探针：8合法fit/8非法fit前拒绝，26.478秒、0API，A累计1007；无运行任务。docs/batches/A66_semantic_training_matrix_results_zh.md及D3验收清单已更新。本项只证明合成worker语义性质，不是完整自主搜索/迁移；剩余96案例/独立封存、A2语义标注与范围、B1矩阵仍待补或明确修订。

- 最新A64/A65补D3审计完成，0fit/0API，累计仍999，无运行任务。D3尚未全部验收：先读docs/D3_ACCEPTANCE_CHECKLIST_20261122_zh.md与docs/batches/A64_A65_results_zh.md。远端fetch无队友新提交（d750c7b/main a38e1b8），本机bundle a787a28已接。用户已授权继续补D3；1000fit为旧计划建议非权限硬限制，后续先登记有限补充预算并保留正式实验资源。

- 最新A60–A63已完成：26+3=29新增实际fit、0API，A累计999，当前无训练/API任务运行。统一finalists与AUC正式入口/概率证据/partial交付已接，详细见docs/batches/A60_A63_results_zh.md、A_TO_B_AUC_FINALISTS_20261122.md。先按D4_freeze_and_formal_budget_20261122_zh.md整合最终目录与预算，不重复开发轮训练、不将下方970/运行中旧快照当现状。H/test仍封存。

- 最新收尾：A50/A51/A55/A58真实队列及A59只读分析全部完成，没有训练/API任务在运行。优先读docs/batches/A37_A59_round_results_zh.md与docs/NEXT_RESEARCH_AFTER_A58_zh.md。下面“队列当前运行”是历史快照，已失效。A累计970次实际fit；共享真账本275行、已知估计¥5.9765785、历史未知11、pending0。H1/H2与正式test仍封存；不要重复启动已完成批次。

- 最新接手先读docs/CURRENT_RUN_20261122_A50.md及EXPERIMENT_PROGRESS顶部。原三小时窗口因00:21待机而中断，A32为8完整/1部分/11未启动，不是完整矩阵；旧completion不改。恢复窗口另记08:42–10:42，A50/A51真实Flash队列当前运行，禁止改superagent、campus_direct.py、run_a50_facts_calibration.py、run_a29_choice_matrix.py、run_a34_transfer_coverage.py及worker观察器，先查实际completion，别重复启动/计费。
- 执行沙箱会拒绝校园网络（DNS/10013），实际Python解析还可能得到198.18.*代理虚拟IP。已核验的显式进程包装器scripts.campus_direct仅把官方域名解析到AGENTS既有10.121.10.250，HTTPS Host/SNI/证书不变，不改系统DNS/代理，不自动失败切换。真实任务必须用获准访问网络的执行权限，并按批协议显式启用，不能在沙箱中误报校园网断了。模型列表探测a50_models_direct已HTTP200/19模型；原失败请求不能自动重试。

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
