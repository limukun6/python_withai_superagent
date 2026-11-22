# A89：A负责的Coupon六组完整流程对照

A88同源Q完成43fit，A起点1290。按九天分工A推进Released Task1/Coupon，B对应Task2。沿用冻结整合源/独立依赖/全15候选、A47训练包、实际split/seed707与A88的300秒trial/1200秒path；不打开课程test/H。六组L0(local固定)、L1(local动态)、R0(raw固定)、E0(checked固定)、R1(raw动态)、E1(checked动态)。只比较既有结构，不开发新调度/提示，不加入第七组。

预登记复用A88 coupon_full作为L1：已完成11fit的源码、菜单、环境、任务/实际分区、默认开关、11fit/300/1200预算均逐项核对，原模型/预测/结果不改。这是开发留出已见后的预先登记复用，不称新独立抽样或首次盲测。其余5路径新训练最多55fit，最多12次新Flash（4组×3），总逻辑6路径最多66fit但新资源只计实耗。运行顺序L0→E0→R1→R0→E1，L1引用现有记录。

Flash统一DeepSeek-V4.1-Flash、temperature0、timeout60、calibrated-v2，task-profile/候选覆盖/local-text-proposals均off，raw/checked共享同一合法菜单/预算。固定与动态只控制探索/复核分配，完整效果是端到端对照，不冒称首次提议差异单独由反馈导致。所有paid请求都经原唯一ledger与原task_budget_id，incremental-research；旧11unknown保留，总学校预算未核验、不得写预算合规正式结论。每条付费路径增量预留¥0.30，四条总估计阈值¥1.20（不是平台扣费硬保证）。任何新增unknown/pending或请求/训练失败立即停整批，不重试、不换模型、不回退猜动作。

复用L1在本批前已评分，这一点必须公开。其余5模型全封存后，再统一对同一已见开发val评分；不据中间val结果改后续路径。记录实际fit/时间/请求/tokens/已知估价及失败，复用与新测分开。CLI唯一入口，自主选候选，不人工接管。学校调用仅显式进程campus_direct绑定官方内网IP，TLS主机/证书保持，公开下载代理不用于学校API。

预计机器15–30分钟（5×1200秒名义上界100分钟+进程余量），额度不足先读completion/checkpoint/真ledger；不重复启动同目录。计划和runner先提交推送。完成后只报告本任务六组，不称跨任务C1或H已完成；相同源码的B对照可共同冻结后汇总。
