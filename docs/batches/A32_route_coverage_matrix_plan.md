# A32：早期路线门槛的同版本完整开关对照

属于三小时窗口，用户授权连续推进。A31已增加默认off的候选投影和0API检查，A29/A30旧源结果只作开发动机。现在冻结新生产源码，不再同时改控制器。H与正式test继续封存。

每任务10组：共享本地L0/L1（无Flash、门槛off），以及R0/E0/R1/E1各off/routes。两任务20条完整路径、最多220监督fit、48真实请求。每路径11fit含refit/最多3请求；600秒路径、worker60秒、外层660秒、CPU线程2。Flash四种控制器完整注册目录相同，route门槛只限制当前可负担新候选菜单，目标min(4,全部已注册路线数)，不会增加方法、替换B调度/复核/停止，也不重问模型。off/default的prompt结构不添加额外字段。

预登记顺序：coupon L1→R0_off→R0_routes→E1_routes→L0→E1_off→E0_off→E0_routes→R1_routes→R1_off；preference E0_routes→E0_off→L0→R1_off→R1_routes→E1_off→E1_routes→R0_routes→R0_off→L1。单训练队列，不因分数选臂或改顺序。所有20模型封存后统一原指定val评分，新进程predict与重载核验；原test不可读。

重新prepare当前源码的训练包/TaskSpec/全部候选/实际707、808、909划分，核验与既有开发身份一致，但不冒充与旧源码相同。每路径门槛事件记录目标/已见路线/原eligible/projected与fallback原因；确认门槛是否真正增加覆盖，再看效果，不只看新flag存在。

每paid臂.30元、每任务2.40元、全批4.80元为估计预留/usage结算阈值，非平台扣费硬上界；仍同一最新累计ledger与规范task1/task2范围、原11未知保持、总费用未核验。启动路径前剩余三小时窗口需≥660秒，不足就停止启动、保留未完成批；所有失败停止整批、无重试。预计50–90分钟，路径最坏总上界另由600秒×20体现，时间不足不假称完成。

研究比较：同任务、同反馈/调度内routes−off隔离路线投影；再看门槛下E−R、动态−固定以及相对强本地。观察覆盖数、原始提示是否一致、不同候选/模型与逐行预测差、真实fits/时间/token费用/完成率。不能把一个开发划分当显著性或H迁移，也不能把被门槛排除的旧提案在新轨迹中的成绩凭空推算。

复用scripts/run_a29_choice_matrix.py增加--study route-coverage（默认six-arm旧研究），prepare/execute与--validate-only规则不变；--window-deadline 2026-11-22T02:24:19+08:00按剩余时间保护。原A29执行时脚本hash已保存，不将新脚本冒称旧run当时版本。分析同样0fit/API另落脚本；所有原数据、模型、payload、SQLite、密钥不入Git。
