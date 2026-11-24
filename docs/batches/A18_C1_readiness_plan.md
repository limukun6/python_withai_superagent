# A-18：五开发任务C1公平配置与缺口清单（2026-11-21）

本批0训练/0API，只准备配置，不打开H1/H2或正式测试成绩。目标是五真实开发任务×六组的可审查矩阵；不因账本未知停掉本地准备，也不把生成命令当完成实验。

任务为类别表格Released Task1的原训练部分、偏好文本对Released Task2的原训练部分、已封存DryBean/SMS、照片开发Olivetti。前两者只用load_training_task、禁止打开原validation/test。后面三者复用已有train.zip文件摘要与外层划分。Task2若缺显式标签左右含义，必须记录缺口，不靠任务名猜方向。任何输入/目录/目标声明缺口不允许正式执行。

每任务六臂L0/L1/R0/E0/R1/E1均同一个full目录、fit_limit11、请求上限0或3，缓存off、seed707，300秒/worker60秒；模型/真实API时限按后续预登记实测更新。生成命令、候选摘要/配置、实际分区行号/指纹、任务契约、依赖环境、源指纹、blocked理由。R/E仅改变派生反馈，固定/动态仅改变调度；--catalog-scope固定full。未知账本维持blocked真实请求，不创建账本。

此为开发C1就绪清单，不是D5最终冻结。CLI/候选或数据修改后须重建源摘要。A17照片Q尚在运行时不改生产源码，只做独立配置准备。runner scripts/prepare_a18_c1_readiness.py，completion标prepared_with_gaps或ready_local_only，永不把状态写成已完成C1。B负责的子矩阵/真实请求/最终版本验证在后续接续。

准备审阅修正：费用scope沿用A2既有task1/task2/dry_bean/sms_spam，照片新增olivetti_faces，不按开发批次新建scope。旧的*-dev命令准备尚无真实调用，作废，最终矩阵新目录a18_c1_readiness_final重新保存；A16默认SMS scope同步sms_spam。这是预算身份修复，不是费用对账完成。

最终生产代码补充声明来源审计与否定/疑问句保守拒绝后，重建源摘要；最终可用目录为a18_c1_readiness_current。此前矩阵仅保留历史完成记录，不能执行旧源摘要命令冒称当前C1冻结。
