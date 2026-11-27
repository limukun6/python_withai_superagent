# SA25：PR3与A最新恢复/设计结构整合

冻结起点A=7e9d142066be7f72fa124a0dde37b068af1ef37e，B/PR3=4d4222aeb9f59f30b1596eca6e02709bb15a8072。独立工作区runs/pr3_joint_1127，分支codex/a-pr3-joint-1127。A负责本批统一入口、导出/预测恢复和缓存交接；B方法/调度按PR提供版本保留，不另改排序规则。

## 顺序

1. 保存计划及可复跑检查脚本，merge --no-commit固定B提交，逐处解冲突；保留B首次/最终交付及规范family ID、方法限制绑定，同时保留A预测适配、共享scope、独立凭据、JSON修复和native-v3设计元数据。原A/B分支保持。
2. 先0fit/0校园API检查：A SA24相关360项、B限制/专门模型/资产/反馈相关检查，补组合路径检查。禁止通过worker训练或socket发请求；明确排除原来两个直接DummyClassifier.fit夹具和B的test_formal_solve_obeys_the_rule_and_reloads实际训练夹具（正式小验收覆盖同类行为），不把未跑算通过。每次失败独立目录保留。
3. 冻结合并源码、输入和oracle后，小规模正式入口验收：普通完整solve、自定义输出solve、晚到输入预测恢复、Windows训练中断。尽量复用公开开发数据和已经核验的模型/原始LLM解释；模拟client仅用于组合控制路径检查，不能称真实LLM成功。解码/输出/模型导出必须实际执行。
4. 正式小验收上限6次实际分类器拟合（完整路径各2，含refit；中断路径最多2），0新增校园API，预计机器3–8分钟。晚到输入恢复复用既有模型，0新增fit；现有真实回复可显式迁移，不向新scope伪记旧用量。若需要额外真实Flash，另立后续小批，不在本批临时扩大。
5. 核对原模型/目标/指标/特征/输出含义保持、独立进程重放、partial与completed区分、首份输出未被失败覆盖、共享账本无新行。Windows中断须核对创建的进程及其worker退出；不能引用B的POSIX结果当Windows结果。
6. 完成报告与当前状态，合并提交并推送独立整合分支；不在GitHub替用户合并PR #3，不开课程正式测试/H2，不重跑B166次已有实验。

## 验收边界

这批解决组合兼容，不估计模型质量提升或任意未知任务成功率。明确manifest或复用的真实解释只说明这一段接入来源，不能说全部由新LLM从零读题。自然模型失败的真实恢复已有SA23，不以本地模拟再次充当付费证据。失败动作应先恢复再考虑已标partial的先验保底；硬盘/模型损坏可保留已有交付。文件名/任意原生-记录互转等能力另批研发。

真实A canonical预期526条、17unknown、0pending，已知课程¥8.428396。B有尚未拿到原SQLite行的待导入请求，保持单独缺口记录；本批不用官网对账，不改unknown/null，不新建空真实账本。

复跑：scripts/run_sa25_joint_checks.py，及在首次正式验收前保存的scripts/run_sa25_joint_delivery.py。检查、正式验收、审计各用新research目录、protocol/results/completion。私有任务包/模型/SQLite/密钥不入Git。

## 首轮实跑后的续跑登记

delivery_01普通路径150秒超时，只交付先验partial，classifier账本已知fit0但有1次ledger_unavailable的worker尝试；原结果保留，不能把未计量worker直接称已证实0fit。其余三条未跑。先用零fit的Windows管道最小诊断区分继承控制stdin与DEVNULL，再修复worker stdin隔离，重做零fit检查，另建prepared_02/delivery_02，不重复旧输出。两轮合计拟合预算仍≤6（首轮未计量尝试预留1次上界，第二轮完整路径4fit，中断夹具在worker模块执行前暂停）。没有HTTP失败/重试或额外API授权需要。

诊断probe_01的worker_started以子串判定，会误命中TimeoutExpired里打印的命令文本；原日志保留，不以该布尔值证明worker启动。probe_02改按独立输出整行核验，脚本/结果分别记录；诊断两个模式都不训练。
