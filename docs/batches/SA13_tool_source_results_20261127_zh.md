# SA13：真实字段工具→后续提案→训练→导出，正式入口通过

本批跨11月26–27日执行，基线a5bd0b9；原SA12失败保留。SA13计划文件沿用开始时的20261126日期，结果按完成日期登记。A只改接入与输出契约，不改B候选/调度，不手选算法。

## 瓶颈 → 结构 → 实验 → 证据

| 瓶颈 | 新结构 | 实验 | 支持的结论 |
|---|---|---|---|
| LLM懂得要查字段，却因省略角色查错物理包 | 请求无source时按已公布member的唯一包位置解析；显式source不覆盖，两个包同名则要求澄清 | 6项新检查；原SA12真实失败动作本地回放成功 | 可修复唯一位置的接口遗漏，不需要文件名特例；回放本身不是LLM续答 |
| 长文字提示里漏掉input_id必填field | 每次语义请求携带结构化输出列schema，明确field/class_label/start等条件必填字段 | schema实际进入模型payload；程序不补ID或目标含义 | 输出格式由LLM填写并经运行时校验，减少依赖散落文字说明 |
| 只有研究注入/缓存通过，正式新任务是否能实时读题不明 | SA12生产账户/网络配置＋SA13真实工具＋原训练/预测入口 | standalone solve，stdin关闭，无client/resolver/分类器/oracle注入 | 当前这一条开发任务可从正式命令实时解释到完整重训并独立进程预测 |

完整相关回归**229/229通过、10.547秒、0fit/0API**。新增6项，其余223含SA12账本/网络、SA11缓存、原A/B集成检查；两条旧Dummy.fit夹具仍显式排除。

## 真实运行发生了什么

1. Flash自然提出inspect_header，member=test.csv，省略source。程序只检查包成员列表，发现它只在外部预测包中，记录unique_package_membership。
2. 工具只返回字段名，含record_id和30个特征，不给行值或答案。
3. 第二次真实Flash请求确实包含该工具观测。模型返回semantic-output-v1，input_id.field=record_id、正类概率class_label=1，并引用原文档。
4. B本地控制器自主选择HistGradientBoosting，实际2次分类器拟合含最终重训。根solve与训练子进程分离，输出60行，原ID/顺序/表头/概率列检查全部通过；状态completed/refit=true。
5. 独立审计从已封存artifact重编译，对任务目标、指标、特征、类别/正类、输出语义和预测形状等11项核对全部通过；usage请求ID和真实账本两行逐条匹配。

**没有人工指定工具请求、ID字段或算法，也没有替换真实模型client。** schema和唯一来源解析都是通用规则，不含V1/test.csv/record_id等案例分支。

## 资源与比较边界

| 批次 | 完整交付 | 实际fit | 真实Flash | 课程费用 | 秒 |
|---|---:|---:|---:|---:|---:|
| SA12原开发尝试，保留partial | 0/1 | 0 | 2 | ¥0.011677 | 60.625 |
| SA13修改后同开发输入 | 1/1 | 2 | 2 | ¥0.008981 | 52.375 |

SA13输入3757、输出1306token。两批合计2fit/4API/¥0.020658。它们是开发前后观察，包含两处修复且LLM输出有随机性，**不是随机配对消融，也不能据此声称稳定提高成功率或节省某个百分比费用**。未打开开发留出标签/重新评分，不能声称分类准确率提升。真实工具自然激活这次有证据，SA7原Q2仍只是语义修复，不能回填旧结论。

账本473→475，已知课程累计¥8.058840→¥8.067821；16条历史未知原行SHA不变，待处理0、新未知0。旧费用仍null，学校实际扣费/完整团队账本未核验。本自动推进窗口SA7–SA13累计18fit、14API、¥0.101426；逐批资源不互相覆盖。

## 部署配置和复跑

正式configured_flash支持既有三项SUPERAGENT_PLANNER_KEY_FILE / SUPERAGENT_PLANNER_LEDGER / SUPERAGENT_TASK_BUDGET_ID。保留strict默认；显式接受旧未知时额外提供SUPERAGENT_ACCEPTED_HISTORY_FILE及SUPERAGENT_ACCEPTED_HISTORY_SHA256，两项缺一拒绝，snapshot.allowed_scope必须匹配任务ID。每次预约直接在真实账本事务中校验，scope累计次数/估计停止阈值不能通过重建client重置。

SUPERAGENT_CAMPUS_DIRECT_IP=10.121.10.250选择当前client的校园HTTPS地址；仍用原域名证书、无系统/全局socket修改，无失败重试。正式solve子进程继承配置。不要直接复跑已尝试scope；本批runner会拒绝重复。

- 计划：docs/batches/SA13_tool_source_plan_20261126_zh.md。
- 离线runner/证据：scripts/run_sa13_tool_source_checks.py；research/sa13_tool_source_checks_01。
- 实时runner/证据：scripts/run_sa13_live_entry.py；research/sa13_live_entry_01（包含实际tool trace、usage和真实账本增量）。
- 旧动作本地回放：research/sa13_tool_replay_01，0新API。
- 独立审计：scripts/audit_sa13_live_entry.py；research/sa13_live_audit_01，0新fit/API。
- 私有模型/日志/快照：runs/sa13_live_entry_01，不上传Git。

开发继续，未冻结。下一优先仍是无manifest图像/数组的全局语义入口，以及独立新任务验收；当前只覆盖一个已使用开发输入，不保证任意未知任务成功。
