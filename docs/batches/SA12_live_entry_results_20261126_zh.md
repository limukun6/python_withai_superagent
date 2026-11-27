# SA12：正式付费配置接通，但完整交付首轮未通过

基线2304403。A改动已把研究runner的历史授权和校园直连接入正式configured_flash，未改B方法/调度。不能把网络成功等同于自主任务完成。

| 瓶颈 | 结构 | 证据和边界 |
|---|---|---|
| 正式入口默认strict，不能使用用户已接受的旧未知 | 显式快照路径+SHA，scope/账本/逐行SHA绑定；reservation事务内直接校验 | 默认strict仍拒绝未知；不使用生产全局mock，不建空账本，旧费用保持null |
| 重建client可能重置小批额度 | 按scope累计请求次数、已知费用+本次估计预留 | 跨client、并发、新unknown/pending、旧行改动、空账本检查通过 |
| 研究进程DNS包装不能继承到正式solve子进程 | 显式环境配置，单client HTTPS socket选择既定校园IP | TLS/SNI原域名、证书、禁代理/重定向/自动重试；不改系统或全局socket |

无fit回归223/223通过，10.266秒、0API。首轮223项有两条测试夹具错误（未建日志目录、断言了错误的urllib3异常基类），修正夹具后重跑；原失败research/sa12_live_entry_checks_01保留，最终_02。

## 正式CLI实时验证

scripts/run_sa12_live_entry.py先落盘protocol和唯一scope，从真实standalone `unified_agent solve --task-interpretation proactive --task-analysis local`启动，stdin关闭，无client/resolver/分类器/oracle注入。使用以前SA8数值开发包，不能当成独立未知迁移样本；最多3API/2fit/¥.25估计停止阈值，未打开外层答案。

实际2次真实Flash、0fit、课程¥0.011677、60.625秒。两次均HTTP200、有最终usage并正常结算，无新增未知。账本471→473，已知课程累计¥8.047163→¥8.058840，旧16未知和所有旧行SHA不变，待处理0；供应商实际扣费/全团队总额仍未核验。

**完整交付未通过**：第一次模型请求`test.csv`字段但省略source，默认按训练包查找，实际该member只在外部预测包；第二次output recipe省略input_id的field。两次提案拒绝后按现有B规则保留训练类先验，refit=false、delivery_status=partial。CLI退出0是“已有披露的部分交付”，本批验收明确要求真实训练+完整契约，因此记failed，不冒充通过。原失败不覆盖、不自动重试同scope。

下一批单独修复：省略source时对已公布成员做唯一物理来源解析（显式source仍严格遵守，重名不猜测），并把输出列条件必填字段以结构化schema给LLM。不能程序猜ID字段、补指标/语义；仍须LLM读取证据并给出完整提案，再实际校验。新的付费验证须新计划/scope并单独记账。

证据：research/sa12_live_entry_checks_01、_02、sa12_preflight_01、sa12_live_entry_01；实际模型/日志/快照在忽略的runs/sa12_live_entry_01。本批无性能提升结论、未冻结。
