# SA15：默认fallback接管与真实回复跨reader重验证

基线e3aaee4。用户要求自动下一步。A维护接入/证据/交付，不改B方法和排名；本批不再重复SA14的proactive实验。

## 问题与实现

1. 默认fallback中，只要记录读取器看见.npy/.npz就抛NativeContract，程序把它当“权威契约拥有者”，即使根本没有manifest也直接停止。调整为：显式manifest仍阻断覆盖；无manifest且已配置、未阻断的LLM有剩余预算时，交给全局解释。无client/已失败client不新增请求。本地可读任务保持先本地、无需LLM。接管写入日志。
2. 显式迁移工具只支持旧标量/输出提案，不能重验证SA14真实native提案。增加native编译分支；在旧artifact存在时核对完整性、原任务/预测包字节及proposal，不绕过当前reader检查。原API用量绑定到语义接受之前对应的request/response，不能误取后续B规划调用；匹配canonical已结算的model/token/费用。
3. reader变化仍让旧artifact失效；必须显式迁移到新文件，不能修改旧缓存或把旧回复标成新API。迁移不恢复搜索历史/模型训练。

## 预登记实验

- 零fit/API：真实默认NativeContract分支的无manifest数组接管；显式manifest、无client、blocked/耗尽、已有本地成功不调用；预fit检查点验证正式入口；native迁移、旧包/回复/usage篡改拒绝及后续LLM调用不串用。加既有245项检查，先落盘scripts/run_sa15_default_checks.py。
- 迁移SA14两条原真实提案到当前reader，新进程复用两已有模型predict，字节与原54行一致；0新fit/API，同真实账本SHA不变。不把迁移当新真实LLM回答。
- 离线全部通过后，预先准备一个公开开发NPY封装（由SA14 Iris训练/预测特征重封装，非独立任务迁移）。文件名非X_train等约定，触发真实NativeContract阻断点。原始文档描述目标/指标/输出，不提供manifest。
- 从standalone solve默认参数启动，不加--task-interpretation或--task-analysis；最多2次实际fit含refit、3真实Flash、¥.25估计停止阈值、240秒。此路径也保留默认B的Flash决策，所有调用共用3次上限，逐角色报告，不归为纯读题收益。无client/resolver/算法注入，不读预测答案，不新评分。
- 起始账本477/known8.083541/unknown16/pending0，保留原行/null。新unknown/pending停付费，不失败重试、不循环到成功。每次尝试scope唯一；新目录保留失败。

预计实现与离线20–45分钟，真实小试1–4分钟。结果只证明默认路径的具体接管及重验证能力，不保证任意未知任务成功。协议/结果/completion落盘后提交可审查代码和脱敏证据；数据/模型/凭据留runs。
