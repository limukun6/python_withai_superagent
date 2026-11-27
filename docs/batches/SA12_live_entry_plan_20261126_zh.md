# SA12：正式新任务实时读题入口

基线2304403。先写本计划与可复跑runner，再执行。归属A：账本配置、HTTP传输和正式入口；不改B方法目录/排名。用户已授权保留既有16条未知继续研究，并要求自动下一步；本批不再索要同一授权。

## 瓶颈与结构

研究runner能用Flash，正式新任务solve却仍strict，且研究进程的临时DNS选择不会继承到训练子进程。加入显式部署配置：已接受未知快照路径+SHA，逐请求原行SHA/真实账本/任务scope绑定；账户每次reservation事务内直接核对，不在生产使用unittest.mock全局替换。strict默认不变，新unknown/pending仍阻断。快照的请求次数和估计停止阈值按scope累计，不因新建client而重置。

显式SUPERAGENT_CAMPUS_DIRECT_IP=10.121.10.250仅作用于该client的校园HTTPS连接。保留原域名、TLS/SNI/证书校验，禁代理/重定向/自动重试；不改系统DNS/代理/防火墙，不全局改socket。正式solve训练子进程继承配置。

## 先验收再实跑

1. 新增零API/零fit检查：strict拒绝未知、快照哈希/路径/scope/行变动、新unknown/pending、额度/次数跨client累计、非空导入历史；精确连接地址与TLS域名保留、失败不重试、配置完整性；187项相关回归继续。
2. 若全部通过，另登记正式CLI live协议并核对真实账本471/未知16/待处理0和原快照。选已准备真实数值开发任务（不是独立未知泛化测试），从正式solve启动新读题，禁用缓存和任何resolver/分类器注入，B本地开题+protected review。
3. live最多3次Flash、2次实际fit含refit、估计停止阈值¥0.25、墙钟240秒；不打开外层答案或比较分数。新失败不自动重试；出现新unknown/pending停止付费，保留全部证据继续离线。
4. 完整验收要有新鲜语义契约/artifact、原ID/表头/行序正确预测、实际请求及canonical账本结算。调用失败不能改记零费用；只能报告实际边界。

零API runner：scripts/run_sa12_live_entry_checks.py。live runner：scripts/run_sa12_live_entry.py（先落盘协议，再付费）。结果输出独立research/sa12_*，模型/数据/账本/密钥仍在忽略目录。
