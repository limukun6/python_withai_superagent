# SA5：固定四案例的真实Flash语义机制探测

基线e43c31d。先保存案例构造器、明确oracle、无fit验收、付费runner，再运行。案例由本轮开发者预先写定，属于开发机制探测，**不称独立人类盲测或真实任务迁移**。付费回复封存前不按结果修改提示词、案例、oracle或生产代码。

四案例：P1标准CSV＋非默认提交表头/原ID；P2多公告＋跨句目标释义＋陌生文件名；P3文本NDJSON＋预测字段改名＋指定正类概率；P4同容器自定义行角色＋中文说明＋无表头输出。各自仅有8行合成训练数据，另有3行预测；P4含2条不入训练的验证行。题面与oracle分别存放，oracle不进入payload，受控哨兵值不发给模型。

两路径共用包、client参数和调用/停止阈值：旧native→局部LLM适配（不调用A新增global fallback），对照新proactive全局解释。每格最多1条请求，四案例最多8条、总停止阈值¥0.80，每格最多预留¥0.10；0fit、0分类预测、0课程测试。旧路本地成功时不硬补请求。新路若请求工具而单格额度已满，保留未完成结果，不补第二次调用。本轮不是有界修复成功率实验。

读题时只让LLM看到文档与文件元信息；旧路沿原局部证据过滤，新路用合并版完整工作区。指标、字段类型、目标、正类、类别域、选中训练行、预测字段映射、输出recipe均与预先oracle逐项比较。真实解码和预测输入预检负责可执行性；引用存在不当作语义正确。有效JSON率、结构可执行率、各语义字段正确率和完整契约率分别报告，拒绝与误接受分开。即便四例全过也不代表完整未知任务保证成功。

运行模型DeepSeek-V4.1-Flash，temperature0、thinking disabled，既有流式客户端/相同输出上限；两路径统一用非特殊role标记，deadline90秒，连接/读取timeout沿现有客户端(10,60)，相同设置。显式campus_direct只在进程中用既有10.121.10.250解析，TLS/系统代理不变。请求失败或新增unknown/pending立即停止整批，不重试；已成功计量但契约错误只记录失败，不扩本格额度。

真实预算前置：使用D:/codex/python_withai/runs/a129_B_ledger_return/private/ledger.sqlite的同一非空累计账本，保留全部未知费用null，不新建个人/空账本。只读核对目前457请求/16未知/0pending/已知¥7.925792；旧授权清单覆盖12条，另外4条不在清单中，依AGENTS规则当前预检会拒绝，**本计划不自动扩大授权**。B新交付含尚未收到可导入逐请求原件的多批用量，单列为团队累计未核验，不能补造ID或以汇总造真账。先完成离线材料；付费需明确扩展现存未知清单或收到可验证用量解决阻断。无需用户查官网。

材料位置：scripts/sa5_semantic_cases.py、scripts/run_sa5_semantic_panel.py、tests/test_semantic_panel.py。离线prepare输出忽略的runs/sa5_panel_cases以及脱敏research/sa5_preparation；付费使用全新目录research/sa5_live_NEW/private对应runs/sa5_live_NEW。JSON oracle/包/原始请求回复可留本机，不提交真实数据/账本/密钥。结果对比与课程口径费用另报，实际学校扣款仍未知。
