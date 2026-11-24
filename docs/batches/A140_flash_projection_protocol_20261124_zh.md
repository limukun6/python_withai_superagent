# A140：三任务固定历史×三视图的真实Flash机制小试

准入来自A139B：三任务exact roundtrip、共同字段相同、各任务缩短、附加开销减23.9898%≥原20%。第一版3.268%负结果保留。三前缀都来自local运行，尚无原自然LLM解释可纠正；本次评新回复，不声称旧错误已修。当前累计1614fit，ledger363/known6.7532615/unknown12/pending0。

固定A139B inputs和源码94个文件，不再更改生产/参数/上下文。轮换次序Coupon raw/full/compact；SMS full/compact/raw；Flowers compact/raw/full。每视图1次，每任务最多3次，总9；同一模型Flash、温度0、calibrated-v2输出、共同菜单/任务/历史/预算。无监督训练、无外层/课程test读取、无API探测或重试。固定scope映射coupon→task1、sms→sms_spam、flowers→tensorflow_flowers_b_cohort，与已有同任务账本合计，不新建空账本或虚假scope。

原实际ledger runs/a129_B_ledger_return/private/ledger.sqlite，A继续独占；先一致性私下备份，再显式accepted_history绑定原12未知ID与行SHA。默认strict不改。三个客户端各至多3请求/估计停止阈值¥.20，合计¥.60；timeout沿既有校园客户端的10秒连接/60秒读取，无自动重试。官方域名进程内解析至既定10.121.10.250，TLS/SNI不变，不改系统代理。新增unknown/pending或请求/用量/契约失败立即停止批次，已完成调用和费用原样保留，不能补呼叫追满9。预算阈值是估计准入，不是平台扣费上界。

每个回答立即封存context SHA、response、usage，再做执行契约合法性检查；全部成功回答封存后统一做语义诊断。主要记录每视图实际prompt/completion tokens、折算费用、延迟、合法动作、独特候选、受限程序可判定的事实矛盾及覆盖率。旧检查器的unknown/否定限制保留，报告“自动检查标记”而非人工错误率。既有参数语义仅用于事后诊断，不额外给compact参数知识。重复response_id与相同提议分别报告，不作独立抽样。

只有3个固定状态，每状态每视图一次；不能推出稳定节省、统计显著或分类准确率提升。不同动作的质量需后续另批实际训练，本批0fit。模型/完整key/ledger/原始私密日志不提交。公开summary/usage/回答均先排除credential；owner在每次和结束后更新为真实累计，避免B接回旧363副本。
