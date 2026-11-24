# A48：新契约的整条请求链验收（模拟HTTP，不是真实Flash）

先写计划与代码，再在A47训练结束后串行执行。目的补齐A45只测parse/context的边界：正式CLI→训练摘要→ExperimentLLM→SSE解析→v2提案→机械检查/记忆→调度→重训/预测，并检查真实transport形状发生超时后自动partial交付。

只使用已有匿名合成数值数据夹具，raw/checked各一条动态5fit路径；mock选择第一个合法候选，第一请求purpose=diagnose且expected_change=null，后续可声明score_gap下降。检查实际所有共同上下文相同，null的检查必须not_stated。第三条checked注入单次requests.Timeout，最多1基线fit、1mock调用，随后生成partial包但exit=2、requested_planner_completed=false、不重试。

上限11真实监督fit（两完整各≤5＋故障1）、7模拟HTTP、0真实API。模拟历史ledger与fixture credential独立在本批目录，标fixture_only/paid_ledger_eligible=false，永不导入共享账本。系统audit拒绝真实socket，HTTP Session全拦截。真实账本前后SHA必须相同。匿名31行无标签query在新进程禁训练/网络预测，原/搬迁输出一致。

来源生产代码与A44最终、A47相同，期间不改。所有结果仅证明集成/计数/失败行为，不能当真实LLM理解、诊断效率或分类性能收益。记录mock与真HTTP分别为实际数，不把模拟token或金额计入真API。结果目录a48_calibrated_mock_loop。
