# SA6：真实探查失败的离线工具修复

基线e9f0c31，SA5已封存，先落盘本计划、runner和测试，再执行。0新fit、0真实API、不改SA5成绩/案例/旧回执或账本。

瓶颈：三次真实回复都选择inspect_header。P1/P4给扁平CSV或根JSON列表提供了并不存在的命名集合；P2把同包预测成员声明为source=prediction，但工具将逻辑角色等同于独立包位置。

改动：source表达训练/预测角色，物理位置由实际提供的包决定，明确返回package_source；存在独立预测包时不悄悄跨回训练包。工具可对确实无命名集合的扁平容器返回实际字段和null collection，并记录requested_collection及归一化依据。多集合JSON/SQLite/工作簿不能忽略无效选择器或随意打开默认表。已知验证/答案来源和伪造引用仍在读成员之前拒绝。

实验：原三次请求逐字段回放，用合成原包与真实reader得到结构；这是事后开发回放，不能算模型修复成功。另测根列表/CSV别名、带命名集合JSON不回退、SQLite错误选择器不打开其它表、同包预测角色、独立包缺成员不跨包、验证/答案/伪引用拒绝、仍只发字段名、总调用计数不变。加SA4无fit150与SA5 oracle5相关回归，不运行含fit的两条旧夹具。

runner写research/sa6_tool_repair_NEW/{protocol,tests.log,results,completion}，禁止Executor.worker和网络，登记源SHA。修复后真实多轮语义效果仍需新案例/新预登记；本轮不花剩余额度重跑旧题追成功。
