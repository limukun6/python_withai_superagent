# SA3：主动解释、结构探查和训练前预测契约试读

基线9258318。用户继续A结构开发；先写本计划、runner与测试再运行。0新fit/0真实API；不改B方法菜单/排序或真实账本。

1. 显式task-interpretation=fallback|proactive：fallback旧路径保持；proactive不因本地读成功就跳过模型理解，模型未配置时明确不可用而非假称LLM运行。unified_agent train/solve及其训练子进程、superagent unified贯通并记录模式。
2. 对无manifest的陌生标量包，模型可先请求inspect_header工具获取已授权成员的列名/codec，再提出任务契约。只发结构，不发训练或预测行值；已知评价答案/验证来源不可读。工具和语义修复共用原客户端调用上限，每类最多两步，不无限循环；API异常不重试。
3. 对已能原生读取的显式manifest，输入目标保持权威；proactive由模型解释附加输出要求，不覆盖已声明目标/指标/特征。输出元信息仍引用真实公告。
4. 预测路径可显式给定，也可按同包冻结query成员试读。在任何fit之前检查实际预测特征、原ID/样例契约；错误回送本次语义修复。训练与预测字段改名使用显式一对一field_map，不能将target/split角色映射进模型特征。

范围：已有record reader的组合、预测字段映射、CSV output recipe；不生成任意代码、不新造评分公式、不吞掉网络/账本错误。原生格式原ID提取若没有稳定读法，明确拒绝或延后，不用猜测替代。有效显式manifest不因新模式随意改义。

开发测试含普通native成功但错过输出要求、一次工具后提案、坏列名训练前修复、外部prediction-input、缺ID/模板前置诊断、无客户端proactive拒绝、manifest目标保护、未知工具/伪造引用/验证答案拒绝、预算到达、传输不重试、CLI传参到solve子进程、原fallback不发模型调用。仍用模拟LLM与确定性预测夹具，真实Flash成功率另立实验。

runner禁止Executor.worker和网络，输出新目录research/sa3_proactive_probe/{protocol,tests.log,results,completion}。已有无fit套件回归加本批测试；不运行含真实训练的全仓。实验/代码SHA记录，可交给DeepSeek续跑。当前未执行。
