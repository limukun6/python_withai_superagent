# SA1：全局语义契约与试读修复，先零fit/零API

基线d501807，独立分支codex/a-semantic-adaptation-1126。用户允许四天结构开发；本批先实现核心，不调用学校API、不训练、不改B目录或排序。

实现：已有native/adaptive解释失败后，允许LLM读取多份真实说明的证据块、选择非标准训练/预测文件、目标/特征/指标与split映射，输出版本化适配计划；程序试解码并构造现有TrainingTask，失败回送诊断，最多一次语义修复，且共享原客户端调用上限。异常传输不重试。显式有效manifest继续权威；包路径、已知测试/验证来源、标签/目标进入特征、证据伪造等错误不可被解释器放宽。

本批支持已有标量record读法，不生成代码、不执行插件、不新增指标计算器。输出只冻结既有predict可重放的input_recipe；公告驱动OutputRecipe和完整工具循环留SA2，不声称本批已完成所有设计。

开发场景：多份非标准命名说明、跨句/同义目标、未知文件名、非标准行split、字符串标签、首次错误计划后修复、伪造引用、测试/验证成员拒绝、显式manifest保护、传输失败不重发、剩余请求不足。用模拟客户端和真实数据解码验收，不训练模型；检查原输入不改、编译结果及预测期字段/行序重放。保留组未来单独冻结，本次案例仅工程开发。

先保存计划/测试代码，再运行：python -m unittest tests.test_semantic_contract tests.test_adaptive_formats -q。相关广回归选择无真实fit的类；不直接全仓运行。输出research/sa1_semantic_contract/results.json与completion.json，新目录不覆盖。新生产源和runner/test SHA记录在回执。

预算上限：0物理fit、0真实模型API、0课程新增费用。测试时间预估分钟级。先验证编译/修复/重放和实际入口接线，不能把JSON合法或模拟提案当真实LLM语义正确率。
