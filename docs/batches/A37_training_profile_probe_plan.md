# A37：训练数据事实摘要接入原型（独立0API工作）

代码检查发现superagent/profile.py已有类别分布、缺失、文本长度/图像形状摘要，但正式AB context_for只发送TaskSpec、历史与候选，没有这些初始训练数据事实。这是接口接入缺口，不代表已证明它导致分数损失。

当前A32及队列生产源码冻结，故只在tools/a37_profile_contract.py实现独立原型，复用原profile算法，不改superagent或正在使用的脚本，不发API/不训练。使未来raw/checked共同得到同一份训练内事实，不能把数据分布猜测伪装为事实或把额外信息只给checked组。

五个已见开发任务各读已核验train-only包，取统计不取原样本、不读外层val/test。用显式y_val=None适配既有profile，核对样本数、标签计数总和/域、feature shape与主指标；保留训练包/标签哈希来源，但不把文件名/路径加入模型可见profile。输出每任务简洁JSON事实及“AB当前尚未接入”的缺口；用实际分类器fit观察器与网络audit保证0fit/0API。

检查摘要不随任务展示名称变化、不包含validation_label_counts、标签计数与训练行数一致、raw/checked公共增加字段逐项相同。报告UTF-8字节长度，不冒称模型token数。只是原型/事实验收，不叫LLM收益、性能或未见迁移。未来需要新版本开关/同信息对照再接正式客户端；本批不改变当前实验条件。
