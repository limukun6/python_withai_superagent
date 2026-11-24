# A77：正式入口的有界文本提议实训验收

A76三方增量和显式CLI开关已过76项禁fit检查，A起始累计1228。本批不重复B2三组效果研究，只验“研究策略能否从正式入口自主执行与交付”。已见SMS原训练包/原macro-F1，local ab-v1、完整相同菜单、seed707、11次fit上限含refit、trial180秒/path1200秒、--local-text-proposals evidence；不人工挑动作，0API，不打开H或外层标签。

必须能在events日志找到至少一次由合法已观测试验触发的local_text_recipe_proposal，核对所选方法实际执行、动作引用存在、真实计数不超预算。无提议则如实记未触发，不修改阈值重跑。最终模型冻结后预测31条无标签输入，正式新进程与独立源码包CSV一致；这31条来自训练输入，只验证交付，不算泛化效果。

另一个预先指定的有界pair_text recipe经真实worker执行，最多1fit，核对按preference语义交换后的概率互补。总上限12fit、预计1–4分钟，首失败停批、不自动重试。保存source/runner/plan哈希和protocol/results/completion；真实语义/执行与性能效用分别报告。编码器资产未就绪则保持隐藏，不用mock替代实训。
