# A83：共同运行环境入口与冻结前指纹

A81执行期间只新增包装脚本/文档，不改superagent源码或活动runner。目标是避免主Python3.14缺权重/依赖而默默隐藏强方法、另一台机器目录不同也看不出原因。提供--assets指向已校验资产登记，从environment.json选择真实解释器，并设置两个明确权重环境变量；转发原superagent CLI，不另写调度或换候选。

本批只以--dry-run run匿名.zip和实际help验证包装（不会读任务/训练/API）。固定本机runtime/资产/source/候选指纹作D4候选快照，不把它当正式协议已经共同冻结。A81完成后，用已完成的实际fit耗时安排Q/C1；B的原结果保留原source，H/test不开。

0fit/0API，预计5分钟；不把资产或venv提交Git。未来运行仍可选择local/raw-history/checked-history，但Flash须原真账本、同任务ID和既有历史规则。包装只负责环境，不改变学校API直连和60秒超时/失败停策略。
