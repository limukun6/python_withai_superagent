# SA4：合并A语义适配与B最新自主研发版

用户要求把刚完成的A功能合入队友最新自主版。B完整研发分支codex/task-driven-planning-1125固定b0944ae；个人发布c6ea0b7的releases/autonomous-integration-2026-11-26是其发布副本，交接说明指定从完整研发分支根目录开发。A来源0d09760，含SA1–SA3，双方共同祖先d501807。

集成工作树runs/semantic_joint、分支codex/ab-semantic-autonomy-1126。保留双方历史与原工作树，做真实Git merge；不覆盖个人发布包或改旧分数身份。共享控制循环、unified配置与train/solve子进程逐段接线。

本批首先零训练/零真实API：计划与可复跑runner先落盘，再跑A的100项、B相关自主性/开题/反馈/预算回归，以及A+B组合参数与共用调用上限测试。Executor.worker和联网禁用；涉及真实训练的旧用例单列，不用pass掩盖排除项。若需要实际训练烟测，另登记小批预算后执行，不混入本批0fit结论。真实账本和课程测试不变。

必须保留：A主动/回退契约、输出recipe、概率接口过滤、field_map、预测预检；B反馈选型、自主开题、复核/提案/规则继承、已验证可恢复交付。两边选项同时经官方CLI传入同一个训练子进程；语义适配消费的调用计入同一client，B开题在额度不足时不能新增调用；网络失败不重试。B原研究上限16保留为显式研究能力，当前验证不做训练，团队11fit公平对照约定不因此改变。

输出research/sa4_ab_merge*/protocol.json、tests.log、results.json、completion.json；对冲突、失败、测试排除和实际资源诚实记录。通过后提交、推到新集成分支，提供队友可直接使用的SHA及命令。只表示本阶段整合，不冻结整个项目，也不把B旧真实分数冒称本次合并版实测。
