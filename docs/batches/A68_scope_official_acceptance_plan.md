# A68：比较范围修复的正式入口验收

A67开发基线44/48，4个跨指标/划分/payload错误确认；修复后同一案例48/48。中间两次runner失败保留（缺fixture search_accuracy；slice scope变量落错函数）。A67额外55项回归通过，但误包含HypothesisAgent.IntegrationTests中的Dummy.fit：6+3+3+3=15真实拟合，必须回溯补记，不继续报零。测试无真实API；15基于源码与通过记录重建，不冒称原有物理观察。起点1007，修正后1022。

本批先保存runner再运行，新增最多3实际监督fit/API0，累计最多1025；只做一个匿名合成numeric任务的local/ab-v1/fit-limit3正式CLI，既有fixture生成器不训练。拟合物理观察器与正式admission账本一致，两个primary观测带payload/label/rowID指纹，hash与prediction_evidence逐条一致。新进程predict通过，无外层分数，H/test不打开。

同时用实际历史观测验证：同primary范围的数字预期可评价；改split/payload/primary metric或仅一侧提供身份，必须not_evaluable。0额外fit，明确是受控元数据反例。原缺完整身份的legacy检查仍可按旧值比较，但新报告provenance_complete=false，不声称已完整核验。正式AB当前提供完整身份；其它旧入口全面改造属于后续边界。

仅跑本次相关无训练测试类，排除HypothesisAgent.IntegrationTests，不为重复验证再拟合。保存计划/源码/结果/物理轨迹与资源更正。约10–30秒，单路径240秒、每worker60秒，失败即停。

```powershell
python -m scripts.run_a68_scope_acceptance --output NEW_OUTPUT
```
