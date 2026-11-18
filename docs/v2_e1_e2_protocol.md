# SuperAgent v2 E1/E2 批次协议

本协议接续 Flash 流式 P0。P0 的 8 条运行用于确认流程，seed 101 不进入 E1 的配对结果。

E1 固定两个发布开发任务、5 个种子（41、52、63、74、85）和四种策略（D/O/F/R），共 40 条搜索轨迹、最多 240 次训练和 90 次 Flash 调用。每条运行分配 6 次训练尝试，前 2 次为同任务共同基线。单次训练限时 180 秒，搜索限时 3,600 秒。策略串行运行，前四个种子轮换执行位置，第五个沿用第一种顺序，两任务的轮换起点错开。实际顺序与源码哈希写入 `protocol.json`。

在官方 train 中生成相同的 F/S/C 分组划分。E1 只用 F 拟合、S 搜索，冻结 S 排名前两名及其文件哈希，直接选 S 第一名。C 的行索引单独保存，训练 worker 和规划器的输入不含 C 特征或标签。选定后才访问官方 validation。正式 test 数据在全程关闭。

E2 从 E1 的冻结候选重放同一个确认规则。只保留一次 C 评分机会，挑战者至少高 0.005 才换模。先检查划分和候选文件哈希，然后计算同轨迹 C-on 减 C-off 的开发验证差值。自动输出每条轨迹、换模数、获益/受损数及平均差值。只有一个成功候选的轨迹不能换模。E2 无新增训练与 API 请求。

在仓库根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_v2_e1_e2.ps1" -ApiKeyFile ".\api_key.txt" -ValidateOnly
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_v2_e1_e2.ps1" -ApiKeyFile ".\api_key.txt" -BatchBudgetRmb 2
```

当前使用 Flash 流式客户端、60 秒的读超时和共享 ¥2 的暂估批次预算。该数值来自 P0 用量规模与开发防护需求，不能作为课程认可的官方计费。遇到请求失败且用量未知，保留记录并结束批次，不自动重试。输出位于 `runs/v2_e1/`。完整 E1 结束后，入口自动生成 `analysis.md` 和离线 E2 的 `confirmation_ablation.json`、`confirmation_ablation.md`。

主要配对对比是每任务每 seed 的 R−F，F−O 和 O−D 为次要对比。几个 seed 共用同一数据集，报告配对差值、完成率和成本，并注明这是两份开发任务上的观测。
