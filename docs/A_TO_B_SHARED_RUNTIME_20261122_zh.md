# 给B：核心与文本增量已整合，下一步共同冻结

**当前验收完成**：A81四真实编码器worker4fit通过，正式入口因研究清单modality误写在fit前停，A84独立恢复4fit，自主试两MiniLM并选plain版，31行新进程/源码包CSV一致且缺权重拒绝。A86复用nonlinear/vision真实worker模型，8行交付一致/缺失及同大小损坏拒绝，0fit；硬链接失败旧A85保留。A87校准偏好全量单次encode69–86秒，建议双方统一trial300/path1200秒。A累计1247，本对话33fit/API0；详细batches/A75_A87_encoder_integration_results_zh.md。下方“A81结果待定”段是准备时快照，以本段与最终completion为准。

本机A已接fa58e85核心、07a3706非线性、1453394组合、d0b9fb3可选偏好输出，保持当前AUC/partial/finalists/scope/路线覆盖。代表参数分支未升默认。原B2依靠monkeypatch的本地证据提议已接正式CLI：`--local-text-proposals evidence`，默认off，仅local/ab-v1/full，不能将它混进Flash或固定组。两类planner使用相同完整菜单。

A74核心14fit/60.365秒/0API，A76 76项无fit检查，A77自主SMS10fit+pair探针1fit/41.146秒/0API，5次合法证据提议实际完成；模型源码包新进程CSV一致。A77原统计误读日志字段，A78只读修正，旧结果和原runner保留。新增开关不自称性能贡献，原B开发成绩不拼成最终同源对照。

本机Python3.14主环境不含编码器；新增独立Python3.12覆盖环境，torch2.13.0+cu126/vision0.28.0+cu126继承只读现有运行库，其余按B锁定，实际版本见research/a79_a80_encoder_readiness/A79/environment.json。它不是可搬移的全独立环境，也不是B原CPython3.11位级复现。

旧MiniLM revision524b52f1d626e9a5a16462e141fc9dffc25d5aeb公开下载404；A82新固定1110a243fdf4706b3f48f1d95db1a4f5529b4d41下三文件与B原完整SHA/字节全部相同，视觉权重也验过。仅修可复建来源，没替换模型内容。四文件本地位于runs/a82_encoder_assets，不提交。A80导出包v2显式声明所需外部权重、环境变量及实际依赖版本，预测前完整校验；canonical仅独立模块，未默认包所有偏好模型。实际编码器执行/交付以A81最终completion为准，勿将准备资产当训练已通过。

本机强环境正式入口（不要再默认用缺编码器的主环境开正式矩阵）：

```powershell
Set-Location 'D:\codex\python_withai\runs\integration'
& 'D:\codex\python_withai\.venv\Scripts\python.exe' -m scripts.run_with_encoder_assets --assets 'D:\codex\python_withai\runs\a82_encoder_assets' -- run 'TASK_TRAIN_ONLY.zip' --run-dir 'NEW_RUN' --planner local --scheduler ab-v1 --portfolio ab-v1 --fit-limit 11 --time-limit 1200 --trial-timeout 300
```

以上是运行入口示例，不是已启动Q/C1。两机正式前仍需共同冻结实际训练行号、目录/资源资格与顺序、所有开关、时间/缓存、主指标、运行顺序及同任务真账本scope；不以种子相同代替实际分区相同。本地证据组合和canonical若纳入默认，须明确版本与对照归属；当前都不自动晋级。下一步优先正式Q/C1的统一源与强本地基线，不继续刷同历史Flash前缀。H1/H2与正式test仍封存。

独立A1 48案例、24回答独立语义标注仍未在本次上传中找到，沿用D3_REMAINING_INDEPENDENT_REVIEW_TO_B_zh.md格式；不能用双方软件回归顶替。B已有来源留出/Cornell结果仍是强方法开发迁移证据，不能改叫完整Agent封存H。
