# A1-02：配置事实与测量引用绑定

运行前计划，2026-11-20，负责人 A。先 baseline，再改实现，最后 fixed 与离线回归。零 API、零性能训练；回归的模型夹具另计。

假设：实际分类器不能按相似名称猜算法；测量证据必须绑定 trial/split/metric/value。仅“数字在历史出现过”不得确认整条陈述。无法理解的否定、未来或混合指代保持 unverifiable。

范围：Ridge/核 SVM 的实际配置辨别；新增结构化测量核验；缩窄自由文本的确认范围。下一请求的状态投影放在 A1-03，本批不宣称已经形成闭环。

固定开发历史：trial1 search accuracy=60%、train accuracy=90%；trial2 search accuracy=70%、train accuracy=95%，另有 primary_metric=macro_f1，train/search score=80%/65%。加入失败试验并验证不能伪造有效观测。案例先登记在 runner，覆盖算法家族、核函数、错试验、错划分、错指标、未完成引用、无作用域百分数与否定。它们是已知故障回归，非独立泛化集。

验收：错误配置不再确认；显式绑定的合法测量确认，错值矛盾，未知/缺失/歧义不确认；百分数无法绑定时保留未知，不能用全拒绝掩盖覆盖率。保存每类合法确认数、错误确认数与未知状态。

从主仓库执行（新目录，不覆盖）：

```powershell
& .\.venv\Scripts\python.exe .\runs\integration\scripts\run_a1_evidence_batch.py --phase baseline --output .\runs\a1_02_baseline
# 修复后；已完成 baseline 不要再用新实现假造修复前结果。
& .\.venv\Scripts\python.exe .\runs\integration\scripts\run_a1_evidence_batch.py --phase fixed --output .\runs\a1_02_fixed
& .\.venv\Scripts\python.exe .\runs\integration\scripts\run_a1_evidence_batch.py --phase regression --output .\runs\a1_02_regression
```

脚本先保存 protocol 与源码指纹，逐例保存 results，结束写 completion；regression 保存日志和退出状态。预计探针数秒，回归 1–2 分钟。DeepSeek 续跑先读 EXPERIMENT_PROGRESS.md，复用完成记录，错误保留；本批不读取 API key 或发请求。
