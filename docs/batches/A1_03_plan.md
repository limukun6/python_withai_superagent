# A1-03：执行核对进入下一轮决策

运行前计划。使用同一个 benchmark.run_one、真实数值分类训练与模拟客户端，比较 raw/checked 两组；先 baseline，修复后 fixed，最后回归。模拟客户端不读密钥、不发 API。

每组 3 次搜索，不追加复核、不最终重训；两组每轮共 6 次真实分类器拟合，baseline+fixed 共 12 次。预计机器 30–90 秒，回归另计小模型拟合。所有输出使用新目录并保存源码身份。

受控故障：仅把某个 ExtraTrees 候选的事前分类器元数据改为 unknown，模拟新注册工具缺少说明。模拟提案声明 logistic，实际训练器仍是 ExtraTrees。执行器必须能发现矛盾并送达下一请求。另一条 hypothesis 故意引用 trial1 的 99.99% 分数，验证不含 assessment 的提案及 hypothesis 能触发审计。人为故障不代表模型自然犯错频率。

对照：R/E 共用原始指标与实际 estimator；R 屏蔽派生检查和记忆，E 提供配置核对、解释核对与失败记忆。客户端按是否收到配置矛盾选择不同后续说明，用于验证反馈可达性，不证明真实 LLM 会纠错或选择更好。

验收：worker 真实 classifier 与声明矛盾被记录；DecisionState 保留实际对象事实；下一 E 请求包含配置矛盾、假说失败与 prose 警告；R 不泄漏派生检查；不含 assessment 仍审计；hypothesis 被检查；每组模拟调用 2 次、真实 API 0、搜索完成且无随机恢复。

运行命令（仓库根目录）：

```powershell
& .\.venv\Scripts\python.exe .\runs\integration\scripts\run_a1_loop_batch.py --phase baseline --output .\runs\a1_03_baseline
# 修复后使用：
& .\.venv\Scripts\python.exe .\runs\integration\scripts\run_a1_loop_batch.py --phase fixed --output .\runs\a1_03_fixed
```

Regression 使用已有 run_a1_evidence_batch.py --phase regression，并增加本批的测试模块。源码变化导致失败则用新输出目录重跑，不覆盖旧日志。DeepSeek 从 EXPERIMENT_PROGRESS.md 续做；下一步正式 CLI 接入与真实 A2 应另写计划，本批不自动转付费实验。
