# A97：同一花卉完整目录的六组决策对照

2026-11-23。A96已完成，A累计1385。瓶颈假设：可用强表征不等于预算内一定会选到；需要把方法库收益与Flash/反馈/调度的收益分开。

使用A95的800训练/200已见开发留出、五花类、同一9候选完整目录、真实内部分区707/808/909、相同生产source和3.12依赖/视觉权重。每路径最多11实际分类器fit，含最终refit；Flash最多3调用，trial300/path1200秒。不人工指定赢家，不依据外层标签改候选或策略。L1显式复用A96 full的10fit及已评分事实，旧文件只读；其余五路径全部冻结后再统一评分。复用路径不当独立新样本，其旧耗时不作新因果效率证据。

六组：L0本地固定、L1本地动态、R0原始历史固定、R1原始历史动态、E0核对历史固定、E1核对历史动态。新增顺序L0/R0/E1/E0/R1；与Coupon不同顺序降低同一顺序锁定，但单轮不能消除顺序/缓存混杂。每组同候选集合、实际划分和预算。比较准确率、逐行输赢、所选方案、探索/复核动作、事实/解释核验、实际fit/墙钟/API/tokens/费用。预期完整目录在本地已较强，Flash可能持平或退化，负面结果同样保留。

最大55新fit、10复用fit、12新请求。唯一原真实非空累计ledger不重建，任务scope tensorflow_flowers_b_cohort；每条付费路径增量估计阈值¥0.30，四路径合计¥1.20，usage按原课程口径结算，不是平台扣费硬上界。历史11unknown保持，incremental-research沿用户既有授权；新unknown/pending或任何请求/训练失败就整批停，不重试/不自动换模型。检测其它ledger写入也停止。学校API仍60秒，固定官方域名直连10.121.10.250且保留TLS Host/SNI，不改系统代理。

先用--validate-only生成独立预检目录（0fit/API），再启动实际目录。首个失败原样留completion/log/账本；不覆盖旧批。所有训练worker禁网，只有规划器可调用学校API。使用临时防待机上下文，退出即释放。按A96单路径约1分钟加12次网络响应，预计10–25分钟；理论五路径100分钟加交付余量不是实测耗时。A96完整组已见开发val，因此本次是已见任务机制对照，不是独立泛化或H验证。

运行（在integration工作区，用已配置强运行环境）：

```powershell
& D:\codex\python_withai\runs\a79_encoder_assets\venv\Scripts\python.exe -m scripts.run_a97_flower_C1 --data D:\codex\python_withai\runs\a95_object_photos --reference D:\codex\python_withai\runs\a96_flower_Q --assets D:\codex\python_withai\runs\a82_encoder_assets --ledger D:\codex\python_withai\runs\a23_real_local_history\ledger.sqlite --api-key-file D:\codex\python_withai\api_key.txt --output D:\codex\python_withai\runs\a97_flower_C1_preflight --validate-only
# 实际执行使用同参数，去掉--validate-only，output改成新的a97_flower_C1目录。
```

数据/模型/权重/凭据/SQLite不提交；只归档脱敏协议、动作、实际fit和指标摘要。H/test封存，独立48例/24语义标注仍不是本批可以替代的。
