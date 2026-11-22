# A88：A线同源Q对照，Coupon与真实照片

前置A74–A87整合、真编码器与交付已验收，A起点1247。冻结当前superagent源文件/依赖/资产和A47已登记两个训练包及实际划分，优先完成A负责的Coupon和photo，不重复B原源的其它任务。四路径：Coupon baseline→photo full→Coupon full→photo baseline；各local ab-v1、同任务共同菜单资格/seed707/分区、11fit含refit、trial300/path1200秒，最大44fit/0学校API。

baseline/full分别14/15、7/9，原/扩目录均从统一注册过滤origin而来，不人工点赢家。其差值评价整个新增方法组合（照片梯度+冻结视觉），不单独归因于视觉编码器，也不归因LLM。保留default local-text-proposals off、其它研究开关默认，不改变调度。真实CPU顺序一条队列，所有子进程禁网；每path真实外部fit计数和正式账本核对，保存可执行菜单、实际split、动作与模型哈希。

四模型全部冻结后才统一读已登记开发val/照片开发留出标签，独立进程预测/核验；原课程test/H绝不打开。比较主指标、同样本预测差异、拟合/时间和赢家，不在本批看中间外层成绩调下一路径。不追seed，首失败停，无自动重试，保留partial但未齐则不提前评分。

这是当前整合源的A线Q子矩阵，不冒充五任务Q、六组C1或未见H已完成；若之后共同source/行号/环境改变，旧结果仅留开发身份，不能自动拼作最终同源。预计机器5–20分钟（四路径名义上界80分钟+进程裕量）。输入/源码/权重不进Git，只交代码与脱敏记录。计划与runner先提交，额度中断可由DeepSeek同环境续查completion，不能重复运行同目录。

```powershell
Set-Location 'D:\codex\python_withai\runs\integration'
& 'D:\codex\python_withai\runs\a79_encoder_assets\venv\Scripts\python.exe' -m scripts.run_a88_A_track_catalogue_Q --reference 'D:\codex\python_withai\runs\a47_catalog_Q' --assets 'D:\codex\python_withai\runs\a82_encoder_assets' --snapshot 'D:\codex\python_withai\runs\a83_shared_catalogue_snapshot' --output 'D:\codex\python_withai\runs\a88_A_track_catalogue_Q'
```
