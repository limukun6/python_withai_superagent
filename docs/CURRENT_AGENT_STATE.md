# 当前接续状态：先查实际产物，再执行

## 活动队列

- A89 Coupon六组C1已经实际启动，代码/计划提交87aa0d1；superagent生产文件内容与aa47bb9一致，后续只改文档/工具/环境包装器。输出D:/codex/python_withai/runs/a89_coupon_C1。L1预登记复用A88同源11fit；其余5条最多55新fit/12Flash。先查completion.json、checkpoint.json、physical_fits.jsonl与唯一真ledger，禁止重复启动或编辑superagent/活动runner及其protocol.script_hashes。
- A91禁用预训练视觉特征的照片full8消融已实际排队等待A89 completion，无并行训练，输出D:/codex/python_withai/runs/a91_restricted_photo_Q。复用A88 baseline7的10fit模型，最多11新fit，0API。教师说明已允许通用下游特征，故它不再是课程资格补救；菜单/顺序/预算仍按原计划保持。先查completion，不重复启动等待器。
- 两个线程级防待机请求分别等A89/A91 completion释放，另有共同2小时截止；输出runs/a89_sleep_guard、runs/a91_sleep_guard。结束时核对completion，不留下永久配置或启动其它模型。

## 资源与课程边界

A89前A累计1290。当前实耗＝1290+A89 physical行数+A91 physical行数，复用模型不能再计fit；之后优先各批completion的A_cumulative_fits_after，防重复相加。A89前真ledger299/known¥6.1417335/11历史unknown/pending0；新请求实时读真账本，不用这些旧数字覆盖当前。唯一ledger为D:/codex/python_withai/runs/a23_real_local_history/ledger.sqlite，密钥为根api_key.txt，均不提交。

A93用户转来老师明确说明：Qwen/DeepSeek限制针对SuperAgent语言模型，下游允许通用预训练特征，但必须自主选用、不得任务名称路由。说明原文在TEACHER_ENCODER_CLARIFICATION_20261122_zh.md；不再等待同一确认。wrapper默认auto，有校验资产时提供通用编码器、无资产保留常规方法，full菜单15/32/16/9/5；restricted保留消融。A90未知状态为历史，不改成当时已知。当前Coupon C1完全不调用编码器、目录不变。H1/H2与课程test封存；独立48案例及24语义标注仍缺。

## 已完成与下一步

- A74–A87整合、真实编码器执行/交付、成本校准完成；技术研究结果见batches/A75_A87_encoder_integration_results_zh.md。
- A88 Q两已见任务四路径43fit/363.890秒：Coupon净+4/2030（+0.1970pp）；photo净+2/80（+2.5pp）。photo full曾自主试视觉但选梯度，资源范围现已澄清；A91补不含视觉的消融，隔离菜单可用性作用。不要拿旧A47分数当本轮对照，也不因此说未知任务已验证。
- A89发现相同主观测Fields参数/硬预测一致而时耗约5.12倍，根因未定位；复用L1旧秒数不作新效率因果结论。当前两执行环境可见CPU均16，不据此认定权限导致差异。
- 继续监控A89→A91直到完成/失败，汇总实际新fit、真实请求/tokens/估价、复用、正负结果与边界；完成后更新本卡和EXPERIMENT_PROGRESS并提交推送。不要自行打开H或把单任务C1称完整跨任务矩阵。

具体命令与协议：CURRENT_C1_A89.md，batches/A89_coupon_six_arm_C1_plan.md，batches/A91_restricted_photo_Q_plan.md。运行环境为runs/a79_encoder_assets/venv/Scripts/python.exe（独立覆盖，借只读torch运行库），环境入口scripts.run_with_encoder_assets默认auto；常规主解释器仍根.venv/Scripts/python.exe。
