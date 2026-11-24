# A139–A140：紧凑反馈少用输入token，三个固定状态的下一动作相同

瓶颈：完整核对报告和记忆增加上下文，但前期没有稳定质量收益。结构为默认关闭的`--feedback-view compact`：共同任务、原历史、候选和预算原样保留；只对附加checks/memory做有来源引用和同构记录表格化，提供反解恢复所有值。它不新增C知识、不删unknown/失败、不把自由解释变真，不修改执行与选型规则。

先保存协议/代码再运行。第一版只引用去重，三任务附加开销仅减3.268%，没过20%预登记线，0付费，负结果在f61cfc3与research/a139_feedback_v1。另登记第二版字段表格化，同样三个已见工程前缀、同门槛，40项无fit/无网络检查通过，全部roundtrip与共同字段守恒；附加开销减2268/9454＝23.9898%，整请求字节约减5.75%。不把同一开发上下文写成第二批独立验证，也不把字节当token。

A140固定Coupon/SMS/Flowers各state_3（前两个真实本地试验，不按错误挑例），raw/full/compact各1次、任务内轮换顺序，统一Flash/温度0/calibrated-v2/完整合法菜单；每任务最多3请求，官方DIRECT、无重试。全9回答封存后统一受限诊断，未训练下一动作、未看课程/H留出。三个原状态没有自然LLM解释，因此不宣称已修复旧LLM错误。

| 视图 | 调用/合法动作 | 输入token | 输出token | 本轮折算费用 | API延迟合计秒 |
|---|---:|---:|---:|---:|---:|
| 原始历史raw | 3/3 | 10157 | 2447 | ¥0.019945 | 85.283 |
| 完整核对full | 3/3 | 13157 | 3316 | ¥0.026421 | 81.078 |
| 紧凑核对compact | 3/3 | 12348 | 2480 | ¥0.022268 | 71.642 |

compact比full少809输入token（6.149%）；按本项目冻结输入单价，对应输入部分少¥0.000809。总折算少¥0.004153（15.719%）还包括少836个输出token；输出长度和延迟有模型/时段变化，单次样本不能承诺稳定省15.7%或稳定更快。compact仍比raw贵¥0.002323，不能说核对反馈比无核对更便宜。价格是课程折算，不是平台实际扣费。

三个任务三种视图下一候选均相同：Coupon选catboost_depth6_reg3.0，SMS选minilm_text_with_nonlinear_lexical_features，Flowers选image_mobilenet_v3_small_linear。执行契约9/9通过，支持本次三个状态动作未受压缩改变；不证明其它状态也相同或分类准确率无损。

受限prose检查raw/full/compact分别检查15/16/14条，其中unknown6/3/7，自动矛盾标记均0；C参数语义unknown分别5/0/1。未知不算正确，缺少人工金标准、只有每格一次回复，不能报完整错误率或声称纠错变好。compact的unknown较多也不藏起来；不按这组小样本晋级默认，现有full默认不变。

总9次成功请求、0新增fit，输入35662/输出8243tokens，¥0.068634，整队列238.359秒。A累计仍1614。实际同一ledger由363→372请求，known6.7532615→6.8218955，unknown仍12、pending0；原363条请求/全部旧transition逐项保持，非重建账本。原12未知按用户授权保留null，之后新增unknown/pending仍止队列。当前owner A/exclusive_idle，不再用旧363副本继续写账。

证据research/a139B_feedback_v2、research/a140_flash_projection；原始真实ledger和私下前备份在忽略目录，不进Git。复跑代码run_a139_compact_feedback、run_a139B_table_feedback、run_a140_flash_projection、evaluate_a140_flash_projection与归档工具均已保存。下一步中断/关机后的成果恢复，用已有模型0fit/0API，不补调用追更好结论。
