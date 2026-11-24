# A44–A46：网络失败后继续结构研发

在任何新源码/训练前登记。A42第一条新路径1基线fit后DNS11001失败，队列停止；请求绑定的DNS证据追加审计为not_dispatched，原unknown转移与日志保留，原11历史未知不变。当前不再发真实API、不改网络。原A32/A42原结果保持旧源码身份；本计划开始新生产版本，不能用旧prepared冒充同版。

## A44：Agent先看见实际训练数据

瓶颈：AB提示有任务类型/目标/候选，却没有已计算的类别分布、缺失/长度/图像形状。将A37独立训练facts抽取接入正式AB状态与日志，新增显式--task-profile off|train，默认off保持旧行为。raw与checked得到逐项相同的training_profile；本地也保存，但不改变B选择逻辑。只用供给训练数据，不读validation/test，不传原始文本/路径/数据集名。统计列列表有公开截断规则和来源标注。

验收：旧off上下文与历史请求重建一致；五真实开发任务训练摘要/改名不变/两组共用的合同测试；新的本地正式CLI验收训练、refit、重载。只证明接入与事实正确，不证明Flash会因此选得更好。

## A45：允许诚实的不确定性与诊断动作

瓶颈：AB v1强制每个新方法都承诺search_score上升，诊断/覆盖动作被迫写涨分故事。新增显式--proposal-contract strict-v1|calibrated-v2，默认strict-v1。v2仍只从同一候选清单选一个PRIMARY动作，不增加训练/请求；增加purpose=improve|diagnose|coverage与uncertainty，expected_change允许null或训练内score/gap的增加/下降。数值预期仍须有限数、同PRIMARY已完成reference，目的本身不能放宽证据/配置/预算核验。

诊断/覆盖没有数字预期时记录not_stated，而不是自动supported；若明确预期下降，则机械核对下降，不能将“下降预期实现”解释成最终目标变好。记忆保存purpose/uncertainty为未验证计划，不当成因果事实。raw/E只在checks/memory不同，schema/instruction共同升级。

验收：错split/未来引用/伪配置/不合法字段照样拒绝；null和下降等新合法用例v1拒绝、v2接受并记录正确检查状态。测试为人工故障/接口案例，真实LLM用法和性能待验证。

## A46：失败后自动交付已有模型，保持失败身份

瓶颈：异常后trial模型留在磁盘，但统一入口没有可预测的包。新增显式--partial-delivery off|checkpoint，默认off。异常且已有成功PRIMARY时，核验split/prediction/configuration/fit账本，按内部primary指标选择已完成模型，0新增fit/请求、不跑refit、不读外部val，生成partial标记的模型与任务契约。完整run仍failed、返回码非0、requested_planner_completed=false；不会用部分包假装完整成功，也不因恢复而触发下一付费阶段。

验收：完整CLI五任务故障注入（在两次成功训练后模拟后续失败），导出模型搬迁到新目录、新进程禁训练/网络预测。另做无成功模型、损坏证据、不同split/账本不完整等拒绝案例。原故障不能被恢复错误覆盖。人工注入不算真实网络试验。

## 资源与顺序

先实现和窄合同测试（0fit/API），再冻结这三个开关的新源码，执行一轮五任务本地入口与故障注入批。批次代码先落盘，实际worker观察器计数，初始上限每任务完整3fit+失败2fit，共25fit；不做性能种子搜索。若为raw/E各跑失败路径，另登记后才能增加，不临时扩量。

五任务均为已见开发任务，不开H1/H2或正式test。先只训练，不读外部标签；预测仅用已登记feature包。若batch失败，保留目录/实际fit，不自动再跑。真实API=0，mock回调/故障数量单独报告。结束前更新报告/交接/Git并写下一批真实Flash开关实验，不能把本地契约验收当精度改进。

## 首次验收后的修复登记（未读任何外层标签）

a44_a46_offline首条coupon完整训练3fit后，运行器发现第二个状态的training_profile被同名worker facts局部变量覆盖，停止、未预测/评分、0API。原目录保留。已将训练摘要保存到独立变量，避免后续观测覆盖、也避免默认off被错误加字段。

修复后在新目录a44_a46_offline_fixed重新执行本计划的25fit上限，必要重训不能从账本漏掉。两次合计训练上限28；旧124项回归已实际7fit，最终源码回归如再执行也另计。额外默认off持久状态检查优先复用最终回归的实际CLI夹具，不为比较同一模型成绩重复搜索。

第二次验收已验证coupon完整3fit/搬迁预测；故障分支2fit后暴露checkpoint读取假定错误：trial worker包没有final model包中的kind字段，恢复被守卫拒绝，原故障没有被覆盖。该5fit也保留。先用这两次成功训练的原模型做0fit的恢复/损坏证据检查，修正为核验可信payload.kind、capability.input_kind和metric，不再假定worker包字段。通过后才在新目录a44_a46_offline_final跑25fit上限；三轮累计上限33fit（3+5+25），不漏掉失败验收的8fit。
