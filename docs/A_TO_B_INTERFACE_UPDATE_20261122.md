# A给B的接口更新：源码1319e5c及后续结果提交

已推送GitHub codex/expanded-feedback的1319e5c包含正式A结构与Q证据。当前真实A50/A51尚在跑，别改A正在用的源码；新结果以之后completion与报告为准。B方法/目录/调度实现没有被A覆盖。

## 实时输入与动作保持兼容

内部state仍ab_schema_version=1，TaskSpec/history/available_candidates/prediction_evidence/budget/registered_confirmation_splits保持原接口。启用--task-profile train时额外有training_profile：仅供给训练数据的聚合事实，含类别分布、形状/长度/缺失、primary metric、截断来源。raw与checked拿到同一份，不让一组多看训练事实。没启用时字段不存在。

Flash默认strict-v1仍9字段；显式calibrated-v2新增purpose（improve/diagnose/coverage）与uncertainty，expected_change可null，或已完成同PRIMARY score/gap的增/降。它不改变你返回的explore/confirm/stop六字段，也不增加fit/API。null的机械检查是not_stated，不是supported；预期下降猜对不表示主目标变好。数值/配置/证据核验仍在A执行器。

--partial-delivery checkpoint在后续异常时包装已成功模型，不重训/请求、不打开外部val；run仍failed、requested_planner_completed=false、exit2。model_available=true只是有部分交付，批次不能因此自动继续收费。终态delivery_status区分complete/partial/unavailable。

## 导出器已补当前正式格式

tools/export_b_handoff.py不再只支持早期trials/task_contract/split，也支持现在history/多个split/run_outcome，保留实际fit含refit、LLM请求数与training_profile，逐划分校验向量/指标。输出为**终态调试快照**：reserved_final_refit=0、provenance.executable_for_continuation=false，不能把它当还可训练的预算。正在运行、无outcome时拒绝导出；内部实时入口仍会给你正常实时预算。

examples/ab_v2/terminal_fixture.json将由A56从匿名模拟SSE run生成（以实际存在为准），含明确fixture_only标记，不能导入真实账本。真实五任务逐行向量不进Git。导出命令仍是：

`python -m tools.export_b_handoff --run-dir 已结束的AB目录 --output 新文件.json`

另外，examples/ab_v2/decision_fixture.json是匿名run实际state_2的历史决策快照，含新training_profile及当时尚未用完的预算；action_fixture.json是当前B纯函数对它返回的explore动作，供你适配“下一个试什么”。这比终态更适合看输入格式，但也标historical_replay/executable_for_continuation=false，不是现在可以重开的真实收费实验。内部入口直接给你这个形状的实时state，不需要从终态导出器伪造剩余预算。

## 现在最该补的方法和成本证据

A47同源目录Q显示：coupon14→15，系统accuracy .738916→.750246；photo7→8仍.975。preference22/22、SMS6/6、Bean5/5是完全相同目录，去重执行。偏好22个候选实际4个词字符路线，SMS6个是3路线各2超参，Bean5个只有logistic/extra_trees，不能把候选数量当强方法广度。冻结文本编码器本地没激活，本轮没有语义模型收益证据。

B下一批优先把**合法、已具备依赖与权重且真正能跑**的通用强路线放进共同清单，同时明确每动作内部监督fit数、时间/内存和语义限制。不要按数据集名字特判，也不要重复实现A26四个对称词法候选。未见H1/H2仍封存，等统一版本/方法部署规则冻结再开。

真实Flash首个A50路径11fit却花499秒；同版本地Q coupon full11fit约81秒。不能跨路径把它全归为API：深树模型选择与多次复核也花时间。你的实时budget已有elapsed/time_limit、history里有worker耗时，后续应共同设计“探索还要留出重训时间”和成本已知/未知的资源调度对照，而不是只按fit个数分配。A维护硬执行/交付守卫，B负责调度策略，先用提交交接共用字段。

## 网络与账本

获准执行环境下Python可能拿198.18.*代理虚拟IP，显式scripts.campus_direct只对官方域名使用现有10.121.10.250，Host/SNI/证书不变，不改系统/代理/hosts，不自动重试。必须在有校园网权限的执行环境启动；沙箱拒绝网络不等于学校挂了。真实models已200/19项，A50已有正常Flash usage。

真实累计SQLite不随Git。A50前快照224请求记录、known5.5403165元、原11unknown，之后持续增长；不能拿旧快照在另一机并发分叉计费，也不能导入A48模拟ledger。等A当前真实队列结束，用SQLite一致备份/请求ID核对交接最新账本。所有研究仍标课程总预算未核验、实际平台扣费未知，不需要用户反复提供无法取得的旧账单才研发。
