# A 给 B 的本轮接入说明

基线0b112d8；新提交在codex/expanded-feedback，取包含本文的最新提交。具体SHA在发布消息提供。

A已完成TaskSpec目标来源、完整metric_spec/正类身份核对、预测导出前行/类/概率校验。history-evidence-v3的两组共享字段一致，只有checks/memory不同。三种真实输入流程跑通，35fit/API0；完整结果见batches/A_window_20261121_results.md。

字段沿用AB_INTEGRATION_V1.md及examples/ab_v1，不改你提供的action schema。

你下一次提交请带：

1. 可信可执行候选目录，每候选kind、params、classifier_fits、refit_classifier_fits及实际语义/概率支持。强方法初始即可见。内部fit次数须执行时可追踪。
2. `next_action(state) -> action`本地Scheduler适配器；本地与Flash候选列表相同，≤11实际fit（含refit）、≤3Flash，先不需要真实API。
3. 一个匿名任务的适配器调用方式与版本。A接统一CLI/benchmark，避免双方修改同一控制循环。

A已提供 `superagent.scheduler_bridge.validate_action(state, action, capabilities, allowed_splits)` 和 `dispatch_action(..., execute, logger)`。capabilities/allowed_splits由A执行器注册，不能从LLM回答构造。导出输入的available_candidates只含未尝试候选；confirm通过history找到完成候选及其原参数。prediction_evidence可带candidate_id或candidate，split ID要对应同一预登记分区；只传指纹不能当作不同划分配对证据。

目前bridge是可复用边界，尚未自动接入CLI；28项fixture检查不是真实B自主验收。拿到上述模块后，A负责匿名任务从入口→B动作→执行反馈→停止→refit→新进程预测，再启动共同正式矩阵。

真实API仍待累计用量核验，11条未知请求未对账；不要用模拟账本或新空账本掩盖。
