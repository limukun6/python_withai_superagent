# SA13：字段探查来源与输出JSON结构

基线a5bd0b9。SA12两次API均已结算，但完整交付失败；本批保留原失败，不重复同scope，不把开发修复算独立泛化。

## 结构改动（A）

1. inspect_header省略source时，仅按已公布member在物理包中的唯一存在性解析来源；显式source严格遵守，同名存在于两个包则要求LLM明确角色，不读取内容猜测。单包保持原兼容行为。日志记录原source、解析来源及只披露表头。
2. 输出JSON提供结构化条件schema：input_id必须有field，probability必须有class_label，row_index必须有start；label mapping不由程序填写。LLM仍负责读文档/探查字段/选正确字段，Python只验证事实和执行，不猜ID、指标、标签或目标。
3. 保持共享最多3请求/2工具/2拒绝、失败不重试；不为案例加文件名特例。不改B部分交付语义/候选/调度。

## 实验

- 零fit/API：原SA12实际省略source动作本地回放得到真实表头（仅回放，不是真实模型续答）；外部唯一、内部唯一、重名歧义、显式错source、伪造引用、标签向量拒绝，schema到达实际payload；加既有223项回归。先落盘scripts/run_sa13_tool_source_checks.py。
- 通过后另冻live协议，复用已公开开发V1任务从正式solve重新读题，最多3Flash/2fit/¥.25估计停止阈值、240秒。使用新scope和同真账本473/16未知/0待处理；不注入工具请求/模型答案，不开外层标签。
- 判据：真实工具是否自然激活单独报告；若激活则核对真实后续提案是否用了返回字段。输出完整契约、实际refit、独立进程60行预测。若未激活不能称已证明真实工具闭环。
- 结果可为失败。新unknown/pending停止付费，原费用保留null；本批不追加循环试到成功。只用一个已登记新scope。

证据research/sa13_*；运行目录runs/sa13_*；不提交数据/权重/账本/credential。SA12原223项及失败证据不改写。
