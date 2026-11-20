# A-05 执行协议

在三小时总计划下实施：36例六类、24开发/12预留。用例由runner的cases()一次生成并保存，输出只读标签；生成后不修改已冻结用例来迎合结果。
baseline固定为0b112d8的独立源码快照；fixed为本工作树。先开发集baseline/fixed，修复冻结后才开预留两组。预留由同一开发者编写，不是独立盲测。
对照合法误拒、错误接受、未知误确认。训练0/API0。概率缺列属于证据不足，现有MetricSpec拒绝执行时由probe标记unknown；这不是新增实现收益。
runner: scripts/run_a05_task_contract_batch.py，支持source-root/output/cases-file/subset/validate-only。
全部产物在runs/a05_*，不覆盖。source_changed不作为冻结结果。生产修复聚焦TaskSpec来源与输入约束，不修改B的方法和调度。
