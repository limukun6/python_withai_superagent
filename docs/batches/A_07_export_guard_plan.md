# A-07 补充：导出边界故障注入，0新增训练

正常三模态交付已完成。审查发现predict.create_predictions直接写出predict_with_model返回值，尚未统一核对行数/标签域/概率列；接入B的新候选时会影响完整交付。
复用a07已冻结digits模型与对应外层无标签features包，不改模型或标签。通过mock注入预测输出：合法、少一行、未知类别、二维输出、NaN。另复用同模型写入本地测试副本的roc_auc契约（仅作导出接口夹具，不声称模型按AUC训练）并注入缺概率行、错类别列、非归一化概率、合法乱序类别概率四种，共9例。
先固定基线0b112d8，再新版本；每例新目录，失败保留。预期合法2例通过、非法7例在CSV创建前拒绝。执行器/model对象真实重载，模型输出由受控替身提供；拟合0/API0，不增加性能证据。
runner scripts/run_a07_export_guard_batch.py --source-root SOURCE --run-dir DIGITS_RUN --features FEATURES --output NEW_DIR。

修复后用scripts/run_a07_export_recheck.py对已完成三个fixed模型运行新导出器及独立进程核验，不重新训练、不重新评分外层。新导出代码源码指纹独立记录；旧训练实验仍属于其原指纹，不能冒称在新代码上重跑训练。
