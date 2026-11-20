# 三小时窗口受影响回归

五个suite：task_goal、architecture_contracts、general_extensions、confirmation、image_files。
预计真实worker训练21次（1+12+8），上限24；wrapper在test模块导入前包裹benchmark.train_candidate，逐次记录真实worker返回状态与classifier_fits。模拟worker不计真实训练。
另有一项仅拟合文本预处理，不是分类器。真实API0、模拟HTTP0。预算独立于49次性能/流程实验额度，但在整个窗口资源汇总相加，不漏记。
执行前冻结生产源码，在run_a_window_regression.py输出protocol/tests.log/worker_calls/results/completion。
失败保留，修复后新目录重跑；不是每次小编辑都重跑整个仓库。

最终补充验证：predict导出检查及history-evidence-v3投影发生变化，另运行tests.test_checked_cli（预计17真实fit/5模拟HTTP）与tests.test_feedback_loop（0fit/HTTP），沿用run_a1_cli_batch.py --phase fixed记录源码/请求/资源。此为新增17fit回归预算。另重放A-05/06及A-08仅作最终版本回归，不再称预留盲测；API0。不重复原35fit性能路径，最终导出器已在旧冻结模型上三类型重载通过。
