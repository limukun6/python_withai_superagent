# A102：A线SMS完整六组，同c64bee4生产版本

2026-11-23，A起点1438。A负责单文本子矩阵，不等待B回复、不接管B建议任务。已有A28原始/核对两臂仅7fit且目录/源码不同，A77开启本地recipe提议，旧本地矩阵没有当前16候选，均不当本轮同身份复用，因此预登记6条新路径。

复用A47已登记SMS训练包3302行、原826行开发留出、macro-F1主指标和实际707/808/909内部分区。隔离工作树固定c64bee4生产源码，逐文件与已完成A89协议比对；MiniLM与非线性路线均可用，共16候选，所有组共用。模型由Agent自主选择；local-text-proposals、coverage、profile等额外研究开关off。

六组顺序L1/E0/R0/L0/R1/E1，不按已见留出成绩调整顺序或规则。每路径11实际分类器fit含最终refit、Flash最多3请求；trial300/path1200秒，最大66新fit、12新API。四付费路径每条估计停止阈值¥0.30，总¥1.20，非平台扣费硬保证。真实账本原sms_spam scope、incremental-research，保留历史11unknown；新增unknown/pending、意外其它写入或训练/请求失败整批停，不自动重试。A在此批为唯一写入方，若收到队友付费交接请求先协调串行，不并发。

计划/脚本先保存并推送，先独立--validate-only（0fit/API），实际另建目录；所有六模型冻结后才打开开发留出并统一评分，核对新进程预测、逐行差异、macro-F1/准确率、配置核验、fit、时间、tokens和估价。训练worker禁网，规划器仅按既有campus_direct访问官方学校API，timeout60。临时防待机随控制器退出释放。

预期15–35分钟，名义六路径时限合计120分钟加交付裕量；不是实测承诺。已见SMS仅是开发集，seed不是新任务，H/test继续封存。效果不论正负均归档；不为追收益换seed。脚本与protocol/results/completion可移交DeepSeek，数据/模型/权重/密钥/SQLite不提交。

复跑入口：用强Python在本工作树执行python -m scripts.run_a102_sms_C1 --reference D:/codex/python_withai/runs/a47_catalog_Q --assets D:/codex/python_withai/runs/a82_encoder_assets --ledger D:/codex/python_withai/runs/a23_real_local_history/ledger.sqlite --api-key-file D:/codex/python_withai/api_key.txt --output NEW_OUTPUT；只预检加--validate-only。NEW_OUTPUT必须不存在。
