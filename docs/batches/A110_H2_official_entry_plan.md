# A110：H2 正式入口训练前验收

2026-11-23。原计划先完成五任务 C1、选定 F*、冻结共同版本后再打开 H 性能。当前 Task2/Bean 本轮仍0/12，远端没有 B 报告的9ed50e6，所以本批不提前运行 H 模型或选择F*。本批完善可复用启动配置，排查“分区清单正确但正式入口偷偷重新随机划分”的接入风险。

输入仅为A109已验收的两训练包/两清单。每外层分别走 L0/L1 正式 CLI + tools.ab_registered_run 的真实分区包装器，在 freeze_partitions 完成后、Executor 创建前以专门 BaseException 截停。要求每次 CLI 生成的候选、源文件、实际六分区的 worker 内容SHA与 B 登记一致；禁止分类器fit/网络。预期4次训练前截停、0fit/API/预测、0留出图片访问，预计1–4分钟。

其余四种 Flash 方案只解析/validate_options，明确是8份参数配置检查，不用模拟账本冒充付费入口已验收。输出12份包含注册分区启动包装器的命令模板，凭据/账本用待填本机路径，不读取密钥或账本。全部模板在共同冻结后仍需核验source、F*晋级、唯一账本写入人和预算；不自动执行模板。

后续正式 H2 每外层 L0/R0/F* 去重，合计4–6路径，最多44–66fit、6–12Flash，每付费路径估计停止阈值¥0.30，总估计¥0.60–1.20。沿原协议每路径1200秒/每fit300秒是待冻结方案：4–6路径墙钟上限80–120分钟，加导出评价预留10–15分钟。它是上限预算，非基于H性能调参，也不是已测实际耗时。不按H的成绩改变策略。

本批执行：强Python在integration下 `-m scripts.check_a110_H2_entry --inputs D:/codex/python_withai/runs/a109_H2_canonical --assets D:/codex/python_withai/runs/a82_encoder_assets --output NEW_OUTPUT`。输出protocol/results/completion，Git只归档元数据/模板/运行清单，不上传worker或图片。未完成的前置条件写入results，不伪称性能实验已启动。
