# A-20：五任务C1本地子矩阵（2026-11-21）

运行前登记，生产基线506441c。用户无法取得学校历史记录，明确要求不停工；本批不依赖人工对账、不发真实API。真实历史继续unknown，不能把它填0。另批A21审查本机可恢复证据/保守上界。

## 当前执行范围

五开发任务×L0固定/L1动态，共10完整路径；这是开发版C1本地子矩阵，不是D5最终C1六组。H1/H2模型和成绩封存。其他四付费组不作本地模拟成绩替代。每条11实际fits含refit、300秒/worker60秒，批次≤110fits，0API/模拟HTTP；预估5–12分钟，上限50分钟训练+3分钟交付。

## 数据和公平条件

复用a18_c1_readiness_current的源码/目录/实际行ID。DryBean/SMS/照片沿用A07/A13封存训练和外层留出。课程Task1/Task2只把原train成员与唯一README复制进新训练包，C_train/y_train或train.csv字节完全相同，不复制原validation/test。打包后TaskSpec/能力目录/实际内划分必须与预登记相同。

两课程任务的外层评分使用原指定val（不是test），只在所有十模型冻结后读val特征/标签并生成无标签预测包；正式test成员不打开。其余三任务的外层holdout也在全封存后才读取标签。五任务所有方法参数/特征/阈值都不根据外层成绩更改。

L0/L1同full目录、anchor、探索规则、晋级规则、同样CPU线程/缓存off；只改变探索/复核预算分配。顺序预登记：coupon L0→L1，preference L1→L0，bean L0→L1，sms L1→L0，photo L0→L1。后续不得按成绩改顺序/seed。source/caps/真实train-search行ID必须一致；失败即停，保留物理fit账本与所有已封存模型，无自动重试。

## 产物与验收

scripts/run_a20_local_c1.py，支持--validate-only仅冻结数据/配置（0fit）。运行前protocol包括代码/runner/原文件摘要、每条命令/配置、预登记划分指纹。每路径独立worker前限额和原physical观察器对账。十模型全封存后正式predict新进程；独立验证进程核对模型/行ID/预测并评分，每任务两组逐行比较。模型/训练缓存/数据只在忽略runs目录。

报告主指标、失败/完成率、实际fits、CLI/交付秒、候选及配对预测差，不将accuracy和macro-F1混成一个总准确率。单划分不能证明统计显著性、普遍迁移或稳定提速。两组若同模型同预测，报告无分数收益；不加seed追涨分。完整Q、四Flash组及H仍需后续各自证据。
