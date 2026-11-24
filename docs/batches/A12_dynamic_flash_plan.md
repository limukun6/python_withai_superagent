# A-12：今日A线动态Flash接入与验收（2026-11-21）

从0418ae1继续。今日A线交付范围：六组（本地/原始历史Flash/核对历史Flash × 固定/动态）可从正式CLI运行；动态历史正确绑定划分；共用候选和11fits/3请求；真实客户端SSE/累计账本/失败停止与模型交付有端到端验收；自然图像和H登记另批推进。未对账的真实API不在本批执行。

## 接入设计

raw-history/checked-history在ab-v1/ab-fixed中均使用同一可执行候选全集，第一候选本地共享；之后最多前三个新候选提议调用Flash，复核不调用，额度耗尽后使用预登记的同一本地规则并日志标记，绝不因为请求失败走本地兜底。已有模型/观测/账本保留，失败停止整条路径，无重试；付费失败不伪装本地成功。

新决策上下文版本ab-history-v1，显式传每条trial的split_id；不含逐行原始输入/标签、外层结果、机器路径、墙钟细节。两组基础字段完全相同，checked仅多checks/memory。候选配置事实来自可信能力目录。提案严格JSON，必须合法候选/任务类型、引用已发生trial、与可信classifier一致；数值预期仅search_score且只能引用completed primary，与确认划分的成绩不串用。所有结果都记录metric identity、配置核对、期望核对；解释核验不当因果证明。

累计真实账本须已导入历史、有实际记录、无未知/待决、同任务/共享限额足够；不能用explicit_new_project/空账本绕过已有11条未知。模拟验收单独目录及明确fixture历史，不进入真实账本。客户端请求时限沿用已有60秒读取上限、直连、失败不重试。

## 小批验收与资源

1. 无训练软件检查：候选限制、reference split、严格字段、任务/配置、R/E公共上下文一致、不暴露数据/时间、anchor/confirm/stop不请求、最多3请求、已用满3后本地动作清楚记录、无真实账本时加载key/网络前拒绝。零实训/真实API。
2. 匿名合成输入复用B1 fixture种子932（256行训练、31行无标签）。四条路径raw/checked×ab-fixed/ab-v1，经正式CLI、真实ExperimentLLM的SSE/真实SQLite账本逻辑，用mock HTTP固定选择共同eligible首项。最多44真实fits、12模拟请求；每条240秒/worker60秒，模型新进程及搬迁预测一致。R/E可能轨迹相同，这是接入验收，不证明模型推理。
3. 三个故障路径：timeout、usage缺失、无效JSON，各在anchor后第一请求停止，每条1fit/1模拟HTTP；用相同scope下一条请求须被未知账本挡住（前两例）。无效JSON用量已知仍终止本路径。cap合计47fits、15模拟HTTP、真实API0。
4. 对称/自然图像后续批次必须先单独落计划和数据资格，不拿合成验收作为迁移证据。本批代码冻结后才运行；失败保留输出，修复后新目录重跑，资源累计。

runner scripts/run_a12_dynamic_flash_acceptance.py；输出protocol/results/completion与逐worker实际fit账本，紧凑归档research/a12_dynamic_flash。完成后记录耗时、fits、模拟/真实请求、仍缺内容，推送GitHub。不为了追求分数额外调用或调参。
