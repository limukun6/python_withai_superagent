# A-07 执行协议：两任务配对与真实小图像完整交付

冻结当前A-05/06实现后，基线0b112d8与新版本分别跑DryBean/SMS，digits只跑新版本，顺序bean基线/新、sms新/基线、digits新。固定2搜索+4确认+1refit，每路径7 fits，上限35。共同目录expanded、anchor catalog、local interleaved_routes、seed707、60秒单fit、300秒整路径；无LLM、无人工挑候选。
只打开原包train；按grouped_partition seed45101划出外层，保存原始训练行索引。CLI包只含内层可用训练，不含外层标签或正式validation/test。digits来自本机sklearn内置真实8x8手写图像，以image NPZ输入，不当自然图像或封存迁移结果。
统一CLI完成后冻结模型，再在两个独立新进程做CLI predict与导出核验，比较逐行预测和模型哈希，外层只评价一次。原包与外层标签不用于选型。保留类别序、行索引、候选目录指纹与源码指纹。
runner: scripts/run_a07_delivery_batch.py，prepare/run两阶段，支持validate-only。基线与当前目录为独立源码根；当前源码在整个batch内禁止修改。父进程硬超时终止自己的子进程树，未知fit记unknown，停止本批而不是继续耗预算。失败不自动复跑。
如无生产变更影响输出，预期两真实任务分数一致，这属于兼容性证据，不宣称分数改进。
