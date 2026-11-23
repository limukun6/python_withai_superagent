# 给B：当前源的真实训练行与分区交接

**交接状态纠正（2026-11-23）：本文件是建议执行包，不是队友已确认接下任务的记录。用户说明尚未要求队友跑这些具体实验。** 下文分工与实验数量源自A侧拟定计划；不能直接按已在跑询问完成情况。应先对齐对方当前工作及承接范围，避免重复或越过用户分配。24份语义标注属于后续验证安排，不是已确认交付义务。

交接代码与输入提交c64bee4（codex/expanded-feedback），**本轮C1请固定这个提交的生产内容**。A97已完成：六组均87%、逐行预测全同，53新fit＋10复用、12成功Flash、18.18分钟、估计¥0.097466。A无活动付费队列，已停止写账本；真账本当前323/known¥6.3627855/历史unknown11/pending0。B付费前明确交接唯一原账本及写入方，不能各写副本。本地组可独立推进。

A97结束后A98/A99修复了两类检查器误报，最新HEAD的生产source因此已不同。B本轮不要直接用最新HEAD冒称与A89/A97同源；用c64bee4完整source与A94登记数据完成当前矩阵。后续冻结H或研究新修复时再统一升级，保留旧测量版本。

老师已允许通用下游预训练特征，说明见TEACHER_ENCODER_CLARIFICATION_20261122_zh.md。共同superagent文件内容与A88/A89冻结源一致；不要用旧A31控制循环覆盖。A89 Coupon六组完成，结论见batches/A89_C1_results_20261123_zh.md；原始/核对动态同分，固定核对反而少对4条，不默认晋级反馈。

**优先接Task2与Bean整组实验，别先重复A的Coupon。** research/a94_portable_joint_inputs提供：

- preference：8301训练行，full32/baseline22，稳定费用ID task2。
- bean：8710训练行，full5/baseline5，稳定费用ID dry_bean。
- coupon：8117行，full15/baseline14，task1，仅供共同输入核对。

每任务registered_rows.json是你原tools.ab_registered_rows的v1格式，含真实train/search索引、行序/标签摘要，707/808/909三分区。三任务九分区已用原registered_partitioner反向验证，Coupon与正在用的A88/A89实际索引吻合。不是只说“seed一样”。results.json给完整TaskSpec/能力与成员SHA；ordered_training_row_sha256.json给有序逐行内容哈希（含标签，无原始文本或特征），可从完整原始数据重建有序子集，重复行按次数匹配。

你旧6629标题/8714 Bean不能直接与此拼表。先对齐training_content_sha256、目标域、完整TaskSpec、实际索引及候选参数。zip时间戳可不同，但成员和加载内容须核对；容器相同也不能代替内容一致。B注册工具会把split_resource注释改成pre_registered_source_rows，可比较实际索引/行标签摘要，不因注释不同误判数据变了。

正式入口可复用你已有的包装：

```powershell
python -m tools.ab_registered_run --split-manifest "research/a94_portable_joint_inputs/preference/registered_rows.json" --split-sha256 "<results.json里的registered_rows_sha256>" -- run "MATCHING_TRAIN_ONLY.zip" --scheduler ab-v1 --portfolio ab-v1 --fit-limit 11 --time-limit 1200 --trial-timeout 300 --seed 707 --run-dir "NEW_RUN"
```

该工具仅接已登记分区，仍由同一CLI执行。依赖/资产须与environment记录对齐；Task2强路线无资产时会少两项，不能把30/32目录混作同组。优化输入是完整8301，外层评价用原包指定val，不能悄悄换成你旧从训练另切的1672。evaluation_registration.json仅提供既有外层来源摘要，没有提前打开新标签；模型全部冻结后才统一评分。

六组L0/L1/R0/E0/R1/E1用相同full菜单/数据/实际预算/300与1200，calibrated-v2统一于Flash组，其余研究开关off。API使用同一非空真实项目账本，保留原历史未知和task IDs；不同电脑不能新建空账本或同时各写一份并当累计已对齐。先跑本地组不受影响，付费前由项目持有人交接唯一真账本/正式审计导入，不通过Git发布SQLite或密钥。A这轮付费已结束，后续交接要明确唯一写入方。

图像后续用花卉/场景对象分类，不再扩展旧Olivetti身份任务。H1/H2与正式test继续封存；独立48例和24语义标注按原格式补齐，软件回归不能顶替。A94仅准备数据身份，8.391秒/0fit/0API，不冒充迁移效果。
