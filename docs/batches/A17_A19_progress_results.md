# A17–A19：照片Q零分数增益，C1五任务输入已齐（2026-11-21）

基线9b4fc18；每批预登记与runner分别见A17_photo_catalog_Q_plan、A18_C1_readiness_plan、A19_task_readme_contract_plan。H1/H2未运行模型或打开成绩，真实API0。

最终脱敏归档research/a17_a19_progress_final；初版归档仅本机runs/a17_a19_archive_initial保留，避免旧矩阵命令被误当最终配置。

## 瓶颈：可执行方法目录增加，是否带来实际收益？

**结构：**同一正式CLI新增--catalog-scope baseline/full，用A执行器从B可信目录投影；baseline只保留shared_baseline，full增加B移植方法，默认full。B目录/训练器/调度/晋级规则未改。Q两组同源码、同实际行ID、同初始候选、同资源上限，仅候选集合变化；不按数据集名称路由算法。C1六组继续统一full。

**实验：**已见Olivetti开发任务，baseline7候选、full8候选（新增梯度直方图）。同320/80照片、内部707/808/909、300秒/worker60秒、11fits上限、无缓存。两模型全部封存后统一评分，原目录与搬迁目录各由新进程预测，均逐行一致。

| 目录 | 赢家 | explore/confirm/refit | 实际fits | CLI秒 | 外层准确率 |
|---|---|---|---:|---:|---:|
| baseline | 像素logistic C0.1 | 5/4/1 | 10 | 22.746 | 78/80=97.5% |
| full | 梯度直方图logistic | 8/0/1 | 9 | 18.488 | 78/80=97.5% |

**结论边界：**分数差0，不能宣称扩展方法提高准确率。两组3行预测不同，其中full独有正确1行、baseline独有正确1行，另一行两组都错。full在本次轨迹少1fit并更快，但仅一个小型闭集人像开发划分；没有稳定提速或任意自然图像泛化证据。新增方法可达/可执行/可被选择已验证，强方法增益尚未建立。

Q子矩阵共19真实fits、独立观察器也是19、0HTTP，56.029秒；数据/源码均未变。外层80行的一行是1.25个百分点，不做无重复支持的显著性断言。A13旧9fit路径没有充当本批full组，两臂均按本批同版新跑并计账。

DryBean/SMS的baseline与full在本机逐项相同，0新增训练，标deduplicated_identical_catalog；不能把它们重复两遍凑Q样本。类别表格/偏好文本Q仍由B接续，完整五任务最终Q未完成。

## 瓶颈：旧数据适配器丢掉任务明示目标，完整入口因此缺方法

A18首次预检发现：Task1准确率只有默认来源；Task2的22个合法偏好方法因缺标签语义而不可见。我们读取包内README确认Task1要求报告test accuracy，Task2明确1表示headline_A点击率更高/0表示B，亦要求test accuracy。没有读取正式test或新打开validation数据。

**结构：**A新增保守的legacy task README声明读取器，仅唯一README.md/txt、忽略__MACOSX，记录member/SHA256/规则/短原文；只识别明确正向声明，否定/冲突/多个README/反向标签维持未知。manifest优先，不以任务名选择算法，不执行README里的代码或任意指令。该规则是有限声明语法，不等于完整自然语言任务理解。

`load_training_task`与旧development入口均保留来源；goal_resolution能区分declared_task_readme与默认accuracy。正式manifest在训练开始前保存task_description_source，即使训练失败也保留声明来源。五项新反例检查及原有回归均通过。

## C1就绪配置：五任务×六组，尚未执行正式矩阵

最终目录runs/a18_c1_readiness_current，源摘要对应本批最终实现。所有任务输入/目标/目录预检通过：

| 开发任务 | 初始合法full候选 | 累计费用scope |
|---|---:|---|
| 优惠券类别表格 | 15 | task1 |
| 标题点击率偏好文本对 | 22 | task2 |
| DryBean数值表格 | 5 | dry_bean |
| SMS单文本 | 6 | sms_spam |
| Olivetti照片 | 8 | olivetti_faces |

每任务六组L0/L1/R0/E0/R1/E1共享full目录、指标、真实划分和11fit上限；Flash每路径至多3请求。生成30条配置，其中10条local为prepared_not_executed、20条Flash因历史11条未知用量为blocked。准备/运行状态明确区分，formal_experiments_completed=0，不能把该清单当C1结果。

准备阶段曾使用*-dev费用ID，审阅后发现与A2的既有scope不一致，已修为上述固定ID，A16默认也改sms_spam。旧准备命令未产生真实API，已作废；不因此宣称历史费用已对账。最终清单保留原训练包/输入文件/README摘要、源指纹、真实行ID与依赖环境；B接入改源码后需重新生成。正式D5冻结前仍属开发配置。

## 本批资源与验证

- 初始catalog回归81项通过，4.070秒，0分类器fit/HTTP，1次预处理fit。
- 任务README回归86项通过，3.903秒；补充早期manifest声明来源后86项通过，3.860秒；再加“No requirement/疑问/可能”反例后最终86项通过，各0分类器fit/HTTP，各1次预处理fit。结合前述81项检查，本批4次预处理fit。是代码变化后的回归，不作独立实验样本。
- C1五次准备（发现缺口→补声明→修scope→审计来源→保守声明最终源摘要）均0fit/API，保留历史；最终10条本地/20条付费配置，无实际执行。
- 本次新增19监督fits，A窗口261→**280fits**；模拟HTTP仍25，真实API0；历史及队友资源另计。Q发生在README声明修复之前，保留当时源摘要，不冒称最终版本全实验同源。

## 下一步

给B：当前完整目录的照片增益为0分，应保留；其他三种类型的补方法/可用依赖需要独立效果证据，目录相同的臂去重。H1合法对称方法仍待B，不能用标题偏好方法代替。我们A继续任务/证据/入口与图像交付。

真实账本核验后先跑已写A16的动态raw/checked小试，再按统一最终代码/候选/实际分区执行C1；不新建空账本，不重复同任务随机seed追求正收益。H保持封存，D5唯一部署策略冻结后再按计划评价。完整Q/C1/H/Released尚未全部完成。
