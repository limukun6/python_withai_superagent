# A114：保留原 H1 登记范围，准备训练侧输入

核对发现两套PAWS登记：A13已封存2000行池、outer61411/61421，各1599/1600训练和401/400留出；B另封存全49401行、outer45101/45102，各39502/39176训练。原包SHA相同，样本范围/分区不同，不得称为同一H方案或静默替换。

本批在A112–A114队列计划下作范围细化：优先准备A原2000行登记的两训练侧输入；B全量仅核对原包及名单元数据，暂不构造庞大worker。最终H1范围仍待共同冻结，未看任何性能，不能凭准备耗时改选数据。

仅读取原Parquet中当前outer训练IDs的sentence1/sentence2/label，使用pyarrow filters；不导出/解码对应outer留出行，不加载课程validation/test。原包字节hash和列级/行组底层扫描不等于把留出文本交给Agent。两个outer互有重叠，逐outer守边界，不能称独立数据源。

保留A原训练ID次序；新生成707/808/909训练内shared_text分组划分，固定LF清单并正式registered_partitioner/freeze_partitions核验。它们是本批明确新登记的内部分区，不冒充B全量或旧默认分区。输出两train.zip/inner_splits.json及六worker，0分类器fit/API/预测/outer评分。预计1–3分钟。full8均须preserves、一次fit含每次refit计数，不允许L12有方向候选进入对称任务。

复跑：强Python `-m scripts.prepare_a114_H1_inputs --source D:/codex/python_withai/runs/a13_H1_publisher/train.parquet --assets D:/codex/python_withai/runs/a82_encoder_assets --output NEW_DIRECTORY`。元数据归档，文本和worker忽略。全量/小型选择与性能运行另行冻结，不自动开启。
