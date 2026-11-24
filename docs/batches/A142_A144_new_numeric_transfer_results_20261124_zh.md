# A142–A144：新真实任务迁移完成，两组均280/280，无Flash提分收益

瓶颈：标签变换/故障快照和同历史请求不能替代真实新任务的完整自主开发与交付。结构沿现有94源的统一任务契约、共享合法目录、注册训练内分区、固定fit预算、完整重训与隔离预测；本批不修改生产规则、不按数据集名字选赢家。

## 来源与封存

任务为[UCI Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote)，Lohweg，DOI10.24432/C55P57，CC BY4.0。官方1372行/4连续特征/二分类，输入是预计算小波特征，不是原图像。项目既有docs/research/scripts检索未发现开发使用；公开数据可能在LLM预训练中出现，不能保证模型未见。只按输入/规模/许可选，不查基准性能。

先登记A142–144计划和runner，再下载官方ZIP一次（46562字节，HTTP200，约4.078秒）。原包SHA `1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227`，数据成员SHA `d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9`。固定outer71424，按完全相同特征分组：1092训练/280封存评价，原24重复行未跨边界，1348特征组中训练1078/留出270。没有实体ID，不能保证纸币实体独立。

运行器只给匿名train.npz和manifest，外层X做无标签查询，y另存私下joblib。707/808/909训练内实际行号先签名，两组payload和split文件SHA完全相同。默认full7合法候选，numeric-kernel/置信路由/参数诊断off。固定源在看新成绩前已确定；原H1/Task1/C1成绩不改标。

## 完整自主运行结果

L0＝local/ab-fixed；EC0＝checked-history/compact/calibrated-v2/ab-fixed。每条上限11fit含refit、单fit300/路径1200秒，同完整目录和预算。先L0后EC0，两模型全部封存后才统一解封外层，不依据L0成绩调整EC0。

| 配置 | 自主交付模型 | 外层答对 | accuracy / macro-F1 | 实际fit | API | 正式循环秒数 |
|---|---|---:|---:|---:|---:|---:|
| L0本地 | ExtraTrees leaf1 | 280/280 | 1.000 / 1.000 | 11 | 0 | 15.562 |
| EC0紧凑核对Flash | ExtraTrees leaf3 | 280/280 | 1.000 / 1.000 | 11 | 3 | 155.312 |

训练多数类参照为159/280＝56.79%。两个系统的外层逐行预测全部相同，改对0/改错0/净差0；不能因为模型参数不同就声称选择更好。两组都从相同线性anchor开始（内部accuracy .9680），之后分别试不同树配置及SVM，最终自主保留树。两组各4探索/6复核/1重训，无人工下一方法或赢家。

所有280行native/搬迁新进程/隔离源码包预测逐字节一致，模型重载评分相同；评估阶段0fit/API，18.828秒。实际队列22新fit/3Flash，173.297秒；EC0模型等待计API延迟140.766秒，已包含在其循环时间内，不能重复相加。单次顺序时间不证明稳定速度比例。

Flash输入15611、输出4250tokens，新增折算¥0.032611；本地0API。两组拟合次数相同，本次没有质量增益、没有训练节省，付费配置增加了等待和费用。这个任务本地基线已达本划分上限，因此不改种子/换数据集追Flash正收益，也不据此否认H1已经观测到的任务依赖增益。

## 结论边界与接续

支持：不加任务专用代码，两套预先冻结配置都能在一个新的真实数值任务上自主训练、选型和交付。不能推出任意任务满分、稳定100%总体准确率、Flash优于本地，或单独归因紧凑压缩/事实检查；本对照改变的是整体决策配置，A140才是同历史三视图机制试验。它是补充迁移T1，不替代B的EuroSAT H2或课程最终Unseen，也不晋级EC0为F*。

A累计1614→1636fit。实际原ledger372→375请求、known¥6.8218955→¥6.8545065、unknown12/pending0；原372条请求及全部旧transition逐项未改，不是新账本。原12未知按用户授权继续保留null，新增unknown/pending仍停队列；本次3调用全部成功。当前A持有空闲，最新owner在research/a142_a144_banknote_transfer/training/EC0/current_ledger_owner.json，旧372/363副本不用来写账。

代码：prepare_a142_real_transfer、run_a143_transfer_path、run_a143_transfer_queue、evaluate_a144_real_transfer、archive_a142_a144_transfer。协议/脱敏来源/内部行号/动作/计量/结果在research/a142_a144_banknote_transfer；数据、模型、密钥、SQLite和holdout标签不进Git。真实模型与独立包在本机runs/a143_banknote_transfer及runs/a144_banknote_evaluation。H2/Released2仍B负责，项目最终提交版本/test未冻结。
