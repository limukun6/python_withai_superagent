# SA14：无manifest数组与图像的语义契约

基线dee7140。A维护任务/证据/reader/导出，B方法目录和选择规则不动。先保存计划与可复跑代码，再运行；不覆盖SA13结果，不直接对课程测试调优。

## 瓶颈与结构

proactive全局语义编译只能描述标量记录；原生数组/图像即使本地可读，也会被该入口拒绝，或只在有manifest时允许输出overlay。新增semantic-native-v1数据契约，LLM根据原文选任意安全成员/数组键或图像根目录、标签来源、指标、正类、图像处理与输出规则；Python用已安装npy/npz/image_folder读取器实际执行并冻结预测recipe。不是按任务名分流，不用Python补指标或目标。

JSON plan固定字段modality/training/prediction/metric/positive_label/image_preprocessing/output；evidence对应非空声明必须引用真实文档ID；uncertainties未解决不得编译。描述符白名单只代表可执行工具能力：npy特征/标签路径、npz特征/标签键、image_folder类目录根。无新文件解析器或任意生成代码。显式manifest仍权威，不能被新plan覆盖。

只读取训练选择，拒绝训练指向test/validation与label向量当预测特征；数组allow_pickle=False，预测不打开标签，shape/kind/decoder契约不匹配在fit前报错。新契约和reader加入artifact指纹，恢复和新进程预测共用同一recipe。

## 小批实验与预算

1. 零fit/零API：npy与npz非标准文件名、NPZ键、图像非标准根目录/尺寸/颜色；主指标和目标来源、预测重放与跨进程、artifact恢复；误用eval路径、伪造引用、任意额外字段、显式manifest覆盖、预测shape/正类错误拒绝。和既有229项相关检查一起跑，先登记runner协议。
2. 准备两个公开数据开发封装（数组数值、图像文件夹），不把重新封装当新任务迁移。封存文件/输出规范/期望训练行数；预测答案留在包外且本批不读取。计划/脚本先落盘。
3. 全部离线验收通过后，两条正式standalone solve：各最多2真实fit含refit、3真实Flash、¥.25估计停止阈值，最多合计4fit/6API/¥.50，墙钟各240秒。stdin关闭，无client/resolver/算法/工具注入，B自主选择。不得循环试到成功。
4. canonical ledger起点475/已知8.067821/旧未知16/待处理0；显式按原行SHA继续授权清单，每路径独立scope。新unknown/pending停止付费批次，失败不重试。原行/费用null保留，预算合规仍未核验。
5. 分别报告：读取/语义完整性、真实CLI训练/交付、实际fit/API/耗时；不打开外层标签比较分数，不声称任意未知成功。

预计实现/离线30–60分钟；真实小试每条约1–4分钟，最多约8分钟。脚本scripts/run_sa14_native_checks.py、scripts/run_sa14_native_delivery.py；新输出research/sa14_*，模型/数据/账本/密钥留忽略runs。
