# A112–A114：原生十二层整合、交付验收与 H1 准备队列

2026-11-23，按用户“开始”授权推进。保留原c64 C1独立checkout，不重跑旧性能矩阵，不改F*晋级规则。全程不调用学校API，不写共享费用账本，不打开H性能。

## A112 原生整合

从A当前5fde567建立隔离分支，复用B c658477中ab_sentence/ab_methods/ab_catalog/external_assets/ab_edit_views，CLI只接原生standard/deep/deep-edit，不引入preference-components控制器。保留A98/A99审查修复。修正CLI标准模式可能继承ambient deep环境的问题：显式为每次A/B调用设置和恢复目录，记录manifest。标准目录默认不扩张；deep与deep-edit是可选研究能力，按语义契约和校验资产出现。

先运行定向无监督拟合测试：原有核特征、外部资源、视觉资格和新增目录作用域/交换语义/深资源识别。禁止网络及分类器fit。新增回归必须覆盖标准模式不受前一次运行环境污染、成功/失败恢复、relation/symmetric不泄漏preference候选。测试过后用A111同环境菜单核对脚本比较默认目录。

## A113 资产与真实交付探针

本机尚无L12权重。固定B资产声明：sentence-transformers/all-MiniLM-L12-v2，revision52f9d2dd711ffc3434e6cff17284b13a11629a90，三文件大小/SHA按B ab_sentence.py，不变更校验值。复用同SHA本机tokenizer；其余文件从官方固定revision下载至忽略目录，不改系统代理。timeout60，单文件/分段失败即停、不自动重试，公开下载总窗口1800秒，优先4MiB Range分段避免长流中断。下载不算模型API。

资产ready后做小型匿名合成偏好任务的原生fit/保存/新进程预测/可搬移源码包验收，最多4次真实监督拟合，API0，15分钟墙钟。它仅证明新候选与交付能执行，不用于真实性能/通用性结论；实际fit计入A累计。若下载失败，保存原错误与0fit，并继续独立H1准备，不重试资产。

## A114 H1登记核对与训练侧输入

先厘清A13小型登记与B全PAWS登记的范围：不得将不同样本/seed当同一H方案。优先核对B已封存49401行原包SHA、两outer名单及六inner名单。原包已在本机a13_H1_publisher/train.parquet；只构建outer训练侧文本，不导出或评分outer留出，不调用候选。规范化清单固定LF。记录规模/方法合法性/分组隔离，再准备正式CLI训练前截停验收与启动模板；全PAWS规模的耗时未校准，不沿用小型任务的20分钟路径预测。

若旧登记规模不适合现有预算，仅报告并登记待共同决定；不为了跑通而自行抽样重封。PAWS不是MRPC开发任务，不借用MRPC成绩作为H证据。

每阶段先落盘可复跑代码、protocol/results/completion，失败也保留；代码合并须定向检查通过，资产/模型/训练包/worker不提交。预计整合30–60分钟，下载最多约30分钟，工程交付探针5–15分钟，H1准备20–40分钟（实际记录）；本轮预算最高4新fit/0学校API。独立48/24及Task2/Bean正式C1仍另待B交接，不以等待阻塞本队列。
