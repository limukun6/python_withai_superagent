# 今日A线：正式六组入口、真实照片与迁移数据登记（2026-11-21）

从0418ae1继续，完成A12/A13/A14/A15，A16真实调用协议与runner已交付。源码变动范围为A所属执行循环、Flash适配器、固定分配回调、对称CSV契约；B方法/动态调度/晋级规则未修改。每个小批先计划，原始输出位于主仓库runs目录，紧凑归档research/a12_a16_day。

## 1. 反馈机制和预算调度终于使用同一入口

**瓶颈：**原来raw/checked反馈能在旧固定流程运行，B调度能在新本地流程运行，两者未连成完整系统；混合历史还可能把不同划分的结果串用。

**结构：**ab-v1动态与ab-fixed固定均可使用local/raw-history/checked-history，形成六种正式CLI组合。第一候选共用本地规则，之后最多三次新候选提议由Flash提供；确认/停止不请求。额度用满才按预登记本地规则继续，API失败立即停，不走兜底。ab-history-v1显式记录split_id，数值预期只能引用completed primary；两组共同上下文相同，checked仅增加checks/memory，原始标签/逐行输入/外层成绩/机器时间不发给模型。

**实验：**匿名合成任务，通过正式CLI、真实worker/SSE解析/SQLite账本逻辑，HTTP完全模拟。四条路径各11fits/3模拟请求，分别完成固定/动态×raw/checked；每条模型在两个新进程（含搬迁目录）对31行无标签输入的预测一致。R/E每个请求的共同字段逐项相同，合法目录与fit额度共用。三条故障路径（timeout、missing_usage、bad_json）各在anchor的1fit后第一请求停止；前两条未知用量会阻止同任务下一调用，bad_json保存已知模拟用量但仍停止当前路径。

合计 **47实际分类器fits、15模拟HTTP、0真实API、106.462秒**，7场景通过。不是Flash真实推理或效果实验；四成功路径都是模拟确定性选择。

运行前接口首轮46项因SQLite连接未显式关闭导致Windows临时目录清理失败，修复为显式close后新目录46项通过（3.986秒，0fit/API）。失败记录保留。它是软件资源释放问题，不是网络故障。

## 2. 真实照片从文件输入到模型交付

**瓶颈：**此前图像只有digits数组输入证据，不能据此主张真实照片文件流程。

**结构：**AT&T/Olivetti 400张64×64灰度人像、40个闭集身份，原文件按sklearn登记SHA256校验。PNG＋CSV声明文件格式，通过正式image_path解码；固定外层45101留出80张，内部707/808/909，任务/目录/划分在训练前封存。标注规则是已有身份分类，不是未见人物识别。

**实验：**一条正式动态本地路径，8探索＋0确认＋1重训＝**9fits**，正式CLI18.040秒、训练子进程19.874秒、交付5.064秒，全批25.173秒。自主选择 **image_gradient_histograms_logistic**；模型封存后留出准确率 **78/80=97.5%**，搬迁目录新进程CSV完全一致，API0。

这次新图像方法既实际训练也实际被选；没有固定/动态或新旧方法的外层对照，不能将97.5%归因于Agent创新。40类小样本人像不代表任意自然图像高效果。它成为开发任务，不再作H。

数据来源与署名：[Cambridge档案](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)、[sklearn数据说明](https://scikit-learn.org/stable/datasets/real_world.html#the-olivetti-faces-dataset)。使用时署名AT&T Laboratories Cambridge；不额外声称其具有CC许可。

## 3. 同义判定的标签契约修复

**瓶颈：**pair_text CSV读取器把标签1强制解释为left，不能正确表示“两个句子同义”。

**结构：**明确声明pair_semantics=symmetric、pair_label_one=equivalent，0/1按不同义/同义加载；symmetric+left/缺失声明在训练前拒绝。原preference+left继续兼容。B方法未补时正式入口明确无合法方法、0fit，不把错误翻标签方法用于对称任务。

25项输入/目标/语义检查通过（计数版0.299秒）。首轮runner将既有DummyClassifier夹具的1fit误记0，审阅后保留原记录并明确更正；计数版重跑另1fit，**累计2次dummy回归fits**。它们不是模型性能实验。

## 4. 两份H任务完成登记，保持未评价

| 项目 | H1：PAWS-Wiki同义判定 | H2：Rice Leaf Diseases照片 |
|---|---|---|
| 来源 | Google Research发布者的labeled_final train Parquet | UCI id486原始图像压缩包 |
| 登记样本 | 2000行，0类1105/1类895，完整文本连通组抽样 | 全120张，3类各40，固定32×32 RGB |
| 原文件SHA256 | 8dc9ad3e5f30ad9a86b290fe236d528ef23a5751fec9a35d99cbacf68ba277cf | 39b572e6fcef37c2c6ac6be22b09e337c996ebbc86fa9ef7d5bfe3c314c69d5d |
| 种子61411/61421实际划分 | 1599/401、1600/400行 | 两份均96/24行，实际行ID不同并登记 |
| 标签交换 | preserves，1=equivalent | 不适用 |
| 输入/方法就绪 | A解析/任务包已就绪，B合法候选0，待接 | 任务包就绪，当前8候选可见 |
| 模型/成绩 | 未训练、score=null | 未训练、score=null |

均已生成统一CLI可读任务包并保存实际划分ID；数据包留在本机runs/a13_holdout_packages，不提交原文或照片。PAWS官方dev/test未下载，publisher train的字节摘要已核验，未与不可下载的原tar声称字节等价。H2只有精确重复图隔离，缺植物/拍摄会话ID，不能保证田块/实体独立。

PAWS原Google下载直连/现有代理均403，保留失败后显式修订来源为同一发布者的[训练文件](https://huggingface.co/datasets/google-research-datasets/paws/tree/main/labeled_final)。其[原始许可](https://github.com/google-research-datasets/paws/blob/master/LICENSE)允许自由使用并鼓励Google署名，不误称CC。Parquet库pyarrow26.0.1仅装入忽略的隔离准备目录，未改主venv或Agent依赖。

H2[UCI官方来源](https://archive.ics.uci.edu/dataset/486/rice%2Bleaf%2Bdiseases)为CC BY4.0；署名Shah/Prajapati/Dabhi，DOI10.24432/C5R013。直连过慢部分下载保留并停止，同源经既有代理完成，未改系统配置。H1准备3.030秒，H2最终下载/准备126.999秒；前次失败/慢速等待另存，不能把这两个数当总耗时。

## 5. 最终回归、真实调用门槛和今天完成范围

收紧Flash入口：非空导入历史，当前task/shared及全项目无unknown/pending，才允许读取key/新请求；新task ID也不能绕过别的历史未知scope。最终 **78项检查通过、4.105秒、0分类器fit/HTTP**，含1次预处理fit。此前77项通过后增加此反例再跑，均保留；预处理共2次，独立于监督fit。A12端到端与A13照片保留各自运行时源码摘要，不冒称最终schema变更后的重复实训证据。

A16真实小试计划/runner已备：默认仅校验，SMS动态raw/checked两臂最多22fits/6请求；账本足够且显式--execute才执行。用已知unknown模拟账本做预检反例，正确在0fit/0请求处阻断；此反例不是学校实际账本。**真实历史11条未知仍未对账，今天没有新建真实空账本或请求真实API。**

今天新增：47端到端夹具fits＋9真实照片fits＋2dummy回归＝**58次监督fits**；真实API0、模拟HTTP15。A窗口由203增至 **261fits**，模拟HTTP由10增至25；历史/队友资源另计。数据准备、最终入口检查、blocked preflight均不加监督fit。

今日A线可自主完成的接入与验证范围已完成：正式六组入口、split-aware反馈、3请求/11fit与失败链、真实照片文件流程、对称输入契约、两份H数据登记、最终回归、继续运行脚本。剩余条件明确：平台历史对账后真实Flash实验；B接合法对称方法后H1可训练；D5冻结后才打开H模型成绩；完整Q/C1/H/Released仍按后续研发日推进。没有把今日工程验收等同整个项目完成。
