# General Algorithm SuperAgent — v0.42

实验入口：`python -m superagent adaptive-local TASK.zip --run-dir NEW_DIRECTORY`。这是仅用训练数据和本地算法的准确率实验，支持时才显式选择。它对同一训练内划分逐行比较当前方案和挑战者，决定探索、到另一划分复核或提前停止；最终模型和判定理由写入新目录。至多10次搜索/复核拟合和1次重训，默认入口不变。输出支持已有 predict 命令。此原型尚未验证其他指标；分数接近的判别是启发式规则，不是置信区间。

可选 `--anchor schema`：类别表格先试 CatBoost，左右偏好文本先试词＋字符＋通用统计，其余输入沿用目录首项。首项占用一个正常试验名额；后续决策和确认选型不变。可单独与 expanded 使用。`--portfolio challenger` 在 expanded 上增加一个适配结构的本地候选：偏好文本的定向编辑特征，或类别表格的低基数组合。示例：`python -m superagent run TASK.zip --planner local --policy interleaved_routes --portfolio challenger --anchor schema --trials 4`。默认配置保持原状。

v0.40 可选`--refinement research`接入运行时文献检索、受限sklearn管线生成和普通实验反馈：Researcher在两次结果后选择搜索/跳过，生成有来源和代码哈希的新候选，Planner可以实际试验再继续决策。所有模型角色共享预算，最多2次额外模型请求，默认关闭。当前只检索arXiv/Crossref元数据/摘要；生成范围是已安装库的受限工厂，不是任意Python或自动安装新算法。详见[运行时研究边界](docs/runtime_research.txt)。

v0.39 扩展常见分类输入和评估：多文本/文本混合表格、明确关系语义的多分类文本对、PNG/JPEG类别目录、无外部validation任务、2至5折小数据分组划分，以及AUC/F1变体/balanced accuracy。AUC使用概率并按类别导出；没有外部验证集时不生成外部成绩。仍不声称覆盖全部常见任务。详见[输入与评估边界](docs/general_extensions.txt)。

v0.38 增加可选`--portfolio learned`：从随机初始化训练的小型成对文本模型（32/64维），以及按训练内分组OOF错误互补性决定是否训练的融合器。安装`requirements-neural.txt`；没有外部权重下载或API。所有内部选轮数、交叉验证和重训计入分类器拟合；默认expanded保持，实验收益需验证。详见[本地训练工具](docs/trainable_pair.txt)。

v0.37 增加六方向研究的可选模块：`--error-feedback` 提供训练内错误分组；`--architecture selective_review` 在新路线、提醒、执行失败或最后一步触发审阅；`--portfolio semantic` 增加训练内拟合的TF-IDF/SVD潜在语义及混合特征；`--refinement compose` 按已有反馈生成有界文本方案；`--allocation uniform|successive` 比较统一资源与分阶段样本资源。默认均不启用，能力与收益须分别验证。详见[六方向实现和实验边界](docs/six_directions.md)。

目前 agent 规划仍使用 DeepSeek。学校 key 已通过一次千问对话模型连通性检查，不能据此声称 Qwen Embedding 已接入。本地潜在语义特征不调用大模型，不等同于预训练语言模型语义向量。

v0.36 修复真实v0.35中审阅理由略超500字符便中断决策的问题：500字符改为推荐长度，1200字符为硬上限；超推荐长度记录提醒，原始响应保留，不截断、不重试。v0.37已完成四组共同源码的真实试跑，各652/940且无格式恢复；未证明审阅提高准确率。v0.35失败记录保持原样。详见[审阅长度处理与证据范围](docs/review_length.md)。

v0.34 在审阅前提供当前提案的参数变化与有限的一致性提醒。检查只在同路线、仅C变化时核对英文正则化方向用语；提醒不等于错误证明，不自动换模型或追加调用。审阅可保留候选、只修解释；原提案和提醒仍保留。详见[审阅前检查](docs/proposal_consistency.md)。

可选 `--architecture single|self_review|reviewer|selective_review`，共用任务指标、候选、训练、确认、重训与费用账本。默认 single；审阅模式须显式使用 `--planner llm`。历史及研究对照不能证明Reviewer普遍提高准确率。详见[决策架构与对照边界](docs/decision_architectures.md)。

本版本增加共同任务/提案/结果/决策接口、任务指定Accuracy或Macro-F1和假设记忆，详见[架构契约](docs/architecture_contracts.md)。

本版本在训练集内部选择方案，记录实验、原始模型决策和可复核的数值结果。支持旧 Released ZIP、manifest v1，以及部分常见原生 CSV/NumPy 文件布局；覆盖数值表格、混合表格、普通文本和图像数组，支持二分类和多分类。

## 安装与本地运行

本轮使用 Python 3.14.7、NumPy 2.3.4、SciPy 1.16.3、scikit-learn 1.7.2、joblib 1.5.2、CatBoost 1.2.10、requests 2.34.2、Pillow 12.3.0。其它环境需运行检查。

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m superagent run /path/to/task.zip --run-dir /path/to/new_run --policy balanced_blind --trials 5
```

run 和 experiment 是同一执行入口的两个名字，默认只使用本地规则，不读取 API 密钥、不调用 API。主搜索后，内部前三名在三个固定划分上复核，按均分选型，再用完整训练集重训并评分指定验证集。输出目录必须是新目录。

可选 --time-policy reserve：根据已完成实验的整个worker耗时（包括启动和模型读写）为复核与最终重训预留时间；时间紧时会提前停止主搜索。复核本身也为重训留出窗口，复核不全仍保留主搜索优胜者。默认shared沿用全部阶段共用剩余时间的方式，便于对照。时间预留使用估计，未知配置或全量重训可能更慢；HTTP超时是连接/读取阶段限制，不是严格总墙钟上限。输出和评分也可能超出预留时间。少跑的主搜索会如实记录，requested_planner_completed不会因此改成true。当前仅支持单一最终模型，不用于集成重训。

可选 --feature-cache run：在本次新运行目录中复用已按当前训练输入拟合的成对文本TF-IDF表示，默认off。缓存键包含完整训练输入及顺序、表示参数、源码和依赖版本；不同确认划分、完整训练集会使用不同键。分类器每次重新拟合，实验次数不减少。缓存只涉及无监督表示，不读取标签、验证成绩或任务名；普通文本管线、NB监督词权重及其它数据类型暂不复用。每个worker记录构建/磁盘命中/训练矩阵内存复用及耗时。

完整缓存目录原子发布并核验内容哈希；损坏条目会记录失败，不静默改写。中断留下的临时目录不会被当成已完成缓存。缓存为本地可信程序生成的joblib文件，校验用于发现意外改动，不应加载外部不可信缓存。保存的最终模型包含拟合器，预测不依赖缓存目录。首次构建、校验和读写也有开销，应比较整轮耗时后再决定是否启用。

本地策略可选 random、balanced_blind、greedy_routes、protected_routes。最后一种探索保护规则不保证更高成绩；已完成的同预算比较存在输赢。

新增可选 --policy interleaved_routes：共享首个实验后，优先探索新路线与利用已有反馈交替进行。它保留其它路线，不把一次失败或低分当作整条路线无效。这样小预算也能留出使用反馈的步骤，但代价是可能少试一些路线；是否提高成绩必须按相同预算检查。该策略只做本地研究，默认仍是balanced_blind。

可选 --refinement midpoints：在original/expanded目录内，某路线有至少两种不同C的成功测量后，围绕该路线当前最佳配置与相邻已观察C，自动提出几何中点作为新候选。C仍须位于原目录该路线范围内，其它算法与预处理参数保持；已有或已生成的配置不重复，每步最多生成2个、整轮最多4个。它扩展可选实验，不自动追加训练或API次数，原候选和路线都保留。默认off；只支持budget_routes，以及本地greedy_routes/protected_routes/interleaved_routes，不能用于blind/random的无反馈对照。

新配置保存在candidate_registry.json及每次生成后的快照中，ID由规范化配置决定；worker校验模板、参数范围、ID和数据类型，主搜索、确认与全量重训都解析同一登记。最终模型自带拟合后的预处理与分类器，预测不需要登记文件。生成规则只使用当时已完成的实验，日志保留两个参照实验；候选登记本身不是训练成果，也不保证新配置会被选中或提高准确率。

可选 --portfolio research 在成对文本原有22项候选之外增加6项：训练内监督词频权重加logistic，以及word+char+style加线性SVM，各有三个固定C值。默认expanded目录不变。监督权重、词典、缩放器都只拟合当前训练分区；新候选不收敛会记录为失败，不会静默使用。SVM没有概率输出，不能直接用于概率平均集成。

可选 --portfolio nonlinear 保留expanded的22项，另加7项：深度3/5的通用文本统计成对树模型、各自与word_char_stats C1.0按固定0.2/0.4概率权重混合的4项，以及该线性模型的交换对称化控制1项。统计特征沿用16维字符/单词形状与标点计数，不使用任务名称或人工指定关键词。预测概率按左右互换取互补平均，完全相同输入的0.5概率仍用固定类别平局规则。新增目录只供研究，原research与默认expanded不变。

混合候选每次训练两个分类器，worker的classifier_fits和候选元数据会明确记录；单个候选评估不等于单次分类器训练。失败候选可能已经训练部分组件，不能按0计算。权重固定，不在指定验证集上选权重；概率平均是否有益仍需训练内部比较和额外确认。

可选 --portfolio adaptive 保留类别表格的14项候选，增加深度6/8、L2=3/10的4个自适应CatBoost候选。每次先在当前训练分区内部按归一化后相同特征分组划分80/20，用内部留出集的Logloss/MultiClass损失选择轮数（最多1500，耐心80），再按该轮数重训完整的当前训练分区。外层评估标签不参与选择轮数；最终全量重训也使用同一内部选轮数程序，不使用指定验证集。其它任务的数据类型沿用expanded目录。

选择轮数与重训实际训练两个分类器，两次都计入计算账目；保存内部索引、损失曲线和最终树数，反馈只接收选定轮数与拟合数量等精简事实。候选的额外计算应计入对照，不能将更多训练量称为同预算优势。该目录是可选研究方法，默认expanded不变。停止与选轮数使用[CatBoost官方fit接口](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit)及[get_best_iteration接口](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_get_best_iteration)。

监督词频权重借鉴 [Wang 与 Manning（ACL 2012）](https://aclanthology.org/P12-2018/) 的平滑对数频率比。本实现改成成对偏好差分配合logistic，是新的适配实验，不是原论文NBSVM结果的复现。本轮只比较训练集内部成绩，不推断课程测试成绩或LLM规划收益。

本版本已替换旧 run 的直接验证集选型路径。每次实验在训练开始前保存源码与数据包哈希、划分规则、种子和预算；结束时再次检查源码是否改动。

## 数据契约

manifest.json 的 schema_version 为 1，objective 为 classification，metric 为 accuracy 或 macro_f1；省略时为 accuracy。

- 数值表格：modality=tabular，NPZ 中 X 为 N×D，y 为一维标签。
- 图像：modality=image，NPZ 中 X 为 N×H×W 或 N×H×W×C。
- 图片文件：modality=image、CSV中声明一个type=image_path输入列，同ZIP保存PNG/JPEG；需显式声明image_preprocessing。
- CSV 表格：columns 显式声明 numeric/categorical 类型，target 声明目标列；目标不能同时作为输入。
- 单文本：modality=text，声明一个 text 输入列。
- 成对文本：modality=pair_text，声明两个 text 列、pair_label_one=left 和 0/1 标签。

split 格式为 csv、npz 或 npy；声明 train 和 validation 的相对路径。npy 的 path 为输入数组，target_path 为单独标签数组。训练过程不访问 test。

图片CSV中的路径相对manifest.json所在目录，列的文字和文件夹名字不作为模型特征。示例契约如下，train/validation清单包含label，预测清单只需image列；同一行对应的图片与目标保持原CSV顺序：

```json
{
  "schema_version": 1, "objective": "classification", "metric": "accuracy",
  "modality": "image", "columns": [{"name": "image", "type": "image_path"}], "target": "label",
  "image_preprocessing": {"width": 32, "height": 32, "mode": "RGB", "resize": "stretch_bilinear"},
  "splits": {
    "train": {"format": "csv", "path": "train.csv"},
    "validation": {"format": "csv", "path": "val.csv"},
    "test": {"format": "csv", "path": "test.csv"}
  }
}
```

图片解码采用固定策略：按EXIF方向转正、透明部分合成到白色背景、转换为RGB或L，再按声明宽高双线性拉伸，输出uint8像素。宽高各1–512，单幅源图最多16,777,216像素、编码文件64 MiB，单分区解码结果最多256 MiB；当前整体加载到内存，训练过程的额外内存仍另计。仅支持单帧的常见8位PNG/JPEG，不自动推断任意图片目录。路径越界、缺图、重复ZIP成员、格式不符或坏图会明确报错。

预处理、解码策略与Pillow版本随task_spec冻结，预测时不得更换尺寸、颜色、缩放契约或解码库版本，也不能换成绕过解码的数组接口。图像文件加载由[Pillow Image接口](https://pillow.readthedocs.io/en/stable/reference/Image.html)与[ImageOps接口](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html)实现；具体行为已针对上列本机版本检查。数组图像入口保持原行为。

没有 manifest 时，额外支持以下标准布局（可以放在 ZIP 内的子目录，但同名文件必须唯一）：

- train.csv 和 val.csv/validation.csv/valid.csv：目标列为 target、label、y、class 中唯一存在的一列。输入列按训练文件顺序冻结，能转数值的非缺失值作为 numeric；普通短字符串作为 categorical。单个长文本列可识别为 text。所有字段类型只参考训练内容。
- train.npz 和相应验证 NPZ：X 为数值输入，y 为标签。
- X_train.npy、y_train.npy 和相应 X_val.npy、y_val.npy（也可用 validation/valid）：数值 N×D 数组按表格处理，N×H×W[×C] 按图像数组处理。

这是一套可检查的格式规则，不是任意文件语义理解。短文本可能按类别处理，整数编码的类别可能按数值处理；明确的 manifest 可以覆盖这些规则。多列文本、含文本的混合表格、未知目标列和含糊的配对标签语义需要显式说明；当前模型库不支持所有这些组合。图像目录和时间序列语义仍不在自动适配范围内；当前指标仅限Accuracy与Macro-F1。新增 NumPy 路径禁止 object/pickle 数组；旧 Released C_train 兼容适配器仍沿用原读取方式。

预测使用保存的 input_schema，不重新根据预测数据推断类型；CSV 表头换序会按冻结列名重排，缺少列或维度改变会报错。预测包可含 test.csv、test.npz 或 X_test.npy；标签不用于预测。模型和 task_spec 都有哈希校验，避免保存后悄悄更换输入语义。这里的预测功能只用人工能力样例验证，课程正式测试集仍未执行。

## LLM 实验与费用

需要真实 LLM 时显式设置 --planner llm，并提供 --api-url、--api-key-file、--model、--budget-ledger、--task-budget-id。同一课程任务在不同种子和迭代中必须复用同一个小写预算标识，不能每轮新建标识。本地模式若混入这些参数会报错。

LLM模式使用budget_routes，关闭思考输出。single 每次自适应决策最多调用1次，self_review/reviewer 最多2次；包含共同首个实验时，总上限分别为 trials-1 和 2×(trials-1)，还受显式请求上限、时间与费用约束，没有自动重试。默认本轮增量额度¥0.10，可用--llm-budget-rmb调整。正常run/experiment入口同时检查持久任务累计¥20上限和单轮额度。未知/未完成请求会阻断对应任务的新调用；共享诊断未知会影响全部任务。旧低层研究接口直接创建ExperimentLLM时仍允许无持久账本的显式研究用法，不能把该单轮接口说成已有跨轮保护。

首次使用已有项目，应导入经过核对的历史审计JSON及其SHA256；确认没有任何既往API费用的新项目才使用--new-project。已有账本不可重置覆盖。

```sh
python -m superagent budget-init /path/to/budget.sqlite --import-audit /path/to/verified_ledger.json --audit-sha256 VERIFIED_SHA256
python -m superagent budget-status /path/to/budget.sqlite --task-budget-id task2
```

```sh
python -m superagent run /path/to/task.zip --run-dir /path/to/new_llm_run --planner llm --api-url https://your-approved-gateway/api/v2/chat/completions --api-key-file /path/to/key.txt --model DeepSeek-V4.1-Flash --budget-ledger /path/to/budget.sqlite --task-budget-id task2 --llm-budget-rmb 0.10 --trials 5
```

调用前使用保守字节数估算预留费用，不构成精确 token 上界；调用后按实际用量记账，超限或账目不明会停止后续调用。返回模型不在已批准标识中时，保留已知 token，但费用留为未知并停止调用。损坏响应若仍有可核实用量，也照实计费。

每次传输前用SQLite事务持久预留，结算后释放预留并记录已知课程费用。进程中断的预留和未知用量不会自动按0清除，重启后仍会阻断同任务。共享诊断费用保守地全额计入每个任务的额度检查，项目实际费用汇总只计一次。费用结算数据库写入失败时，普通用量日志仍保留已知token和费用，未完成预留继续阻断。无法保证网络请求绝对不超估算，只能预留检查并在实际超额后如实记账、停止后续调用。

当前没有自动撤销未决请求的功能；须取得实际用量证据后对账，不能因为希望继续运行就改为0。账本和task-budget-id记录的是课程额度，budget-status显示的剩余额度不等于平台账户余额，也不包含未导入的其它工具消费。

run_outcome.json 区分 model_available 和 requested_planner_completed。规划回退、少跑主搜索、复核或重训未完成、费用不完整、源码改变，都不会宣称整套请求已完成。退出码 0 为完整完成，2 为留有模型但流程不完整，1 为没有模型；参数错误也使用 argparse 的退出码 2，需同时查看日志。

最近已验证的真实运行是v0.29使用冻结v0.28源码，4次请求、656/940、¥0.025619。新增参数与逐行事实已送达但没有准确率提升；本v0.31源码尚无真实LLM或课程新成绩。

费用使用用户指定的更新后官方 DeepSeek 空闲时段价格，不计缓存优惠。已验证的 Flash 1017 版本为输入 ¥1/百万 token、输出 ¥4/百万 token。记录每次实际 token 用量；缺失用量不是零费用，应停止当前批次后续请求。先前未解决的请求必须保留在累计账目中，新的独立批次不能把旧请求标成已结算。

日志中的 actual_cost_rmb 未知时保持 null；课程口径费用不能冒充平台真实扣费或账户余额。

## 反馈与证据

每次决策都能看到分组方式和计划中的最终选择规则，包括复核次数、额外拟合预算、失败回退以及最终重训。成对文本可按共享规范化标题的连通分组隔离。

数值比较由执行器计算。explanation_audit 保留 LLM 原文，生成确定性的事实报告，并对部分明确的持平和显著性措辞给出复核提示。它只是有限的英文短语检查，不是完整语义验证；没有提示也不代表每句说明都正确。

configuration_change额外从候选目录和决策时历史计算参数前后值、方向和参照实验。C下降意味着更强正则化，但不保证分数更高；跨表示/算法的数值C也不能证明有效正则化程度相同。报告区分同路线单参数变化与路线/多个字段变化，不能把后者的收益只归因于一个因素。参照首先采用结构化expected_change中的reference_trial，否则采用同路线最佳完成实验，再回退到总体最佳完成实验；无完成结果时仅比较最近尝试的配置。反馈盲对照不会得到基于测量结果选择参照的这类说明。原始自由文本不被改写，该功能没有宣称自动理解或纠正所有语言矛盾。

paired_error_comparison记录当前模型相对之前最高分模型，修正了多少旧错误（newly_correct）、损失了多少旧正确（newly_wrong）以及净增正确数。持平参照取最早实验，与latest_result一致。双方都对、双方都错和预测不同的行数也保留；多分类中双方都错仍可能预测不同。逐行预测与标签、完整有序评估payload的哈希核对，只有完全一致的分区才能比较；缺失旧证据明确不可比较，不伪造0。记录不一致时不输出数值证据，统一入口不会标记整轮完整完成。

规划者只看到简短计数和明确的参照实验，标签、逐行预测和内部索引不会进入当前budget_routes提示。反馈盲对照仍隔离全部结果派生信息。这些计数是反复使用的训练内部样本上的描述，不是误差范围、显著性检验或因果结论；互补错误也不能直接证明集成有效。该功能不增加分类器训练和API调用，未声称已改善真实LLM决策。

valid_policy_run 仅表示执行完成且没有规划回退。explanation_review 单独记录说明核对范围。反馈盲对照不会收到由结果派生的文案提示。

confirmation_memory 提供可恢复的确认记录和持续调度模块，已做同预算本地研究；当前默认入口仍在搜索末尾集中复核。

## 结果范围

v0.8 的六种本地能力检查全部完成，不能当成课程unseen task的成绩。最近真实Released Task2试跑是v0.29使用冻结v0.28源码，验证656/940（69.787234%）、4次调用¥0.025619，0回退。累计已知费用¥0.702311，另1次旧Task1未知；平台扣费未知。它不是本v0.31源码的运行结果。

最终课程结果应由同一最终冻结版本的完整自主运行生成。所有原始失败、回退和费用记录应随结果保存。正式测试集目前仍未执行。

源码、设计、检查和报告整理使用了 Codex 辅助；团队成员仍需理解实现与实验边界。历史设计文档保存在 docs，当前运行方法以此 README 和对应版本的冻结协议为准。


本轮新增接口只做本地验证，没有真实LLM或课程新成绩；旧版本成绩和费用保留原始出处。主指标贯穿确认和最终评分，但仍需正式单/多Agent架构对照。
