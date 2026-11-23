# 同一套 Agent 换到电影台词后选了什么

标题任务的课程指定验证成绩仍为 **689/940（73.30%）**。这里另用 Cornell Movie-Quotes Corpus 检验两个问题：冻结标题模型能否选中更容易被人记住的台词，以及在拿到新任务的训练标签后，Agent 自主选出的模型能答对多少。台词的参考答案来自 IMDb 收录情况。[语料目录](https://www.cs.cornell.edu/home/llee/data/) · [ACL 原论文](https://aclanthology.org/P12-1094/)

| 评价范围 | 固定普通 E5＋十二层 MiniLM | 固定 E5＋带词频特征的十二层 MiniLM | 在台词训练数据上自主选出的 E5 |
| --- | ---: | ---: | ---: |
| 全部 662 部影片，2,194 对有效台词 | 1,225（55.83%） | 1,201（54.74%） | — |
| 从训练中整部留出的 136 部影片，450 对 | 256（56.89%） | 253（56.22%） | **335（74.44%）** |

原发布文件共 2,197 对。其中三对两边文本规范化后完全一样，按事前约定排除。两侧都取剧本台词，去掉行号。没有一对包含与标题训练样本完全相同的规范化文本。全语料上加入标题词频特征后多答对 77 对、也多答错 101 对，差 −1.09 个百分点，按影片抽样的 95% 区间为 −2.32 至 +0.13。[冻结比较方案](B_frozen_quote_transfer_1123_zh.md) · [结果](../../research/b_quote_transfer_1123/frozen/results.json)

**Agent 的自主选择。** 同一影片和跨影片共享的台词都放在划分的同一侧。Agent 收到匿名的 1,744 对带标签训练数据，来自 526 部影片。它尝试词特征、六层、普通十二层、带词频特征的十二层和 E5，再在第二个内部划分复核 E5 与普通十二层，最后交付 E5 单模型。首轮 E5 答对 264/348，普通十二层为 262/348；复核中 E5 为 266/350，十二层为 270/350。按两个划分判断的组合晋级条件未满足，保留开局领先的 E5。共 5 次探索、2 次确认、1 次全训练集重训，模型 API 0 次。对照中的标题 Agent 在同一目录下交付了 E5 加词频十二层。[事前影片划分](B_quote_adaptation_1123_zh.md) · [每一步决策与分数](../../research/b_quote_transfer_1123/adapted/model_frozen.json)

在同一批 450 对上，适配后的预测相对冻结标题联合模型多改对 136 对、多改错 54 对，净增 **82 对，18.22 个百分点**，按影片组重采样的 95% 区间为 **+12.36 至 +24.16**。这衡量了取得新任务标签后整个训练与选择流程的收益。单独衡量调度器的贡献需要同数据、同训练预算的固定选择对照。此处有一份预登记开发划分。[留出比较结果](../../research/b_quote_transfer_1123/adapted/results.json) · [匿名逐组计数](../../research/b_quote_transfer_1123/group_counts.json)

**交换左右和处理标点。** Task2 训练内保留区的 1,693 对，以及台词的 450 对，交换左右后模型逐行选中了同一段文本。两边统一大小写、去标点后，标题正确数从 1,237 变为 1,218，台词从 335 变为 325，两项配对区间都跨零。剧本原文按空白分词时，450 对中有 119 对两边同长，模型在其中答对 86 对。按较长的一边猜、同长各计半分，在整组取得 256.5/450。语料作者的匹配约束与实读剧本字符串上的空白词数口径有所差别。[预登记诊断](B_pair_surface_checks_1123_zh.md) · [分项和区间](../../research/b_quote_transfer_1123/surface/results.json)

在本发布目录运行 `python scripts/verify_b_quote_delivery.py` 可核对分组胜负数、分母、决策和真实拟合记录。完整复跑先从作者 ZIP 读取配对文件与 README 两个成员，代码会校验两者的 SHA-256。依次运行 `python -m scripts.run_b_quote_transfer --help`、`python -m scripts.run_b_quote_adaptation --help` 与 `python -m scripts.run_b_pair_surface_checks --help`，参数分别指定本地权重工作区、冻结模型、输入与独立输出目录。核对本地原始预测可运行 `python scripts/verify_b_quote_delivery.py --rebuild TRANSFER_RUN ADAPTATION_RUN`。[作者原始 ZIP](https://www.cs.cornell.edu/~cristian/memorability_files/cornell_movie_quotes_corpus.zip)
