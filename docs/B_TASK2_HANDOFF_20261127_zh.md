# Task2 交接 · 2026 年 11 月 27 日

### 有指定验证成绩的交付

**710/940（75.53%）**。使用课程训练原包的 8,301 条样本，统一入口自行完成 20 次拟合，上限 22 次。Agent 给 BGE-small、GTE-base 和基于 GTE-base 的相似案例投票分别分配 37.5%、37.5% 和 25%。从冷启动生成的 940 行与此前评分的输出逐字节一致。模型包换目录并断网后再次复现全部预测。课程正式测试和封存迁移任务 H2 均保留。[草稿 PR #4](https://github.com/limukun6/python_withai_superagent/pull/4) · [固定提交 c03762a](https://github.com/limukun6/python_withai_superagent/tree/c03762a856c2b5b261778afbc63422c5319628ff) · [实际决策与验证](B_DISCOVERED_ENCODER_IDENTITY_20261127_zh.md) · [交付包回执](../research/paired_encoder_search_1127/corrected_delivery_portable.json)

队内传递的可运行包为 `Released-Task2-validation-c7982b0159ad.zip`。解压后，在包的最外层安装依赖并用自己的无标签特征包预测。已验证的环境为 CPython 3.11，模型权重随包提供。

```sh
python -m pip install -r requirements.txt
python -I predict.py --features features.zip --output predictions.csv
```

### 扩大预算后的四模型候选

在同一通用资源目录下，完整课程训练用 29 次拟合，上限 32 次。训练内两次复核分别净增 14/1,616、29/1,696，Agent 最后给 MPNet 50%、BGE-small 18.75%、GTE-base 18.75%、相似案例 12.5%。独立目录断网重载后，重现 940 行输出，其中 65 行与上一个交付不同。**此包的课程指定验证准确率尚未知。** 文件名为 `Released-Task2-research-unscored-7c7ab56f00d5.zip`，使用方式同上。[自主选择日志](../research/paired_encoder_search_1127/recruitment_raw_decision.json) · [与已有交付的逐行比较摘要](../research/paired_encoder_search_1127/recruitment_raw_changes.json) · [交付包回执](../research/paired_encoder_search_1127/recruitment_raw_portable.json)

两次匿名启动的外层题目没有共享标题组。相对各次保留的原组合，第一次 1,268 → 1,285/1,711，第二次维持 1,219/1,645。合计净增 17/3,356，只有第一批提升。[两批对照](../research/paired_encoder_search_1127/recruitment_replication_panels.json)

### 本轮模型与决策实验

开发者将公开的 paraphrase-MPNet 和 BGE-large 固定到可核对的发布版本。统一入口识别兼容的通用资源，候选进入后，由 Agent 根据实际反馈决定训练、融合及最终交付。另设同数据对照，专门给每个新增模型各一次真实训练机会，用来检验它们即使得到机会能否提供补益。轻量 Potion 的对照已经完成。[paraphrase-MPNet 发布说明](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2) · [BGE-large 发布说明](https://huggingface.co/BAAI/bge-large-en-v1.5) · [两份资源的固定版本与校验](../research/paired_encoder_search_1127/published_sibling_installation.json) · [Potion 的实际选择](B_STATIC_ENCODER_TRIAL_20261127_zh.md)

统一入口从 6,590 条匿名训练样本重新启动，训练预算为 32 次。起步时可选的方法从 73 个增到 75 个，两个新编码器都在 14 步决策中可见，Agent 把实际训练机会给了其他候选。共完成 23 次拟合，最终选择 MPNet 与 BGE-small 各半。换进程重载后，1,711 条训练内留出中答对 **1,285 条（75.10%）**，全部逐行预测与扩大目录之前一致。模型 API 0 次，输入和输出均为 0 token，课程计费 ¥0。[候选出现、训练与当选记录](../research/paired_encoder_search_1127/published_sibling_ordinary_choice.json) · [逐行对照及重载回执](../research/paired_encoder_search_1127/published_sibling_ordinary_summary.json) · [有序预测的成对评分](../research/paired_encoder_search_1127/published_sibling_ordinary_results.json)

| 训练内实验 | 程序做出的选择 | 后续证据 |
| --- | --- | --- |
| 两个新编码器与已选四模型融合 | 1,319 条中，原组合 931，新两者各加 25% 得到 937 | 另一组 1,318 条从 941 降至 933，纠正 24、损失 32，程序放弃这一配方 |
| 给每条文本学一个分数，用两条的分差预测 | 在相同输入上比较线性与非线性打分，选线性打分占 50% | 两组分别 941 → 943、917 → 920，合计净增 5/2,627（+0.19 个百分点），低于预定的 +0.20 个百分点 |
| 按训练反馈移除一个融合成员 | 先提名、再确认、最后在隔离的第三份标题组上计分 | 对同一 8,301 条训练数据做两次分组，相对完整组合分别少答对 17、6 条。按每次选择需要的独立资源文件估算，权重体积平均少 7.0%、3.9% |

融合的提名与确认、按文本打分的实际选择、移除成员的逐折结果均保留。上述独立对照只使用课程训练集，模型 API 调用为 0。[融合提名](../research/paired_encoder_search_1127/published_sibling_audition_nomination.json) · [融合复核](../research/paired_encoder_search_1127/published_sibling_audition_results.json) · [单条打分对照](../research/paired_encoder_search_1127/item_utility_results.json) · [移除成员对照](../research/paired_encoder_search_1127/member_removal_results.json)

### 接续范围

研发分支 `codex/task2-static-encoders-1127` 同时包含扩大预算后的调度、静态编码器接入和本轮证据。带成绩交付的个人分支仍由草稿 PR #4 跟踪。两个大模型包通过队内文件传递，仓库提供源码和两份校验回执。字符上下文方法仅登记了研究方案，尚未训练。
