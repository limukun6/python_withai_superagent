# A线最终候选提交与历史分数归属

2026-11-25。此提交交接A线已完成开发，不等于替团队向学校提交，也不宣称Task2当前已通过同版验收。

## 1532/2030是谁跑的

Git历史明确归于**队友B线的运行与交付记录**，不是A线本机当前复跑的结果：

- 早期统一复现：`9fe68c83a10b605af7c4bb1d93a8a8c559a8a84b`，作者Jingle He，2026-11-24，记录历史复现及全训练/校准对照。
- 原始包完整运行复现：`165b1009549dffde208c218797e224548e8aff72`，作者Jingle He，2026-11-24，新增`research/unified_raw_1124/{protocol,results}.json`等。
- 原始包记录Task1为1532/2030，Task2为689/940；模型、预测、任务包及源码摘要均保留。Task1原始包模型SHA为`8c50b3ffef883b00aeb9dd23793362b713bea3dfbcf8ee5efd6e3f2936792c96`。
- 作者与项目分工足以确认B线提供的证据归属；不能据Git作者推断具体物理电脑或声称本工具见证过B的训练。

这是A整合后引用的B历史结果。A本机新统一运行是1527/2030；版本/环境/资产不同情况下两者不强求相同，也不能把历史分数直接填到新模型。

## 此次交接的版本与运行状态

- 本提交前最新合并代码/结果：`e85d8b9`，工作树`D:/codex/python_withai/runs/integration_merge`，本地分支`a151-b-merge`。
- 推送目的：A持续分支`codex/expanded-feedback`，另建固定交接分支`codex/a-line-release-20261125`。不覆盖main或B分支，不强推。
- 生产代码无额外修改；Task1以完整自主9fit运行形成交付候选，1527/2030=75.2217%，独立进程与独立源码副本预测一致。
- Task1清单：`research/release_acceptance/delivery/manifest.json`；模型SHA `2092d8ba08c63e8b6ea90c217de560b413ca4475e30228bfb684ea54d3e10756`；预测文件SHA `70c909a0781416ecb9f1c90b2431a22d929733615b4f28b8f6a8914f1bb95564`。
- 权重/私有新模型/SQLite/密钥按既有约定不新增提交。当前Task1实际模型仍为ignored本机`research/release_acceptance/delivery/model.joblib`，Git仅有清单与预测；接收方需要从本机另行取得该模型或按入口重训。历史solutions目录已有旧模型保持旧身份。
- Task2当前只验到缺资产的回退路径，不能把历史73.30%宣称当前交付成绩。需要B提供已固定的L12/E5资产和实际配置/模型清单，或在其已有资产的机器上用本提交做同版自主运行与隔离交付验收。不能为此手工指定最终赢家。
- 所有Released任务必须按同一最终生产版本生成、预测与日志一致后才是完整正式提交。当前研发探索可停止，Task2同版资产核准仍待完成。

## B接续时需要看什么

1. 从`codex/a-line-release-20261125`取A交接版本，读取`docs/CURRENT_AGENT_STATE.md`及`docs/batches/RELEASE_ACCEPTANCE_V2_20261125_zh.md`。
2. 对照`docs/UNIFIED_AGENT_20261124_zh.md`的权重环境变量、固定版本及注册资格；优先恢复既有L12/E5资源，不先开新的编码器实验。
3. 若能找回历史完整portable包，先验SHA：`aa54293b21861ecb5c63aee08e4e5959ea4bb796d23f22ced4a7ac7f7f144555f0a`。历史Task2模型SHA `16e0a33c05bec0b0932d1d6a40af5604bf4102b64a8837259e8a35cc5dd91f56`。历史模型先做加载复核，之后再与最终源码版本身份对齐。
4. 补齐Task2同版交付后再统一对外成绩；1527与1532的历史区分不得丢失。A本次没有为追5题新增训练。

## 提交前核验范围

先保存本文件，再执行无fit/无API定向回归与当前Task1清单/实际模型/预测SHA核对；结果写`research/final_a_line_intake_20261125/`。不运行包含真实fit的端到端测试类。生产身份不因新增说明文档被重标，训练时的源摘要仍保留。

本次Git历史审查、核验和推送新增训练0、模型API0，不改共享账本。最新最近账本457请求/已知课程¥7.925792/unknown16/pending0；未知费用未核清，不能称总费用预算合规已验。
