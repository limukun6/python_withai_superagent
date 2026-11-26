# SA11：已验证任务契约进入正式CLI，恢复读题不再依赖实验注入

基线00e32db。旧SA9只在实验脚本中复用真实回复，本批新增生产SemanticArtifact及正式`--semantic-artifact-from`，让任务重新启动时可复用解释，不再重复发读题请求。它只恢复任务解释，**不恢复搜索历史或已训练模型**。

## 瓶颈→结构→证据

| 瓶颈 | 结构 | 实验与结果 |
|---|---|---|
| 真实回复已取得，重新启动却还要请求模型 | 成功语义解释自动保存data-only artifact，正式CLI可直接读入 | 14项新检查，含自动保存、无模型client恢复和正式入口到fit前检查点 |
| 缓存可能套到改过的任务/预测包 | 绑定文件包SHA、规范化训练内容/目标/指标/契约摘要及16个reader源码SHA；重编译和预检后再用 | 同字节搬迁成功；改训练、预测、提案、task摘要或reader均拒绝，未发请求 |
| 有缓存仍可能隐式初始化计费client | 缓存默认跳过模型client初始化；显式B付费选项才另走既有路径 | configured_flash禁调用测试通过，CLI solve子进程收到缓存和B本地/自审参数 |
| 只能在研究脚本patch resolver，交付入口没有能力 | 原unified_agent train/solve→superagent unified→真实_train贯通 | 两条真实standalone solve，stdin关闭，无resolver/client/分类器注入，全部通过 |

完整相关回归**187/187通过，10.265秒，0fit/0真实API**；两条旧Dummy.fit夹具仍显式排除。

正式CLI验证使用原真实SA8回复，显式迁移时先核对accepted proposal与原llm_response、canonical ledger已结算请求，再用当前reader重新编译、记录迁移来源。不是假称旧版本当时就自动导出了新artifact。

| 路径 | 正式CLI自主选型 | 新fit含refit | 新进程预测 | 新API |
|---|---|---:|---:|---:|
| 数值V1，原真实解释缓存 | HistGradientBoosting | 2 | 60行，表头/原ID/顺序正确 | 0 |
| 图像V3，原真实解释缓存 | 梯度直方图＋逻辑回归 | 2 | 40行，表头/原ID/顺序正确 | 0 |

两条合计**4次新真实fit、0新API、¥0新增费用、30.531秒**；真账本文件SHA前后完全相同。没有重新打开开发答案或比较质量，fit_limit2与原4fit不同，不能将新模型分数/选型当成优劣结论。实验没有人工指定算法，B方法目录/排名未改。

## 用法与边界

成功的proactive语义解释会在`agent/semantic_artifact.json`保存缓存。对同一任务的相同文件内容重新启动：

```powershell
python -m unified_agent solve TASK.zip --prediction-input QUERY.zip --semantic-artifact-from PREVIOUS/agent/semantic_artifact.json --run-dir runs/NEW --output runs/NEW/predictions.csv --task-analysis local
```

无需重新提供读题API；原已计量请求/费用保留在来源记录，本次新增调用数为0。此命令会新做预算内模型训练，**不是续接原训练fit计数**；要复用已训练模型直接用predict。B显式付费复核仍可能调用模型，不能把所有组合都说成0API。

v1绑定文件包，目录输入的新鲜解释正常继续但不自动导出该缓存。reader字节变化须显式重验证，不会静默把缓存当成新模型判断。内容SHA用于完整性与版本绑定，不是供应商数字签名或抵抗恶意文件所有者的安全证明。无真实usage的开发缓存会明确记录来源未核验，不能当成真实LLM实验。

原三模态模型的历史分数继续归SA8/SA9；本批不重新评分。项目未冻结，未知任务全成功保证仍无证据。尤其**新任务实时付费读题的正式solve部署**还需要把已接受历史清单/校园直连接线从研究runner完善到正式配置，不能用本次零API缓存路径冒充该验收完成。

## 文件与复跑

- 生产：superagent/semantic_artifact.py；semantic_contract、unified、autonomous_run及root CLI接线。
- 无fit runner：scripts/run_sa11_semantic_artifact.py；证据research/sa11_semantic_artifact_attempt1。
- 真实CLI runner：scripts/run_sa11_cli_reuse.py；证据research/sa11_cli_reuse；实际命令日志与模型在忽略的runs/sa11_cli_reuse。
- 旧真实回复显式迁移工具：tools/import_semantic_artifact.py，不调用模型、不修改账本。

后续优先：正式新任务付费入口的历史/网络配置、无manifest图像/数组、真实工具返回后的提案闭环，以及独立陌生任务验收。
