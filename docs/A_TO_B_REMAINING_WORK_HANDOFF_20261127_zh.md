# A线剩余工作交接：从SA26实跑继续

交接日期：2026-11-27。用户因安排将A线剩余任务转交队友。**A不再启动新训练或API请求；B接手实现与实验。** 本文件是唯一主交接说明，历史材料用于查证，不按旧队列重复运行。

## 1. 从哪份代码开始

- 仓库：https://github.com/limukun6/python_withai_superagent
- 接续分支：`codex/a-pr3-joint-1127`。
- **SA26代码提交：`888801bbb225db8b516e1226357182e2290561ab`**。本文件随后追加到同分支，取该分支最新提交，确认包含此代码提交。
- 上一完整整合提交：`91f1a8c55fc5d4630302a6d27a6620043f5f6c88`，两个父提交为A `7e9d142`、B PR #3 `4d4222a`。PR #3内容已经进入本接续分支，无需再从旧PR覆盖入口。
- A本机工作区：`D:\codex\python_withai\runs\pr3_joint_1127`；Python：`D:\codex\python_withai\runs\a79_encoder_assets\venv\Scripts\python.exe`。这些路径不是B必须使用的路径。

在干净独立checkout接手，保留B自己的未提交改动：

```sh
git fetch origin
git worktree add -b codex/b-a-continuation-1127 ../a_handoff origin/codex/a-pr3-joint-1127
cd ../a_handoff
git merge-base --is-ancestor 888801bbb225db8b516e1226357182e2290561ab HEAD
```

若B还有晚于`4d4222a`的新开发，先在独立分支整合、检查；新源码重新登记输入和预算，不能把旧源码的结果当同版本成绩。源码和依赖变化会改变reader/source SHA，不能直接照搬A旧的prepared目录执行。

## 2. 现在真实做到哪里

| 部分 | 状态和证据 | 接手后处理 |
|---|---|---|
| P0：两边最新结构整合 | **完成。** 普通/自定义任务早期partial→最终completed；晚到预测适配与原模型交付共用入口；Windows中断、独立模型导出通过。SA25系列最终396项相关检查通过 | 不再重做整合、不整组重跑旧实验 |
| P1：声明失败交LLM恢复 | **代码已实现，410项无fit/API检查通过；真实实验尚未跑** | **第一件事：跑SA26三案例，然后写真实结果和独立审计** |
| P2a：图像文件名/侧表ID | 尚未开发完整能力 | 完成最小通用身份提取与输出适配，再验收 |
| P2b：数值数组与记录互转 | 目前适配器仍按原reader类别限制，不支持任意互转 | 补明确的数值列投影/排序能力，再交LLM选择 |
| P3：合并版不同真实来源原包验收 | 尚未完成；以往换包装、种子和接口检查不能代替 | 固定版本后做表格/文本/图像各一份真实原包完整无人干预演练 |

SA25的四项通过证据是**SA25B前三项＋SA25C独立Windows补测**，不是同一轮首次4/4。两个最终模型经正式导出、搬到新目录，`python -I`禁网络/禁训练入口的64行预测与原文件相同。P0累计8次已确认fit，另有1次旧未计量worker尝试；新增API0、费用0。原失败和夹具问题均保留。

P0完整报告：`docs/batches/SA25_PR3_JOINT_RESULTS_20261127_zh.md`；核验：`research/sa25_joint_audit_01/`。已修复的Windows问题包括SQLite未关闭、worker继承控制stdin、venv launcher/真实解释器的进程树，以及早期先验漏用字段映射。不要回退这些修复。

## 3. SA26已经写了什么

瓶颈：`RestrictionError/DeclarationError/MetricDeclarationError`会在读题阶段直接返回失败；即使LLM懂了题意，后面的固定解析器又会卡在相同措辞。

新增`superagent/rule_resolution.py`，数据式外层如下：

```json
{
  "version": "semantic-rules-v1",
  "interpretation": {
    "version": "semantic-output-v2",
    "output": null,
    "evidence_refs": [],
    "uncertainties": []
  },
  "rules": [{
    "field": "method_restrictions",
    "quote": "Allowed classifiers: affine class scores followed by a softmax",
    "evidence_ref": "doc:README.md:0",
    "value": ["linear_only"],
    "reason": "The stated model scores are affine; normalize that restriction to the supported linear-model filter."
  }]
}
```

该JSON只是接口示例，**不能注入真实实验当模型回复**。

- rules只解释实际未识别的原文限制/指标，须有真实文档ID和逐字引文，每条都必须被失败解析点实际使用。
- 限制首版只支持已有三个可执行flag：`linear_only/no_pretrained_assets/no_external_data`；指标只能是已有实现的规范名称。不能用空限制忽略题意。
- 调用局部上下文让原解析器在重读时使用这份解释；编译结束就清理，不给下一任务留下全局覆盖。
- 显式manifest仍只允许输出overlay及未识别原文规则规范化；不能换目标、输入列或已声明指标。已知限制、显式冲突、评分示例不能被rules改写。
- `compile_response`统一实时读题、缓存恢复和显式迁移；规则、方法限制、原包与reader源进入digest/来源绑定。
- 声明错误现在进入RecoveryController/LLM查证；真冲突可报告`method_restrictions`未决。条件规则、列顺序冲突等尚未获得任意自动解决能力。
- **引用存在和内部一致不证明语义正确。** 模型可能把不受支持的规则误映射成已有flag，必须用外部oracle/原文核验，不能宣称已消除全部误读。

关键文件：`rule_resolution.py`、`semantic_contract.py`、`task_constraints.py`、`metric_declarations.py`、`semantic_artifact.py`、`task_questions.py`、`recovery.py`、`action_context.py`、`tools/import_semantic_artifact.py`。B的方法排序本批未改。

当前证据：`research/sa26_rule_checks_01/completion.json`为**410/410通过、36.266秒、0fit/0API**；新增14项机制检查。三项旧有真实fit夹具明确排除，不代表全仓所有测试。`research/sa26_rule_prepared_01`和`sa26_handoff_prepared_01`仅为准备状态；目前**不存在SA26真实live结果、真实用量或SA26模型**。不要从准备完成推断实验成功。

## 4. B接手第一小批：SA26真实验证

计划已保存：`docs/batches/SA26_RULE_RECOVERY_PLAN_20261127_zh.md`。

| 案例 | 输入和规则 | 预期 |
|---|---|---|
| R1 | Iris，96训练/24预测，manifest输入/指标明确，分类器限制用“仿射类别分数＋softmax”说明 | LLM规范为linear_only，保留已知no_external_data，自主训练/完整重训/正确交付 |
| R2 | Wine，96/24，manifest不列指标，题面用“正确分类行占比”定义 | LLM规范为accuracy，自主完成并按原类别/行序输出 |
| N1 | manifest的no_pretrained_assets=false，原文却禁止预训练 | 保留冲突，0fit，不得称completed；有明确恢复/未决记录 |

当前无模型本地检查三条都失败，正例分别UnknownRestrictionWording/UnknownMetricWording，负例为真正RestrictionError。它们是**已见公开来源＋开发者编写说明**，不是独立未见任务或提分实验。

上限：正例各2次实际fit（含重训），负例0；每条最多3次Flash、360秒、¥0.30估计停止阈值。总计≤4fit/9请求/¥0.90停止阈值，预计机器3–6分钟；额度是停止阈值，不能保证平台最终扣款硬上界。默认solve、stdin关闭、不注入回复、不手选方法；同一真实账本、每案例独立且未使用scope。

先激活B自己的Python环境，按仓库requirements安装依赖。学校模型用`DeepSeek-V4.1-Flash`，不要用旧`-1017`名。学校网络需在B环境可达；无需修改系统代理/防火墙。Windows学校直连配置沿AGENTS。key用B自己的忽略文件`api_key.txt`，不要写入交接或Git。

在B的新checkout执行，下列输出目录必须不存在；重开批次改后缀，不能覆盖或复用已付费scope：

```sh
python -m scripts.run_sa26_rule_checks --output research/sa26_b_checks_01
python -m scripts.run_sa26_rule_live prepare --output research/sa26_b_prepared_01 --checks research/sa26_b_checks_01
```

取得并核对下文账本，将双方原始记录去重后的唯一写入账本定位到`runs/team_usage/ledger.sqlite`，再执行：

```sh
python -m scripts.prepare_sa26_team_history --ledger runs/team_usage/ledger.sqlite --output runs/sa26_b_history_01
python -m scripts.run_sa26_rule_live run --prepared research/sa26_b_prepared_01 --output research/sa26_b_live_01 --private-output runs/sa26_b_live_01 --ledger runs/team_usage/ledger.sqlite --key-file api_key.txt --accepted-history runs/sa26_b_history_01/accepted_history.json --scope sa26_b_handoff_20261127_01
```

`prepare_sa26_team_history`只核对真实非空历史、枚举当前unknown ID/原行SHA、按用户持续授权生成本机路径绑定的父快照，不发请求、不结算旧费用、不造空账本。runner已移除“未知数必须等于A的17”这一机器特定假设，仍严格要求与显式快照一致，且每次reserve由原事务核对ID/SHA和scope。合并B原记录后可能多于17，按真实快照记录。

请求失败立即停止当批HTTP链、不自动重试；保留原错误与unknown/null。用户已明确“忽略未知用量，继续付费调用”，不再等官网账单或重复询问旧授权。续批仍须先分析失败、写新计划/预算，不复用耗尽scope、不忽略pending。

### SA26还需要B补的收尾

1. 检查正式结果，不只看返回码。R1/R2必须`run_outcome=completed`、完整refit、规则/目标/指标/类别正确；N1不能训练或冒称完成。partial输出不算正例成功。
2. runner已做目标/特征FP、规则值、24行CSV和新进程重放核对；还要单独写`audit_sa26_rule.py`或等价审计。绑定实际接受的LLM回复与proposal，避免把后续B选模回复当读题回复；逐条usage/model/token/course_cost对应canonical账本。
3. R1从history/resolved_configuration/registry核对实际训练方法受linear限制，不能只看spec里写了flag。核对旧unknown行SHA/null未改，引用真在原文，规则缓存可重放且不靠新请求。
4. 保存独立protocol/results/completion、通俗报告和进度。若LLM格式/引用失败，保留失败、在剩余原额度内允许已结算内容修复；未获成功不能把模拟测试写成真实恢复增益。

**上述真实SA26与独立审计尚未完成，预计接手后1–2有效小时。** 本轮实跑前不要顺带大改P2，保持归因和复跑清楚。

## 5. P2：随后要补的结构能力

### P2a 图像文件名/侧表ID（约1.5–3有效小时）

目前native图像目录可以读图片、选预处理并输出行号/类别，但原提示明确不支持任意文件名/侧表ID。A新增的设计来源不能代替身份适配。

建议补显式的图像身份recipe：来源为文件名stem/完整相对路径/已存在侧表字段，定义与图片的可核对连接键和行序，再交LLM按题面选择；输出继续复用input_id/template规则。不要按任务名称写专用分支，不能把数字ID转int吃掉前导零，不能猜测有歧义的sidecar连接。

先用已有模型/开发图像做0fit验收：前导零、乱序、两目录同名、侧表缺行/重复键、模板重排。再登记一条完整图像solve，检验原始要求→自主选择ID→训练→导出→新进程预测；不是仅手工传recipe。模型/像素预处理和目标不应因为ID适配而变化。

### P2b 数值数组↔记录格式（约1.5–3有效小时）

当前prediction_adapter允许原生NPY/NPZ之间或records内部映射；原生模型与记录模型的任意互转尚未实现。应补数据式数值投影：明确规范列/列位置、dtype、顺序和允许的二维shape，接入既有reader，保持model_kind和模型含义；LLM只选可执行投影。

至少覆盖NPY训练→CSV预测、CSV训练→NPZ预测两个方向；缺列、重复映射、标签列误选、额外参考数组、ID混入特征的负例。按原冻结特征逐项核验，不填零、不默默改顺序。优先复用模型实现0fit诊断/恢复，再安排少量完整路径；所有内部fit实际计数。

其他类型如任意数学目标、回归、任意附件格式并非现分类执行器能力；需要明示支持边界，不能用“LLM介入”声称任意任务保证成功。条件限制和复杂声明若不能忠实落地，保留为明确局限或另立小批，不无限扩张本轮。

## 6. P3：共同版本完整原包验收（约2–4有效小时）

P1/P2稳定后固定一次验收版本和依赖/资产/候选清单，表格、文本、图像各选一个**不同真实来源的原说明任务包**。不要把同来源换文件名、换种子当新任务；不要临时写针对某个任务的解法或人工告诉它目标/方法。H1/H2和课程正式测试仍按原封存要求，不因交接自行打开。

先登记包SHA、任务指标、预算、独立oracle、数据访问范围、每路径含refit的实际fit上限和每任务3次请求。普通本地＋Flash选择共用可执行方法清单。小试后统一墙钟；本批时间/费用另行登记，不能默认沿用SA26的¥0.90额度。重训练串行占用共享GPU，不同时开两个写同一账本的队列。

验收：同一命令无人干预启动；读到什么、失败在哪里、LLM为何调整都有日志；能完成训练/复核/重训；输出原ID/标签/概率/顺序正确；新进程和新目录加载模型；partial不混称成功。检查恢复可继续，不能只证明拒绝正确。普通、文本和图像的效果/耗时分别报告，不能只给平均成功率。

若原包恰好无故障，只能证明完整接入，不能说证实了失败恢复的因果收益；受控故障、缓存回放和真实性能分别记。性能不足优先利用B既有通用方法，不从重新追Task1历史最高分开始。

## 7. 真实账本和唯一写入交接

A已导出私有附件`handoff_to_b_20261127.zip`，含`ledger.sqlite`、`ledger_receipt.json`、`README.txt`，**不含API key、模型或任务包**。该附件不进入Git。请用户把它与本说明一起转发。

```text
zip SHA-256:
ec729117abd816003566892bcf2fa4e4b8323592f21a3bb5f93322a462b1c3f9
解压 ledger.sqlite SHA-256:
7bac5c701762a28a4116c889ec839ef36665b34f1cc749a407d4ce8e63a8b6ea
```

导出是SQLite一致性backup，全部请求行SHA逐条与A原账本一致。A状态：526请求、17unknown、0pending、已知课程折算¥8.428396；完整课程费用/学校实际扣费仍未知。**SA26新增训练0、API0，尚未占用任何sa26_*付费scope。** A当前没有训练/API队列，也停止写账。B核对哈希后接管唯一写入。

B先核对自己的原始增量，尤其PR记录但A未收到的unknown：

```text
ID: b77d6d15-a6cc-4eb5-a1d7-6fcd8a830070
scope: joint-nov27-d58
原行 SHA-256: fe877e5bf1271dc9fa3b9f46e851aa79f7453def8f0e874793600f44fdf7ea92
费用/token均null；预留¥0.010092不是实际费用。
```

B文档另有两次契约抽取合计¥0.001096、早期反馈¥0.003963称待合并。用B自己的原请求/原SQLite行核对，不根据摘要造行。按请求ID去重、同ID核整行、保留原transitions与来源；相同ID不同内容要审查，不能直接覆盖，也不能把两个summary金额相加。`TaskBudgetLedger.import_history`是审计JSON导入接口，**不要直接把SQLite当JSON传入**。

没有原始行的项目记为待导入缺口，不让此事阻断本地开发；真实付费前使用核对后的同一非空累计账本与快照。不要创建空/模拟账本绕过历史，未知继续null，不等官网对账。附件含完整A历史，而不是B最近357条旧版快照。

## 8. 最终交付与汇报要求

每个小批次先保存计划、复跑脚本、预算；结果新目录；失败原记录保留。代码/脱敏证据进Git，数据包/模型/密钥/SQLite不进Git。提交时注明运行源码SHA，模型权重/依赖/编码器来源单独打包，不用git提交大权重。

给用户汇报采用：**瓶颈→对应结构→做了什么实验→证据支持到哪一步**。逐项列完成/错误完成/未决、实际fit、API/token、已知课程费用、耗时和未核验部分。不要把410项接口检查、规则存在或缓存命中说成准确率提升。B已有Task2 72.77%/独立内折70.40%的成绩可作为对应历史版本证据，不自动迁移为SA26最终成绩。

现阶段仍在开发，不因接手就冻结。优先级：**SA26真实验证和审计→P2两个通用适配→不同真实来源三模态原包验收→稳定版本交付与演讲材料更新。** 预计剩余约7–12有效小时，取决于首轮自然LLM回复、数据准备和失败定位；不是保证工期。课题范围、正式测试/封存任务授权不因本交接自动改变。

队友最后交回：代码分支/提交、每批plan/protocol/results/completion、真实账本及新增unknown清单、可搬迁模型和预测、清楚的支持边界/失败记录；更新`docs/CURRENT_AGENT_STATE.md`和`docs/EXPERIMENT_PROGRESS.md`，让任何一位Codex/DeepSeek能直接接续。
