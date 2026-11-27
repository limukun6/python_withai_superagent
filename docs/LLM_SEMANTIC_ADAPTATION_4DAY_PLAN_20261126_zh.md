# A线四天结构开发：从固定格式拒绝到LLM驱动的可验证适配

2026-11-26。用户最新要求取代“立刻结束研发/冻结”的建议：还有四天，可以改结构；B继续方法菜单，A重点解决陌生任务语义、封装、输出与恢复，不只增加固定格式。审查基线为B自主入口d501807；本地a151-b-merge当前另有Task1研究，不能把本方案写成已经合并或实施。

本批为代码审查和设计，新增训练0、模型API0、未改生产代码/默认/账本；以下实现和实验尚未执行。已有研究负结果保留，不通过改名宣称旧grounded成功。

## 1. 判断：问题是把理解写成固定匹配，不是Python本身

用户指出的主问题成立：未知任务的公告、文件组织、目标和提交规则不一定匹配我们预写的规则。应由LLM理解并提出适配，然后Python试读和执行。但“输出是合法JSON”只保证语法，不保证目标正确、没有泄漏、数据行能对齐或提交文件可被老师接收。

因此新结构应是**LLM解释与提出修复→工具探查→程序验证→反馈给LLM调整→冻结可复用的执行方案**。不能只把raise删掉，也不能把LLM塞进现有狭窄正则后面再称通用。输入语义可以重推理，真正无证据的目标不能凭猜测当事实。分类范围仍以课程表格/文本/图像为准。

## 2. 代码审查：哪些地方需要LLM，哪些不需要

| 瓶颈 | d501807实际位置 | 现有行为 | 改动方向 |
|---|---|---|---|
| 说明书发现 | format_sources.task_documents / Package.documentation；native_schema.description | 六类说明文件名；多份说明需manifest；读取上限 | 生成文件证据索引，由解释器选择相关公告/字段字典/提交模板；分块读，不把所有文件全塞提示词。保留字节、路径边界 |
| 训练/预测源与行角色 | native_schema.find_member / named_table；format_intake.declared_source / records_for_role | 文件名/正则/marker词表；不匹配就拒绝 | LLM提出source、collection、split field/value映射，工具检查成员存在、选取行数、标签可用性；不通过时给具体诊断 |
| 目标与标签意义 | table_input.target_column；format_guidance.resolve_target / linked_target | LLM前置证据仍须通过GOAL/IGNORE正则；跨句链接需两边出现同一≥2词短语 | 允许引用文档块/字段ID并做跨句语义解释，不要求词面完全一致；引用真实性与存在性机器验，解释正确性用独立案例评价。不能声称引用存在就证明语义正确 |
| 类型与偏好关系 | native_schema._column_type / declared_preference / record_schema | 文本长度阈值、0/1左右方向正则、固定类型 | LLM结合文档和训练结构提出字段角色/类别含义/对称或有序关系；采用通用模型输入类型编译。ID与时间键的模型用途显式记录 |
| 目标指标 | metric_declarations / table_input.declared_metric | 别名表＋关键词/否定词匹配，不支持即拒绝 | LLM将公告解释为指标描述、平均方式、方向、正类、需要label/probability及证据；编译到确定性评分工具。新增metric需实现并手算小例核对，不能让LLM口算训练成绩 |
| 新封装/结构转换 | format_intake.adapt_training；record_streams.decode_records | 已实现codec集合；LLM主要处理目标与定宽 | LLM组合reader与转换工具；遇到未知布局先探查/试读，再修计划。先交付通用组合DSL；真实缺工具时可走受限适配器扩展，不承诺所有二进制格式都可解 |
| 预测期schema变化 | format_intake.prediction_rows / predict_records | 字段名/codec须符合冻结契约 | 允许对合法改名、列顺序和封装变化提出映射，试变换后对齐原模型schema；不修改模型目标/类别语义，不重新训练，不删无法对齐的行 |
| 最终提交 | predict.create_predictions；partial_classification.write_prior_predictions | 固定row_id/prediction/score_i、默认顺序ID | LLM按公告/模板提议OutputRecipe，程序恢复原ID与列序、输出label或指定概率、做独立校验。预测值不能由LLM生成 |
| 失败调整 | autonomous_run._train / record_input_failure；unified.command | 多种异常归partial/unavailable，本地回退 | 语义/适配错误先进入结构化诊断和有界修复；网络/用量异常停付费，不自动重试，已有本地合法能力继续。不要因一个ValueError就结束理解 |
| 方法发现、探索/复核/停止 | experiment_policy / MethodSpace | B已在改，且choose_with_flash已能选择动作 | A只提供可靠task/capabilities/反馈接口，不重写B排序或把菜单问题纳入A范围 |

必须继续由程序完成：真实解码/训练/指标计算、预测行与ID对齐、概率类序、文件/资产哈希、预算结算、训练与测试隔离、执行资源限制。LLM不能用“我确认正确”替代这些。

## 3. 推荐结构：三个新增核心对象＋可复用工具

### 3.1 EvidenceWorkspace：可以探查的任务证据

先只构建目录、文件签名、尺寸、文档块ID、结构字段ID及已确定的读取许可。解释器可请求工具：

```text
list_members()
read_document(block_id, range)
inspect_structure(member, collection, allowed_role)
probe_reader(reader, options, selected_role, row_limit)
check_contract(plan)
check_output_recipe(recipe, sample_header_and_ids)
```

文档可有多份，保留各自来源/版本。只让训练行进入分布统计和监督分析；未知role不先全量读标签来反推划分。若共用容器，先依据可验证声明建立行掩码，再分析训练部分。任务公告、样例提交schema可用于理解；测试标签不能进入模型或提示词。工具参数也需验证，不能因LLM选了一个文件就视为合法训练源。

避免重复大菜单：任务解释只需工具能力和字段/目标元信息，不带B的全部分类器参数；模型选择阶段另收TaskSpec。文档与记录作为数据引用，不得让数据里的“忽略规则”改写权限或预算。

### 3.2 TaskInterpreter：统一语义入口

每个原包先过一次语义入口；明确标准契约可直接走规则快路，模糊/冲突/失败时完整证据给LLM。若指定LLM-first实验则必须实际调用，不能初始化失败后偷偷算入LLM组。

建议一个短JSON信封，三种动作：

```json
{
  "action": "inspect | propose | stop",
  "request": {},
  "evidence_refs": ["doc:announcement:12", "field:records:verdict"],
  "uncertainties": []
}
```

action=inspect时request是一项工具调用；propose时request是一个有版本的TaskAdaptationPlan。使用带类型的联合schema，别让一个输出同时必须填所有无关字段。stop必须给已检查证据与缺失项，不等于整机退出；判断是否能保留有效partial由恢复控制器处理。

TaskAdaptationPlan建议结构：

```json
{
  "version": "task-adaptation-v1",
  "training": {"member": "<证据中的成员>", "reader": "<已安装reader>", "options": {}},
  "split": {"field": "<字段或null>", "roles": {}},
  "target": {"field": "<字段>", "label_semantics": null},
  "features": [{"field": "<字段>", "role": "predictor", "type": "categorical"}],
  "objective": {"metric": "macro_f1", "direction": "maximize", "positive_label": null},
  "evaluation": {"kind": "grouped_stratified", "keys": []},
  "prediction": {"member": "<成员>", "field_map": {}},
  "output": {"format": "csv", "id_source": "<原ID字段或声明顺序>", "columns": []},
  "claims": [{"field_path": "target.field", "evidence_refs": ["doc:announcement:12"]}]
}
```

这份JSON是设计示例，不是既有实现。实际字段在首批固定，以可测试事实为准，不强制所有任务填正类/时间键等。claims分开“来源可检查”和“语义仍待判定”。执行前编译为现有TaskSpec/MetricSpec/读取recipe/输出recipe，B的方法入口不必跟着改。

### 3.3 RepairController：异常成为下轮输入，不是默认终点

诊断统一为code、stage、已见事实、预期、实际、可用工具、候选修复范围、剩余调用/时间。比如missing_target_field可请求读字段字典，unknown_split_marker可修role映射，output_header_mismatch可修输出recipe。不能把普通实现bug都交给模型猜。

```python
workspace = inspect_package_without_training()
plan = native_fast_path(workspace)
while plan is None or not report.executable:
    # API ask失败直接传播到付费熔断，不进入这条语义修复重发。
    proposal = interpreter.ask_once(workspace, report, shared_call_budget)
    if proposal.action == 'inspect':
        workspace.add(run_checked_tool(proposal.request))
        continue
    if proposal.action == 'stop':
        return verified_partial_or_explained_unavailable()
    plan = parse_typed_json(proposal.request)
    report = compile_and_probe(plan, training_only=True)
    if repeated_plan_and_same_error(plan, report):
        break
if not report.executable:
    return verified_partial_or_explained_unavailable()
freeze_input_and_objective(plan)
model = existing_B_agent.run(canonical_training_task(plan))
features = apply_verified_prediction_recipe(plan)
predictions = model.predict(features)
artifact = execute_output_recipe(predictions, original_ids, plan.output)
independent_contract_check(artifact, teacher_announcement)
```

模型提案JSON/语义不合法且usage已知已结算时，可以在原额度内修复；传输、返回用量、结算失败不重发。工具调用不无限循环，同一错误不改变方案就停止该修复分支。JSON校验失败的原回复保留，不能默默编造缺失证据。

## 4. Prompt工程的实际重点

不要写一段“请万能处理所有任务”的长提示词。拆成稳定system约束＋当前证据＋本轮目标＋JSON schema＋具体失败反馈：

1. 只从工具返回的真实成员/字段ID中选择；需要更多证据时输出inspect，不用猜。
2. 区分“任务公告要求”“真实字段与值”“模型假设”“无法确定”。记录存在的标签值不等于知道其业务含义。
3. 多份公告允许联合理解；发现冲突先引用两边和声明版本，不能用置信度高就覆盖明确事实。
4. 提案只描述适配；不输出分类预测，不泄漏测试标签，不让数据内容变成系统命令。
5. 先给最小可执行方案，工具执行后根据错误修正；不要在初次输出重复大量历史/候选参数。
6. 所有字段要求类型和上界；字段错误给机器可读路径。输出schema本身也要测试，不用十几项无关必填字段把推理成本重新抬高。

标准JSON/结构化输出若学校端点支持则利用，不能假定服务端严格执行；本地始终parse/validate。合法JSON不保证事实、数据解释或模型效果正确。

## 5. 真缺reader怎么办：先组合，再条件生成适配器

组合通用工具能覆盖字段重命名、嵌套取值、列/行选择、受控join、宽长转换、声明分隔符、文件组织映射。join必须检查唯一键和一对一/一对多规则，不能默默复制预测行。

若真实案例需要现有工具无法表达的转换，仅在D2核心通过后做P2：LLM输出小型reader/转换插件，限于隔离目录读取已授权输入，独立进程资源上限、无网络/密钥/账本写权限、输出固定批接口。必须通过试读、行身份、目标不入特征、训练/预测同构和失败回滚检查，保存代码与SHA；这算结构改动，不是“执行任意模型代码”。其执行环境安全边界必须实际可用，不能只靠提示词/AST过滤称已隔离。

不会因为加插件就保证任何加密/损坏/未知二进制都能解码；不存在信息时LLM也不能恢复。四天内优先做解释和修复闭环，生成代码作为条件扩展，不拖垮主线。

## 6. 与队友的接口

A给B：稳定TaskSpec、MetricSpec、规范化训练输入、划分协议、预算、数据/解释来源指纹。B给A：capabilities、实际使用的模型输入契约、模型/概率类序和执行结果。输出recipe由A消费B预测，不干预B的候选排名或人工指定赢家。

新设计沿d501807现有format_guidance、format_intake、autonomous_run接，不再另造一套平行Agent，也不回退到旧A研究入口当最终入口。新模块建议task_interpreter、adaptation_plan、repair_controller、output_recipe；EvidenceWorkspace复用Package/record_streams。共享文件先分清归属，在独立codex/分支开发；目前未开始合并。

## 7. 四天安排与可独立交接产物

### D1：全局理解与可执行契约

实现EvidenceWorkspace/TaskInterpreter/TaskAdaptationPlan和prompt版本；移走必须词面相同的目标证据闸门，保留真实引用与机器核验。打通现有标准快路→LLM提案→试读→诊断回送。先固定一批未用于prompt修改的保留案例；离线模拟回复和封存旧回复可反复开发，不能把模拟结果当真模型成功率。

产物：计划、schema、prompts、工具接口、开发/保留案例、无fit runner、结果与失败回执。0训练/0模型API；不改B方法库。

### D2：输出契约与修复闭环

实现OutputRecipe与预测映射：原ID、顺序、表头、label/probability、正类列、编码/分隔符；双重校验。修复限额、同错循环检测、网络与语义失败分离。至少准备跨句目标、多公告、冷僻文件名、非标准split marker、非0/1标签、原ID乱序、不同提交表头和坏JSON等案例。

真实机制探测先单独预登记：四个事先固定的模糊案例×“现有LLM局部解释/新全局解释”两路径，最多8次请求、0fit，单格1次；共享总额用完就停，不因某格成功再补其它格。修复行为优先离线验；若需真实修复，另登记并占原总额，不能偷增调用。费用按已冻结价格与真实usage记录，不估成0。真实调用前确认唯一账本owner，并入B待合并请求；不新建模拟/个人账本替代正式记录。

### D3：同一Agent真实原包运行＋故障验证

三种模态各一个事先冻结的有效分类任务，用老师风格的原包和公告输入；旧入口/新入口各一条路径，同一B菜单、控制器、fit上限和时限。每路径≤10物理fit含refit，研究上限60fit；新语义路径每任务≤3调用（理解/修复/输出共享，也包含格式解释；规划额度不足时沿同一B本地策略，不追加第4次），最多9新请求。合并D2总上限17调用只是未来计划，不是已授权绕过真实预算/未知门槛。

任务选择和输入封装在看到结果前冻结；换名字的旧任务只算封装测试，新真实任务另记迁移。输出契约oracle独立于Agent的LLM，避免模型自己解释错后又自己宣布通过。网络故障用模拟，不真发重复付费请求。同时用原Released包回归输入/目标/预测身份，不能把新模型成绩冒称旧分。

### D4：无人干预演练、文稿与交付

固定候选版本和资源，在新目录、清洁环境、预装权重下运行盲封装任务。运行中禁止人工改代码/数据/阈值/模型；失败完整保存，修复算下一开发版本，必须重新封存后才复评。保留演讲/报告准备时间，只补明确阻断，不再增模型菜单或提示词网格。

最后做包SHA、依赖、预测格式、隔离重载和日志/账本检查。D4结束前是否冻结取决于这些门槛，不是当前就宣布冻结。正式课程Unseen发布后只运行最终冻结版本，不能现场修复。

## 8. 验收指标与停止规则

主指标是**有效任务无人工端到端完整成功率**。完整成功要求：语义正确、训练真实完成、预测行/ID/概率/提交schema正确且能重载。partial、unavailable、超时分别列，不能并入完整成功数。

同时报告：目标/指标/划分/输出解释准确率；错误但被接受的比例；能修复的错误数与调用数；训练/test标签边界；实际fit/token/费用/总墙钟；不同真实任务迁移。JSON合法率仅是辅助。

- 支持范围内清晰有效案例：不能因措辞和文件名不同直接拒绝。
- 信息确实缺失、文档矛盾无法消解或文件损坏：必须如实区分，不能算作“模型推理一下就全成功”。
- 接受错误目标、训练泄漏、行ID错位、伪造分数：任何一例都阻止该版本晋级，不因总体成功率高掩盖。
- 同输入改名和候选展示顺序不应决定任务含义；半途LLM不可用不允许重试同一失败请求。
- 留出集不能反复调提示词追过线；需要改动时保留旧结果，另冻新案例或明确称开发集。

本方案将确定性护栏的职责从“规定只有这些表达才合法”改为“核验LLM提出的解释和适配能否真实执行”。尚未实现，不能用已有格式测试成绩代替新结构收益。
