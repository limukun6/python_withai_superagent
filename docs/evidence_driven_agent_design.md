# 证据驱动、预算约束的算法 Agent：设计与实验方案

日期：2026-11-17。状态：设计提案，尚未实现或执行；本文数值为拟预登记参数，不是实验结果。

范围说明：本文是搜索、诊断与选模子系统的方案。覆盖任务理解、AI判断错误、能力组合、统一运行内核和交付验收的整体主方案见 [system_agent_architecture.md](system_agent_architecture.md)。系统实现优先遵循主方案，本文实验参数仅用于对应机制验证。

## 1. 决策摘要

采用“共同基线 → 诊断 → 可检验假说 → 受控修改 → 实验 → 假说更新 → 一次性候选复核”的闭环。LLM 负责提出假说和选择动作；程序负责数据隔离、指标计算、预算、执行、证据有效性检查与最终选择。保留 one-shot 与无需 LLM 的规则方案，不预设复杂 Agent 必须获胜。

核心研究问题：在相同实验与资源上限下，更具体的反馈是否改善实验选择？独立内部复核是否改善最终选择？按需触发 LLM 是否降低成本而保留表现？

本轮先保持现有 14 个候选不变，验证反馈机制；随后再扩展可修改的表示和算法。不能同时增加候选能力、预算和反馈，然后把收益都归因于反馈。

“假说记录、工具反馈、搜索分支”属于已有研究方向。本项目的贡献定位为：针对有限预算分类任务，实现可追溯的诊断与复核，并通过受控实验解释何时有效。参考 AIDE（https://arxiv.org/abs/2502.13138）、Reflexion（https://arxiv.org/abs/2303.11366），不宣称首次提出这些思想。

## 2. 为什么要改变当前设计

依据：`runs/agent_architecture/20261117T023216973488Z` 的 24 组运行。该批代码哈希与当前 superagent 源码一致；24 组预测和模型选择均可复算；60 次 LLM 调用正常完成。

| 观察 | 支持的诊断 | 尚不能证明 |
|---|---|---|
| 文本 seed=11：C 从 1.5 降至 0.05，训练/搜索差距缩小，但搜索准确率从 0.6964 降至 0.6861 | 差距大不等于加强正则化必然有效 | 正则化永远无效 |
| 表格 seed=22：feedback 选中的配置内部 0.7363、官方验证 0.7355；one-shot 选中的配置内部 0.7301、官方验证 0.7424 | 内部单次排序有不稳定性，需独立复核 | 规划一定比选择器更差 |
| 6 个 depth=8 训练均跑满 120 秒后超时，其中 random 1、feedback 3、reviewed 2 | 资源约束影响可实现的候选质量 | Main 超时由 LLM 挤占时间引起 |
| reviewed 18 次审查仅改了 2 次提案 | 相同信息下追加审查的边际收益有限 | 审查没有任何价值或因果影响 |

当前还存在语义缺口：文本候选实际是词/字符 TF-IDF 差分加 LogisticRegression，但部分 LLM 理由称其为“相似度”“词 unigram”。应提供真实 pipeline 元数据与任务目标。文本对不自动等于相似性任务；顺序互换后是否翻转标签必须来自任务说明，不能只凭输入为两段文本猜测。

## 3. 范围与非目标

第一版实现现有两类任务上的可靠搜索闭环；通用接口不依赖任务名称。先使用受控候选注册表，不开放任意代码执行。不增加 reviewer 常驻角色，不训练/微调 LLM，不同时做大量新模型对比。

后续按顺序支持混合数值表格与多分类、普通文本、图像基础方案。现有 CatBoost 的二分类 Logloss、文本对的 `1-y` 增强及特定读取格式不能被直接宣传成支持这些任务。

## 4. 数据边界与评价协议

### 4.1 三层开发划分

仅将官方 train 按组、分层划分为：

- F（fit），约 60%：拟合预处理器和分类器。
- S（search），约 20%：搜索、诊断、假说更新；明确承认会被反复适应。
- C（confirmation），约 20%：冻结候选名单后，由程序做一次内部复核；LLM 在搜索中不可见其标签、画像或分数。

实施时以分组分层 5 折：指定 1 折为 C、1 折为 S、余下 3 折为 F。保存实际比例、各类样本数和 group ID。类别样本或组数不足以有效分层时，显式返回不支持该划分，不静默泄漏或把重复行拆开。对重复表格行、反序文本对采用现有分组；若任务提供实体 ID，可进一步按实体分组。该规则不能检测所有语义重复，作为限制记录。

官方 validation 由外部评估器在模型和预测方案冻结后评分，不回传 Agent。test 只在最终提交阶段按任务说明使用。官方 validation 早已被开发过程看过，结果只能称开发证据，不能称全新泛化评估。

所有 v2 对照都使用相同 F/S/C，连不使用 C 的方法也只能在 F 上训练；这样复核消融只改变选择证据，不改变训练量。由于训练比例改变，v2 必须重跑基线，不能和旧 80/20 主实验直接作配对因果比较。

### 4.2 冻结与复核

搜索结束后，把成功候选按 S 准确率降序、ID 升序排序，固定前 2 个。冻结 candidate IDs、模型文件哈希、预处理器和选择规则，再开放 C。

- C-off：直接选 S 最优候选。
- C-on：两个冻结候选在 C 上评分。令 a 为 S 第一名、b 为第二名。只有 `acc_C(b)-acc_C(a) >= 0.005` 才换成 b，否则保留 a。只有 1 个成功候选时保留它。
- 同时对两者在 C 上的正确/错误差做按组配对 bootstrap，记录描述性区间；第一版不将“区间不跨零”设为换模门槛。
- 0.005 是预设工程阈值，不是显著性界线，必须在正式运行前锁定。C 仍然参与选择，因此它的胜者分数不是无偏泛化分数。
- 评分后不能修改候选、重新搜索、试第三个候选或再访问新 C。当前任务的 C 结果不能写进可供后续搜索读取的记忆。
- v2 机制实验不 refit，直接输出原 F 上训练的模型。最终产品若改成全训练集 refit，必须统一所有对照、另行验证并冻结，不能混报。

## 5. 软件架构

```text
TaskSpec + train
    │
    ├─ Data adapter / group splitter → F / S / sealed C
    ├─ Pipeline registry → 候选真实能力与约束
    └─ Task profile（只基于 F，必要时含 S 类别分布）
             │
       Shared warm start（两项预设基线）
             │
       Search controller ← Evidence store / hypothesis ledger
          │        │
          │        └─ Deterministic diagnostics（F/S）
          ├─ LLM proposer（按策略调用）
          ├─ Action validator + Budget controller
          └─ Worker（隔离训练、预测、成本计量）
             │
       Freeze shortlist → Confirmation selector（唯一可见 C 的模块）
             │
       Freeze artifact → External evaluator（官方 validation）
```

### 5.1 TaskSpec 与 CandidateSpec

TaskSpec 必须给出：输入模态、目标含义、标签集合、指标、分组规则、允许的增强、资源限制。信息来源为任务说明和 schema，不是数据集名称。

CandidateSpec 必须给出：完整 pipeline、预处理 fit 范围、分类器、参数、支持标签类型、预计资源信息及来源、允许修改维度。例如：

```json
{
  "id": "word_char_C0.7",
  "representation": "word(1,2)+char_wb(3,5) TF-IDF; vector(A)-vector(B)",
  "estimator": "LogisticRegression",
  "params": {"C": 0.7},
  "augmentation": "swap A/B and invert label only when TaskSpec permits",
  "fit_scope": "F only"
}
```

这些语义信息应对所有使用 LLM 的策略相同开放，避免把“修正候选说明”的收益误归因于诊断反馈。

### 5.2 Evidence store

每次成功试验保存：F/S 行 ID、预测、真实标签的本地引用、可用时的预测概率、类别顺序、总体和每类指标、训练/预测耗时、模型配置和哈希。原始文本默认留在本地；发给 LLM 的是聚合证据。worker 不接收 C 或官方 validation。

概率不可用时，置信度分析标为 unavailable，不能把决策分数随意当概率。若以后做校准，校准必须在 F 内完成，不能在 S/C 上拟合。

### 5.3 Diagnostic engine

第一版提供确定性、可复现工具：

| 工具 | 结果 | 约束 |
|---|---|---|
| overall_metrics | accuracy、class recall、F/S 差距 | 差距只是观察，不自动等于过拟合根因 |
| slice_errors | 类别、长度分位、缺失程度分组的错误率和样本数 | 分组边界在 F 上定义；不按 S 错误率任意切出“最差子群” |
| paired_predictions | 两候选分歧率、a对b错/b对a错计数、配对差值区间 | 同一 S 行与同一 group；区间只作自适应搜索诊断 |
| runtime_status | 成功/失败、训练耗时、是否达限、候选参数 | 区分 API、训练和数据失败 |

子组默认至少 50 行、10 个组，否则只记录样本不足；阈值在 Pilot 前配置。无法满足时诊断工具返回 unavailable。多次搜索与多个子组比较不具备普通独立检验条件，不把局部区间当作显著性证明。

学习曲线、特征重要性和小规模训练探针放到第二版：它们需要额外拟合，必须扣训练次数和时间，不能伪装成免费诊断。

### 5.4 Hypothesis ledger

每次提案采用统一结构；scalar feedback 与 rich feedback 使用相同结构，仅可见证据不同：

```json
{
  "action": "train",
  "candidate_id": "word_char_C0.7",
  "parent_id": "word_C0.7",
  "evidence_ids": ["trial_1:overall", "trial_1:slice_length"],
  "hypothesis": "字符特征有助于当前短文本子组",
  "changed_factor": "representation",
  "expected_direction": "search_accuracy_increase",
  "target_slice": "short_text",
  "alternative_explanation": "单次划分噪声",
  "reason": "保持C不变以比较表示变化"
}
```

校验器只保证 ID、字段、证据引用、资源和控制变量是否一致；不能宣称程序已证明文字推理正确。改变多个维度时标为 exploratory，不纳入“单因素假说被证实”的计数。

执行后更新状态：pending / provisional_support / not_supported / inconclusive / execution_failed。方向符合且提升至少 0.005 只能称“该搜索集上初步支持”；最终外部评价才决定是否迁移。ledger 只记录当前任务的 F/S 经验，跨任务记忆在首轮实验关闭。

## 6. 搜索与资源策略

### 6.1 共同起点

按 TaskSpec/schema 选择两个预登记基线；所有比较臂相同：

- categorical_tabular：logistic_C1.0、catboost_depth4_reg10.0。
- pair_text（已确认排序语义）：word_C0.7、word_char_C0.7。

这是工程基线，不声称未知任务最优；新模态需要另建通用基线表。共同起点减少“第一下猜得不同”的混淆。没有 LLM 的固定策略也使用它们。

正式 v2 单次运行最多 6 次训练启动：共同起点 2 次＋后续 4 次。失败、超时、重试训练均占名额。相较旧三次实验增加了预算，因而所有对照重跑；不把增加次数本身归因于新机制。

### 6.2 策略定义

- D：deterministic portfolio。无需 LLM，按冻结的类型级列表执行后续 4 项。
- O：warm-start one-shot。看到共同起点的相同反馈后，一次安排后续 4 项；之后不看新反馈。这是 v2 基线，区别于旧版零试验前的一次规划。
- F：scalar feedback。每轮看总体准确率、召回率、gap、状态与耗时；使用统一结构化提案格式。
- R：rich feedback。与 F 相同，只增加预定义 slice、配对预测诊断；没有资源 gate 或额外训练能力。
- G：gated rich feedback。R 基础上，事件触发才调用 LLM，否则执行已排队计划。用于第二阶段比较。

所有臂共享相同候选池、真实 pipeline 描述、随机种子、训练超时和全局硬件配置。

D 的后续顺序在 Pilot 前写入配置，不根据 v2 官方 validation 选择。建议类别表格为 extra_trees_leaf2、catboost_depth6_reg3.0、logistic_C0.25、catboost_depth6_reg10.0；文本为 word_C1.5、word_char_C1.5、word_C0.3、word_char_C0.3。选择理由为覆盖模型族/表示和正则化范围，承认基于已知开发任务设计，必须有后续迁移评估。

### 6.3 按需触发（G）

共同起点后调用一次 LLM，生成剩余队列。后续出现下列任一事件才重新规划：

1. 队列候选失败/超时，或被预算控制器判为不可执行。
2. 连续两次成功试验都未使 best_S 增加至少 0.005。
3. 同一个预定义、样本量合格的子组，在连续两次成功试验中错误率均比总体高至少 0.10。

同一组 trial IDs 的同一证据指纹只触发一次；不因读取同一诊断而无限自我反思。G 在没有触发时不作 LLM 调用。超过调用预算或响应失败时标记策略退化，不隐藏 fallback。

G 首版只省调用，不新增低保真筛选或成本模型，以便隔离 gate 的作用。若以后加成本预估，则增加独立 C-cost 消融。

### 6.4 预算控制

拟配置：每次训练 180 秒；6 次启动；每运行最多 6 次 LLM 逻辑调用；每调用输出上限 16384；请求超时不超过 720 秒且不超过剩余时间；每运行总墙钟上限 3600 秒。训练上限从旧 120 秒调整是新协议变动，不是修复旧数据。

180 秒仅为待 Pilot 验证的统一上限，不保证 depth=8 能完成。Pilot 如需调整，所有臂共同更新并冻结。不要按正式结果临时放宽某个策略。

另设名义总 token 上限 60000、每运行人民币上限 1 元（均为提案，实际执行前核验官方价格）。调用前计算已知输入加输出最坏预留；若没有可信 tokenizer/上界，不能声称 token 上限是数学硬保证，应记录估计与超额情况。费用同理：平台价只用于开发暂估，课程记账必须用规定的官方历史标准价格。

保留最终落盘和复核时间窗口（默认 30 秒，Pilot 计时后冻结），搜索截止后不再启动新训练。训练超时使用 min(180秒, 剩余搜索时间)，分别记录计划上限和实际可用上限。网络请求与重试也必须检查剩余时间。

请求超时可能已被计费；不能只把 unknown 写进日志就忽略。第一版遇到用量不明停止后续付费调用，保留 incumbent；未来若支持重试，先为未知请求保守计入预留费用，再重新检查预算。网络故障不能当作算法假说被否证。

## 7. 伪代码

### 7.1 主控制器

```python
def run_agent(task_spec, official_train, policy, protocol):
    F, S, sealed_C = group_stratified_split(official_train, protocol.seed)
    registry = build_registry(task_spec)  # 不传任务名作为算法路由依据
    budget = BudgetController(protocol)
    store = EvidenceStore()
    ledger = HypothesisLedger()
    profile = profile_fit_data(F, task_spec)

    for cid in shared_warm_start(task_spec):
        execute_train(cid, parent=None, F=F, S=S,
                      registry=registry, budget=budget, store=store)

    # 即使一个 warm start 失败，也保留相同试验名额和真实状态。
    snapshot = expose_evidence(store, policy.visibility)
    policy.initialize(profile, registry.metadata, snapshot, budget.snapshot())

    while budget.can_start_search_action() and store.train_starts < 6:
        diagnostics = deterministic_diagnostics(store, F, S)
        context = expose_evidence(store, policy.visibility, diagnostics)
        action = policy.next_action(context, ledger, budget.snapshot())
        # next_action 是统一入口；D/O 可无新调用，F/R 每轮调用，G 按事件调用。
        # BudgetController 在真实 API 请求前预留，在获得 usage 后结算。
        if action.kind == 'stop':
            break
        checked = validate_action(action, registry, store, budget)
        if not checked.valid:
            log_invalid_action(checked)
            # 最多一次带错误信息的纠正请求；也占调用和预算。
            action = bounded_repair_or_mark_degraded(policy, checked, budget)
            if action is None:
                break
        result = execute_train(action.candidate_id, action.parent_id,
                               F, S, registry, budget, store)
        ledger.update(action, result, visible_diagnostics=context.diagnostics,
                      scope='search_only', causal_claim=False)

    shortlist = freeze_top_k_successful(store, k=2,
                                        order=(-search_accuracy, candidate_id))
    if not shortlist:
        return export_failure_record(store, budget)
    selected = confirm_once(shortlist, sealed_C, protocol.confirmation)
    # 不 refit；复核结果不再回到 policy。
    artifact = freeze_and_export(selected, task_spec, store, ledger, budget)
    return artifact

def externally_evaluate(artifact, official_validation):
    assert artifact.is_frozen
    return score_predictions(artifact, official_validation)  # Agent 无权读结果
```

scalar 策略的 ledger 也只能引用其可见证据，不能通过假说状态或理由间接获得 rich 诊断。所有策略可在后台生成完整诊断供事后分析，但后台材料不得进入受限策略上下文。

### 7.2 隔离训练与诊断

```python
def execute_train(cid, parent, F, S, registry, budget, store):
    reservation = budget.reserve_training_start()  # 启动即扣次数
    artifact = isolated_worker(
        pipeline=registry[cid], fit_data=F, predict_data=S,
        timeout=budget.allowed_training_seconds(), model_seed=42)
    # timeout/failed 同样落盘，标记训练开始与结束时间。
    budget.settle_training(reservation, artifact.actual_seconds)
    store.append(cid, parent, artifact)
    return artifact

def deterministic_diagnostics(store, F, S):
    slices = predefined_slices(boundaries_fitted_on=F)
    for run in store.successful:
        summarize_errors(run.predictions_S, S.labels, slices,
                         min_rows=50, min_groups=10)
    for challenger in store.successful_except_incumbent:
        paired_compare(challenger, store.incumbent,
                       rows=S.row_ids, groups=S.groups)
    return aggregate_evidence_with_ids()
```

### 7.3 一次性复核

```python
def confirm_once(frozen_top2, sealed_C, enabled):
    a = frozen_top2[0]
    if not enabled or len(frozen_top2) == 1:
        return a
    b = frozen_top2[1]
    assert shortlist_and_model_hashes_are_frozen()
    pred_a = predict(a, sealed_C.features)
    pred_b = predict(b, sealed_C.features)
    delta = accuracy(pred_b, sealed_C.labels) - accuracy(pred_a, sealed_C.labels)
    ci = paired_group_bootstrap(pred_a, pred_b, sealed_C, seed=42, repeats=1000)
    log_confirmation(delta, ci, note='descriptive; C participates in selection')
    return b if delta >= 0.005 else a
```

### 7.4 G 的触发控制

```python
def gated_next_action(state):
    triggers = detect_failure_or_stagnation_or_persistent_slice_error(state)
    triggers = remove_already_handled_evidence_fingerprints(triggers)
    if not state.queue or triggers:
        if not state.budget.can_request_llm():
            return next_valid_queued_action_or_stop(state)
        state.queue = llm_plan_remaining_actions(state, triggers)
        mark_trigger_fingerprints_handled(triggers)
    return state.queue.pop_next_untried_candidate()
```

## 8. 实验设计与预登记

### 8.1 H1/H2/H3

- H1：R 比 F 能在相同训练名额下找到更好的候选，而非仅生成更长理由。
- H2：独立 C 复核能减少内部 S 排序不稳定造成的选模损失。
- H3：G 比 R 降低 LLM 消耗，同时在独立评价上的损失小于预设可接受范围。

允许三项假说均不成立。架构选择依据实际准确率、成功率和开销，不以证明创新有效为目标。

### 8.2 P0：实现与可行性 Pilot

两任务，seed=101，D/O/F/R，共 8 次运行，每次最多 6 次训练，最多 48 次训练启动。只用于流程、耗时、失败处理、诊断可用性检查，不纳入正式均值。

通过标准：数据权限隔离、预测行对齐、结构化提案可执行、usage 可核对、最多 6 次训练、C 仅在冻结后开启、最终输出能独立重载预测。训练超时可作为正常结果，但基础设施故障须解释。修复后冻结源码、配置、依赖与 prompts 的哈希。

不要自动无限重跑 Pilot。新批次必须有明确修复点；失败批次保留。

### 8.3 E1：搜索机制对照

两任务 × seeds [41,52,63,74,85] × D/O/F/R = 40 条轨迹，最多 240 次训练启动。

新 seeds 只是减少对旧 seed 的专门适应，并不使相同数据集变成新任务。各策略相同 seed 使用相同 F/S/C 和模型种子。正式执行采用预先平衡的策略顺序，统一硬件负载；不得同时跑多个重训练以免影响超时概率。

主要比较：R−F（丰富诊断的增量价值）。次要比较：F−O、O−D。E1 首先统一 C-off，避免搜索和选模效果混淆。

共享起点可以复用相同模型文件来保证候选完全一致，但必须对每个逻辑运行计入相同训练名额与其标准成本；物理缓存命中时间不能当作端到端加速结果。正式墙钟比较优先各自实际执行完整路径；缓存重放仅用于逻辑机制分析。

### 8.4 E2：选择器消融（复用 E1 冻结候选，无新增 LLM）

对每条 E1 轨迹固定同一 top2，生成 C-off 和 C-on 两个选择结果；最多 40×2=80 个 C 候选评分，属于预测而非重新训练。冻结这两种选择后再由外部评估器评分。

这形成 policy × confirmation 的设计，但只有 40 条底层搜索轨迹，不能把 80 个选择结果当作 80 次独立搜索实验。每个开关差值在同一轨迹内配对计算。

记录：换模比例、外部改善/损害比例、准确率变化、复核耗时。假设当前候选集合为 U，外部评估器可在所有协议冻结后计算 `max_{u in U} val(u) - val(selected)` 作为“集合内选模损失”，只用于事后解释，不向 Agent 回传，不是部署可用的 oracle，也不是搜索空间全局最优。若担心过度查看官方 validation，可只计算 top2 的该诊断。

R 对 F 的 C-off 差异衡量搜索+S选择的综合结果；确认开关在固定轨迹上的差异单独衡量选择器。两者共同分析，不能用 best_S 上升代替外部性能提高。

### 8.5 E3：按需反馈消融（第二阶段）

G 与 R 比较，均开启相同 C-on，使用 E1 相同 5 seeds、同 6 次训练上限、同候选空间、同 token/RMB/墙钟上限。G 新增 10 条轨迹、最多 60 次训练；只有代码、硬件和协议未变时才能复用 E1 的 R，否则重跑 R，最多新增 20 条轨迹。

预设工程目标：G 的总体 token 减少至少 30%，每任务平均准确率损失不超过 0.005，任务完成率不下降。这个目标不是凭 n=5 就能证明的统计非劣界限；报告每 seed 损失和不确定性。若不满足，不强行采用 gate。

若后续必须加入成本预测模块，独立做 G+cost 与 G−cost 对照，保持候选和名额一致。小规模探针计入训练次数；低保真分数只决定晋级，不能直接与全量训练分数混排。

### 8.6 E4：迁移与通用性（冻结设计后）

预先指定至少一个未参与规则设计的新任务，优先验证混合表格或多分类接口；数据来源、许可证和划分在使用前记录。用同一冻结 Agent、无任务名称特判执行。新模态数据适配完成后，所有方案使用相同适配器。

如果只新增一个任务，不宣称广泛通用。图像支持作为后续独立里程碑，避免两人团队同时扩张所有模态。课程正式 unseen task 必须使用提交时冻结版本；开发迁移任务不能冒充正式 unseen 成绩。

### 8.7 可执行范围与工作量

本文仅制定方案，不启动付费调用。先实施 P0，再登记 E1/E2，最后视结果推进 E3/E4。每运行暂拟 1 元封顶，对应 P0 名义最多 8 元、E1 最多 40 元、E3 最多 10–20 元；这是分批最坏上限，不是预测花费或已授权支出，也不替代课程对每评价任务的计费口径。实际预算和官方单价核验后再启动。

E1 训练时间最坏上界约 240×180秒=12小时，另有 API、预测与启动开销；实际未必达到。先以 Pilot 的分任务时长估计工期，不承诺原主实验的约两小时可完成全部新实验。

## 9. 指标、统计与结论规则

主要指标按任务分别报告：官方 validation accuracy 的每 seed 值与配对差值；均值、样本 SD；计划/完成/策略有效/usage完整四种计数。种子重复共享数据集，不能视作五个独立任务，不能把多个 seed 的差值简单当作广泛泛化证据。

次要指标：总 token（输入/输出/可用时推理）、费用已知部分与未知上界、LLM 调用次数、训练启动数、成功/失败/超时数、墙钟与训练/API分项、每轮 best_S、假说被初步支持比例、重复无效尝试率、最终换模率。

运行失败按预先定义的部署策略处理：有 incumbent 则输出并标记退化；无模型则未完成。所有计划运行保留分母；纯策略有效子集结果与包含退化交付的结果分别展示。用量不明不能记为零费用或悄悄剔除。

预先指定 R−F 为主要机制对比，其他为次要/探索性；小样本不靠大量 p 值寻找“显著”。如给配对 bootstrap 区间，说明按 split 重复的相关性和样本量限制。样本级按组 bootstrap 只描述给定模型的预测差异，不覆盖 LLM 搜索或训练随机性。

采纳规则：先确认功能与账本，再看每任务效果和成本。平均提高但某任务稳定退化应明确说明，不用跨任务平均掩盖。若 R/F/G 没有优势，保留 O 或 D，把资源投入表示、适配与稳健选模；负结果仍是有效 experimental conclusion。

## 10. 与现有代码的接口和实施顺序

| 文件/模块 | 工作 | 验收 |
|---|---|---|
| 新 `protocol.py` / 配置文件 | 记录预算、策略、阈值、seeds、候选说明、全部哈希 | 任意运行可还原实际配置；不再只哈希 superagent/*.py |
| 新 `splitter.py`、`data.py`适配 | F/S/C 隔离，TaskSpec与分组 | C/val不可被worker或planner读取 |
| `experiment_worker.py` | 行ID、S预测/概率、类别映射、计时与错误类型 | 重载模型预测一致；无预处理泄漏 |
| 新 `diagnostics.py` | 固定子组与配对预测工具 | 小样本、缺类、无概率均可正确降级 |
| 新 `hypotheses.py` | 结构化提案与状态更新 | 错误引用/多变量修改/执行失败不伪装成因果支持 |
| 新 `policies.py`、`benchmark.py` | D/O/F/R/G 和共同起点 | O无后续反馈、F/R只有证据可见性差异、G无重复触发 |
| 新 `selection.py` | 冻结top2与一次性C复核 | 只能访问一次，不能重新开放搜索 |
| `experiment_llm.py` | 未知用量预算、超时与剩余时间、结构校验 | 每次真实请求计量；停止后不再付费请求 |
| 分析/校验脚本 | 按协议生成计划笛卡尔积，校验预测和日志 | 缺整个seed/任务也保留分母；从标签复算准确率 |

两人分工建议：A 负责数据隔离、worker、diagnostics、selector；B 负责策略、预算、协议和分析。双方共同复核 TaskSpec、候选语义和统计结论。

建议实施里程碑（非承诺工期）：

1. M1：修正旧报告、存档原始结果；冻结现有可复现版本。
2. M2：数据隔离与预测证据；先跑 D 的无 API 端到端流程。
3. M3：O/F/R 与一次性复核；本地模拟 API 完成权限/预算/失败测试。
4. M4：P0 可行性与耗时验证；冻结 E1/E2 协议。
5. M5：正式比较与配对分析；按结果决定 G 是否值得实现。
6. M6：迁移适配、冻结最终版本、重跑正式报告所需结果、整理交付。

最低测试包括：组不跨集合、预处理只fit F、C/val访问守卫、概率类别顺序、预测行对齐、配对bootstrap保留组、失败占名额、API未知费用阻断、O只调用一次、G触发去重、top2冻结后不可修改、缺失运行仍进分母、CLI导出模型可重载。

## 11. 交付物与诚实的成功标准

每个批次保存：protocol、源代码/依赖/提示词哈希、split、TaskSpec、候选说明、events、逐调用usage、predictions、diagnostics、hypotheses、trials、shortlist、confirmation、selection、summary。报告汇总表必须能从这些文件自动生成。

最终应能回答：

1. 诊断是否真的改变下一步实验，并提高了外部表现？
2. 内部搜索更好但外部不佳时，是候选能力还是选模不稳定？
3. 增加机制带来的成本能否被收益支持？
4. 在不修改任务专用代码的情况下，新任务能否完整运行？

成功不是必须击败 one-shot，而是获得一个可复现、能解释失败、按证据决定复杂度的最终 Agent，并满足课程要求的同版本自主运行与完整记账。
