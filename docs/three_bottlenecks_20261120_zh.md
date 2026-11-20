# 三个瓶颈的落地改动：任务语义、配置事实、散文核验

2026-11-20。按"先做任务理解、工具理解、推理归因这三条"的要求完成。全部为离线改动，**不产生 API 用量，不改变已有实验的判定口径**；新增 35 项测试。

## 1 任务理解：把"交换文本到底该不该翻标签"变成可检查的契约

### 原来的状态

`pair_semantics` 这个字段只在 `schema.py` 里被当作**分流开关**：

| 声明 | 实际加载成 | 后果 |
|---|---|---|
| 不声明 | `pair_text` | 按左/右偏好训练，但没人说过它确实是偏好 |
| `relation` | `text_tabular` | 变成两个独立文本特征，成对结构丢失（这是有意的多分类用法，见 `test_general_extensions`） |
| `symmetric` | `pair_text` | **按偏好训练，错** |
| `similarity` 等任意字符串 | `pair_text` | 静默接受，没有任何提示 |

更关键的是候选库：`pair_text` 的 `original` 14 个、`expanded` 22 个候选**全部**声明 `swap_augmentation: True`，而该参数的实际含义是"把特征取负、标签取反再训一遍"（`models.py:115-120`）。这个增强只有在"交换后标签翻转"时才成立。

于是出现一个没有记录的错误路径：**一个对称任务（例如判断两条文本是否同类）落到 `pair_text` 上，会被用翻转标签的增强训练，且没有任何提示。**

### 现在的做法

新增 `superagent/task_semantics.py`，在训练开始前（连输出目录都不创建）做四项检查：

| 检查 | 内容 |
|---|---|
| `semantics_declared` | 必须是 `preference` / `relation` / `symmetric` 之一；未声明记为 `undeclared`，非法值记为 `invalid` |
| `semantics_matches_package` | `preference`/`symmetric` 必须加载成 `pair_text`；`relation` 必须加载成 `text_tabular`；不符即拒绝并给出原因 |
| `swap_augmentation` | 当语义是 `preserves`（对称）时，把所有假设翻转的候选**排除并记录**；若排除后没有候选，拒绝并说明 |
| `swap_policy` | 写进运行记录，供后续核对 |

同时新增**实测探针** `swap_behavior`：对已选模型在同一批内部搜索行上交换两列各预测一次，报告

- `mean_complementarity_error = mean(|p(a,b) + p(b,a) − 1|)`：接近 0 说明模型学的是"翻转"关系
- `mean_symmetry_error = mean(|p(a,b) − p(b,a)|)`：接近 0 说明模型对顺序不敏感

据此判定 `flips` / `preserves` / `neither`，并与声明的语义比对，写入 `summary.swap_behavior`。这从"相信声明"变成"测量行为"。

### 直接测出的结论

| 场景 | 结果 |
|---|---|
| 未声明语义 | `undeclared`，可运行，但记录"交换行为未验证" |
| 声明 `preference` | 通过，保留全部 22 个候选 |
| 声明 `symmetric` + 默认候选库 | **拒绝**：22 个候选全部假设翻转，没有安全的候选 |
| 声明 `symmetric` + `semantic` 候选库 | 通过，排除 22 个，剩 4 个不含增强的候选 |
| 声明 `relation` 但加载成 `pair_text` | 拒绝，原因写明类型不符 |
| 声明任意字符串 | 拒绝，列出允许值 |

### 在真实任务上的实测

对已发布的成对文本任务（`released_task2.zip`，未声明语义）跑一次 1 试验的本地运行：

```json
"swap_behavior": {"pattern": "flips", "rows": 400,
                  "mean_complementarity_error": 1.1e-13,
                  "mean_symmetry_error": 0.0724,
                  "selected_candidate_assumes_flipping": true}
```

补互误差接近 0、对称误差 0.07，判定 `flips`。

**这个结果不能当作"该任务确实是偏好任务"的证据。** 被选中的候选在构造上就用 `-X` 配 `1-y` 训练（`models.py:115-120`），它的预测必然反对称，所以 `flips` 是**结构性的**。运行记录里因此额外写了 `selected_candidate_assumes_flipping` 与一条 `interpretation_note`。探针真正能回答的是"声明与实际行为是否一致"，而不是"任务本身属于哪一类"。

## 2 工具理解：配置事实从"查表"改成"读训练出来的对象"

### 原来的状态

`candidate_facts` 靠一张 `{训练函数名 → 分类器名}` 的字典反推。它回答的是"这个候选通常会造什么"，不是"这次实际造了什么"；遇到注册表里没有的候选就直接返回 `None`，而 `contracts` 见到 `None` 会把提案判为"引用了未注册分类器"。

### 现在的做法

1. **训练进程记录实际拟合的对象。** `experiment_worker` 新增 `estimator` 字段：`observed_classifier(model)` 会沿 `Pipeline` 取最后一步、沿本包自己的包装类（`StylePairModel`、`SoftVoteModel` 等）取内部分类器、沿 `(vectorizer…, classifier)` 元组取最后一项，然后按类名映射到 `logistic` / `linear_svm` / `rbf_svm` / `catboost` / `extra_trees` / `dummy` / `naive_bayes` 等。
   - 只对 `superagent.*` 里的包装类做属性追踪：`ExtraTreesClassifier` 也有 `.estimator`，但那是它的基学习器模板，不是拟合结果。这一条由测试 `test_ensemble_base_estimator_is_not_mistaken_for_the_fitted_model` 守住。
2. **未知事实显式声明，而不是返回 `None`。** `candidate_facts` 现在返回
   `{'classifier': None, 'source': 'unverified', 'reason': '...'}`。
3. **契约相应放宽、判定相应收紧。** `ExperimentProposal.from_decision` 只对**已知**事实强制一致；未知事实允许提案通过，改由训练后比对声明与实际拟合对象。
4. **每次试验落一条核对记录。** `row['configuration_fact_check']` = `{declared, observed, observed_class, status}`，`status ∈ confirmed / contradicted / unverified / not_declared`。
5. **批次汇总计数。** `summary.configuration_facts` 给出四类计数（确认、矛盾、不可判、未声明）与适用范围说明。

这样，"模型说这是 SVM"不再只跟一张表对账，而是能跟**真正被 `fit` 的那个对象**对账；新增候选也不再因为"不在表里"被一概拒绝。

## 3 推理归因：散文里能查的三个断言，给一个有效分母

### 原来的状态

`audit_explanation` 只做两条英文短语检查（"完全打平"、"显著"），然后硬编码 `freeform_prose_fully_verified: False`。这个 `False` 没有分母，读者无法判断是"查了 3 条都没问题"还是"根本没查"。

### 现在的做法

新增 `superagent/prose_claims.py`，从 `reason` / `review` / `assessment.interpretation` / `assessment.uncertainty` 里抽取三类**高精度、可判定**的断言：

| 断言 | 判定方式 |
|---|---|
| 分类器族（"logistic regression"、"character n-gram SVM"、"CatBoost"…） | 与候选的分类器事实比对；`svm` 视为与 `linear_svm`/`rbf_svm` 都相容；事实未知则记 `unverifiable` |
| 试验引用（"trial 7"） | 超过已完成的试验数加一即矛盾 |
| 引用百分数（"64.63%"） | 与历史里任何一次 `search/train` 的 accuracy/score 在 0.01pp 内匹配为确认，否则记 `unverifiable`（不判矛盾：可能来自四舍五入或未记录的量） |

输出 `prose_claims = {claims_checked, confirmed, contradicted, unverifiable, informational, items, unique_contradictions}`，矛盾的条目同时生成 `prose_claim_contradiction` 警告并带上命中的原文片段。

**精度优先的取舍。** 一个分类器词如果出现在比较语境（`than/over/above/baseline/...`）或试验引用之后（`trial 1 logistic C0.05`），它说的是另一次实验，因此记为 `informational`、不计分。这样做的代价是漏检：它不会因为"text 里提到基线是逻辑回归"就报警。

### 在两个历史批次上的回放（零 API）

| 批次 | 已记录回答 | 计分断言 | 确认 | 矛盾 | 无法判定 |
|---|---:|---:|---:|---:|---:|
| `checked_loop_pilot_v1`（12 次决策） | 12 | 43 | **43** | **0** | 0 |
| `flash_error_feedback_v2`（4 次决策） | 4 | 3 | 2 | **1** | 0 |

唯一被抓到的矛盾正是文档里记过的那条事实错误：

```
43111/detailed  候选 char_C0.3（实际 logistic）
  hypothesis: "A char n-gram SVM with C=0.3 will match or exceed..."
  -> contradicted, observed ['logistic']
```

也就是说：在 43 条历史断言上零误报，同时抓住了已知的那一条真错。

`freeform_prose_fully_verified` **仍然是 `False`**，但 `check_scope` 现在写成"检查了 N 条机器可判定断言，其中 M 条矛盾、K 条无法判定"，并注明因果与方向性措辞不在覆盖范围内。这是把"没有证据"换成"有分母的部分证据"。

## 4 改动清单与测试

| 文件 | 性质 |
|---|---|
| `superagent/task_semantics.py` | 新增，语义契约与交换探针 |
| `superagent/estimator_facts.py` | 新增，从拟合对象读分类器族 |
| `superagent/prose_claims.py` | 新增，散文断言抽取与判定 |
| `superagent/benchmark.py` | 接入：开局语义检查、候选过滤、试验后事实核对、交换探针、两个汇总块 |
| `superagent/experiment_worker.py` | 记录 `estimator` |
| `superagent/contracts.py` | 未知事实允许提案通过，结构校验照旧 |
| `superagent/targeted_feedback.py` | 未知事实返回 `unverified` 而不是 `None` |
| `superagent/explanation_audit.py` | 接入散文断言与计数 |
| `superagent/predict.py` 未改；`tests/test_general_extensions.py` | 修了该测试读取 `prediction_manifest.json` 未指定编码的问题（中文用户名下会误判） |
| 新增测试 | `tests/test_task_semantics.py`（13）、`tests/test_estimator_facts.py`（10）、`tests/test_prose_claims.py`（12） |

```powershell
python -m unittest tests.test_task_semantics tests.test_estimator_facts tests.test_prose_claims -q
python -m unittest discover -s tests -q
```

## 5 仍然没解决的

1. **对称任务没有可用的成对候选。** 排除增强之后只剩 4 个表征类候选，没有一条是真正的"对称成对模型"。要让这类任务跑得有意义，需要新增不假设顺序的候选（例如 `|a−b|` 特征或对称化概率平均）。
2. **`relation`（多分类成对）没有成对结构。** 它按 `text_tabular` 走，左右两列被当作两个独立特征，现有 5 个候选无法表达"关系"。
3. **散文核验只覆盖三类断言。** "因为 C 变大所以更稳"这类因果措辞、比较级、隐含引用都不在覆盖范围，`fully_verified` 仍为 `False`。
4. **交换探针只在内部搜索行上、只对选中候选跑一次。** 这是低成本诊断，不是每个候选的验收。
5. **`configuration_fact_check` 依赖 worker 回报。** 若 worker 崩溃或在 refit 阶段失败，该试验记为 `unverified`，不推断。
