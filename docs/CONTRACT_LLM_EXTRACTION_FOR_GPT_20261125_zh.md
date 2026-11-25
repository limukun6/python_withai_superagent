# 请审阅：用固定 schema 写任务契约会漏掉关键信息，是否该加一层 LLM 抽取

2026-11-25。本文写给外部审阅者，不依赖仓库访问；所有例子都来自真实运行记录。

## 1 背景（30 秒）

我们做了一个分类任务的 SuperAgent：拿到一个任务包（数据 + 说明），它自己决定试哪些方法、什么时候停、最后交付哪个模型。

系统里有一个核心对象叫**任务契约**（task contract），是一个固定的 JSON 记录：

```json
{
  "contract_version": 1,
  "objective": "classification",
  "kind": "categorical_tabular",
  "metric": {"name": "accuracy", "labels": [0, 1], "direction": "maximize"},
  "input_schema": {},
  "selection_data": "training-derived partitions only",
  "formal_test_available_to_agent": false,
  "goal_resolution": {
    "fields": {
      "primary_metric": {"value": "accuracy", "source": "declared_task_readme", "status": "confirmed"},
      "positive_label": {"value": null, "source": "not_applicable", "status": "not_applicable"}
    }
  }
}
```

契约由**确定性 Python** 生成（读 manifest 字段 + 少量正则），不使用模型。理由：契约是下游一切的前提（候选菜单、评分指标、类别顺序、报告措辞），必须可复现、可拒绝、不花钱。

`goal_resolution` 给每个字段标了**来源**（`declared_task_readme` / `declared_manifest` / `default_accuracy` …）和**状态**（`confirmed` / `unknown` / `not_applicable`），用于区分“任务真的这么要求”和“我们猜的默认值”。

## 2 现象：固定 schema 会漏掉真实存在的关键信息

### 案例 A：21 个列名全部丢失

任务包的说明文档（1,084 字符）明确写了：

> The columns in each `C_*.npy` file, from left to right, are `temperature`, `time`, `coupon`, `expiration`, `gender`, `age`, `maritalStatus`, `has_children`, `education`, `occupation`, `income`, `Bar`, `CoffeeHouse`, `CaryAway`, `RestaurantLessThan20`, `Restaurant20To50`, `toCoupon_GEQ5min`, `toCoupon_GEQ15min`, `toCoupon_GEQ25min`, `direction_same`, `direction_opp`.

但生成出来的契约是：

```json
"input_schema": {}
```

**21 个列名一个都没有进契约。** 模型在决策时看到的是“一个 21 列的类别表格”，不知道哪列是温度、哪列是收入、哪列是方向。README 里明明有，是我们没搬进去。

### 案例 B：标签语义丢失

同一份说明写着：

> predicts whether a driver would accept an offered coupon (`1` = accept, `0` = decline)

契约里是：

```json
"positive_label": {"value": null, "source": "not_applicable", "status": "not_applicable"}
```

当前指标是 accuracy，所以不影响算分；但如果外层要求“报告 accept 类的召回”，我们**不知道哪个类是 accept**，只能退化成“排序最后一个”——纯属巧合。

### 案例 C：指标措辞认不出就静默退回

我们用一个测试包验证过：

| 说明文档原文 | 解析出的指标 | 来源 |
|---|---|---|
| `report macro F1` | **accuracy** | `legacy_adapter_default`（= 我们猜的） |
| `report test accuracy` | accuracy | `declared_task_readme`（= 文档说的） |

也就是说：**文档说 F1，我们按 accuracy 选模**。契约里标了 `unknown`（诚实），但**优化方向是错的**。

另一面：manifest 里的指标名是**大小写与连字符敏感**的——`F1` / `f1` / `macro-f1` / `AUC` / `Accuracy` 全部被拒绝，只接受 `macro_f1` / `roc_auc` / `accuracy`。

### 案例 D：任务级方法限制根本没有字段

如果任务说“不得使用任何预训练模型”或“只允许线性模型”，契约里**没有地方放**，于是：

- 菜单照旧生成（默认菜单恰好都是自训练方法，但这是运气）；
- 报告里也没有任何地方能发现“我们可能违反了限制”。

## 3 为什么“只用提示词工程”不够

我们已经同时用了三层：

1. 提示词里写死字段清单；
2. API 层 `response_format: {"type": "json_object"}`；
3. Python 严格校验字段集合，不符就拒绝。

实测结果：**格式不是 100% 稳定**，而且更关键的是——

| 失败类型 | 提示词能不能防 |
|---|---|
| 少字段、类型错 | 大体能（+校验） |
| 值是对的但**内容错**（比如把 21 列名字写成 20 个） | **不能** |
| 不知道时**编一个值** | **反而不利**（强制格式会逼出幻觉） |

结论：**提示词保证“形状”，保证不了“值”。**

## 4 提议：加一层受限的 LLM 抽取，Python 仍握最终决定权

```
① Python 确定性解析（现有）
      → 认出来的字段：status = confirmed
      → 认不出来的：留空
② LLM 结构化抽取（新增，只补缺口）
      → 只能填下面 5 个格子，输出过 schema，允许 null
③ Python 校验 + 冻结
      → 数据对得上才接受：source = llm_extracted, status = proposed
      → 按说明文档的 SHA-256 缓存落盘，之后每次运行读缓存（0 调用、逐字节一致）
```

### 只允许抽取这 5 项（封闭词表）

| 字段 | 允许的值 | Python 校验规则 |
|---|---|---|
| `columns` | 列名列表 | 长度必须等于 `feature_count`；名字不重复；名字不出现在目标列里 |
| `positive_label` | 标签域内某个值 | 必须属于 `np.unique(y_train)`；且指标必须是二分类指标 |
| `metric` | 白名单词（可先做同义词归一） | 必须在 `SUPPORTED = {accuracy, macro_f1, weighted_f1, micro_f1, binary_f1, balanced_accuracy, roc_auc}` |
| `restrictions` | 若干布尔旗标（`no_pretrained_assets`、`no_external_data`） | 必须是已知旗标；不接受自由文本 |
| `split_roles` | 哪个文件是 train / val / test | 文件必须真实存在于包内；标签是否可用要一致 |

### 三条硬规则

1. **允许 null**：模型不确定时必须返回 null，而不是编一个值；null 记为 `unknown`，报告写“未声明”。
2. **不可升级为 confirmed**：LLM 抽取值最高只能是 `proposed`；除非与确定性解析一致，或有人确认。
3. **不允许出现“用哪个模型”**：抽取结果禁止包含方法建议，方法仍由菜单与调度决定（避免按任务名调用专用方案）。

### 成本与可复现

- 一个任务**只调用一次**（说明文档很短，约 1,000–2,000 字符）；
- 结果按 `sha256(说明文档)` 缓存到磁盘并进版本库；
- 同一份文档 → 同一份契约，逐字节一致。

## 5 这套东西能补哪些洞

| 洞 | 现在 | 加抽取层后 |
|---|---|---|
| 21 个列名 | 全丢 | 抽出来，长度对上 21 才接受 |
| `1 = accept` | `not_applicable` | 抽出来，在标签域内才接受 |
| `report macro F1` | 静默 accuracy | 归一成 `macro_f1`，进白名单才接受 |
| 禁用预训练/只准线性 | 没有字段 | 抽成旗标，**过滤菜单** |
| 未知封装格式 | 直接拒绝、零产出 | 先尝试读懂格式说明，可能救回来 |

最后一条对我们尤其重要：课程给的未见任务在**运行期**才公布格式，而提交的版本必须冻结、执行期禁止人工干预。有了运行期抽取层，“未知格式 → 拒绝”才可能变成“未知格式 → 先读懂再跑”。

## 6 风险与对策

| 风险 | 对策 |
|---|---|
| 提示注入（说明文档里写“忽略指令”） | 文档只作为数据；指令固定；字段封闭；结果还要过数据校验 |
| 幻觉（编出列名/指标） | 长度、域、白名单三重校验；不确定必须 null |
| 不可复现 | 按文档 SHA 缓存冻结 |
| 过度信任 | `status: proposed`，报告必须标“模型抽取，非任务原文” |
| 成本失控 | 每任务一次、有缓存、走唯一账本 |
| 越权选方法 | 抽取结果禁止包含方法建议 |

## 7 想请你审的问题

1. 这个分层（确定性解析 → LLM 抽取候选 → Python 校验 → 缓存冻结）有没有更成熟的替代做法？业界一般怎么处理“文档里的关键声明”？
2. `status: proposed` 这个中间态会不会太弱——报告里怎么表述才既诚实又不至于让人误读成“已确认”？
3. 抽取层允许 null 会不会导致大量 unknown，反而不如强制它给值？两种取舍你更倾向哪个？
4. 用文档 SHA 缓存来保证可复现，有没有更好的做法（例如把抽取结果当作任务包的一部分，由提供方签署）？
5. 提示注入：只靠“文档当数据 + 封闭字段 + 数据校验”够不够？还需要什么？
6. 如果只能用一次调用，你会把 token 预算花在“通读全文”还是“只抽结构化字段”上？
7. 有没有一种设计可以**既不需要 LLM**、又能覆盖这四类信息？（如果我们漏掉了某条确定性路径，请指出。）

## 8 附：真实材料的规模

| 项 | 值 |
|---|---|
| 说明文档长度 | 492 / 1,084 字符（两个已发布任务） |
| 契约大小 | 约 1 KB JSON |
| 现有确定性语法 | 只认两种句式：`report [test] accuracy`、成对文本的偏好定义句 |
| 模型 | DeepSeek-V4.1-Flash（课程限定 Qwen/DeepSeek） |
| 单次调用成本量级 | 约 ¥0.001（1,000 字符输入 + 小段输出） |
