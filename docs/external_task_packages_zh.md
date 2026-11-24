# 两个新任务包：数值表格与普通单文本（CC BY 4.0）

2026-11-20。十天计划第一步要求补 `numeric_tabular` 与 `single_text` 两类的真实任务；此前只有代码层的事实覆盖（49 个候选全部有分类器说明），没有跑过。本文记录两个新任务包的来源、许可、构建方式与统一入口实测。

## 1 来源与许可

两个数据集都来自 UCI Machine Learning Repository，许可均为 **CC BY 4.0**，要求署名。

| 项 | 数值表格 | 普通单文本 |
|---|---|---|
| 数据集 | Dry Bean Dataset | SMS Spam Collection v.1 |
| UCI id | 602 | 228 |
| 页面 | <https://archive.ics.uci.edu/dataset/602/dry+bean+dataset> | <https://archive.ics.uci.edu/dataset/228/sms+spam+collection> |
| 下载 | `.../static/public/602/dry+bean+dataset.zip` | `.../static/public/228/sms+spam+collection.zip` |
| 许可 | CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/> | 同左 |
| 原始包 sha256 | `0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f` | `1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eeaf9963ce3` |

署名文本（已写入每个包的 `manifest.json` 的 `source.citation`）：

- Koklu, M. and Ozkan, I.A. (2020), *Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques*, Computers and Electronics in Agriculture 174, 105507.
- Almeida, T.A., Gomez Hidalgo, J.M., Yamakami, A. (2011), *Contributions to the Study of SMS Spam Filtering: New Collection and Results*, Proceedings of the 11th ACM Symposium on Document Engineering.

原始下载保留在 `data/_raw/`（被 `.gitignore` 的 `data/` 规则忽略），未做任何修改。

## 2 构建

入口：`scripts/build_external_tasks.py`。只做格式转换与划分，不改变数据内容。

| 步骤 | Dry Bean | SMS Spam |
|---|---|---|
| 原始格式 | ARFF（另有 XLSX，不使用） | 制表符分隔，无表头 |
| 转换 | 16 个数值属性原样搬成 CSV 列，`Class` 作为目标 | `label`/`message` 两列，文本除首尾空白外不动 |
| 去重 | 无 | 删除 414 条完全重复的文本（5,574 → 5,160），避免同一文本同时出现在两个划分 |
| 缺失值 | 0 | 不适用 |
| 划分 | 80/20 分层，`random_state=20261120` | 同左（去重后） |
| 指标 | `accuracy`（7 类，多数类基线 26.1%） | `macro_f1`（正类 12.4%，accuracy 会被多数类主导） |

类别分布：Dry Bean 为 DERMASON 3,546 / SIRA 2,636 / SEKER 2,027 / HOROZ 1,928 / CALI 1,630 / BARBUNYA 1,322 / BOMBAY 522；SMS 为 ham 4,518 / spam 642。

zip 内部条目使用固定时间戳（1980-01-01），因此**重建字节一致**：连续两次构建得到的包 sha256 相同。

## 3 产物

| 包 | 任务类型 | 主指标 | 训练 / 验证行数 | 列数 | 包 sha256 |
|---|---|---|---:|---:|---|
| `data/dry_bean_numeric_tabular.zip` | `numeric_tabular` | accuracy | 10,888 / 2,723 | 16 | `f0ad82b6789cb139c4811901d2fb4d1e4b81ff1083e53b128928c1bd6d6cf24c` |
| `data/sms_spam_single_text.zip` | `single_text` | macro_f1 | 4,128 / 1,032 | 1 | `da3595c66d055abe030f630cda88dd733aa7534eb1658f3e3e27888abcb2dfe3` |

每个包内是 `manifest.json` + `train.csv` + `validation.csv`，manifest 遵循仓库已有的 schema_version 1，没有新增字段（来源信息放在 `source`，读取器会忽略未知键）。

## 4 统一入口实测（零 API）

用发布配置的本地策略，从分支工作区跑统一 CLI：

```powershell
python -m superagent run D:\codex\python_withai\data\dry_bean_numeric_tabular.zip `
  --run-dir D:\codex\python_withai\runs\external\dry_bean_smoke `
  --planner local --policy interleaved_routes --trials 2 --seed 707 --trial-timeout 180 --time-limit 900

python -m superagent run D:\codex\python_withai\data\sms_spam_single_text.zip `
  --run-dir D:\codex\python_withai\runs\external\sms_spam_smoke `
  --planner local --policy interleaved_routes --trials 2 --seed 707 --trial-timeout 180 --time-limit 900
```

| 任务 | 识别类型 | 主指标 | 选中候选 | 验证成绩 | 拟合 | API | 回退 |
|---|---|---|---|---|---:|---:|---:|---:|
| Dry Bean | `numeric_tabular` | accuracy | `numeric_tabular_logistic_C0.1` | 0.9225 | 7 | 0 | 0 |
| SMS Spam | `single_text` | macro_f1 | `single_char_C3.0` | 0.9395（accuracy 0.9758） | 7 | 0 | 0 |

两条都 `full_refit_completed=true`、`formal_test_accessed=false`、`complete_course_cost_rmb=0`。这说明新类型能走通同一个入口：加载、候选库、搜索、三划分复核、全量重训、验证评分。

**这是管线冒烟，不是性能结论。** 只有 2 次搜索、单一划分、单次运行，不能据此说这两个任务上 Agent 好或不好。

## 5 需要披露的一件事

这次冒烟让两个任务的验证分数已经可见。十天计划要求"至少两个任务在规则冻结前只检查数据格式、不查看性能结果"，因此：

- 这两个包从现在起算**开发任务**，可以用于调试规则和估计耗时；
- 如果要保留一对"冻结前没看过成绩"的任务，需要另外引两个数据集，并在规则冻结后才跑；
- 报告里引用这两个任务的数字时，必须说明它们属于开发阶段。

## 6 复现与边界

```powershell
# 重建两个包（幂等，输出前会校验原始包 sha256）
python runs\integration\scripts\build_external_tasks.py --raw-dir data\_raw --output-dir data --verify
```

`--verify` 用分支执行器自己的 `load_development_task` 和 `TaskSpec.from_task` 加载两个包并打印类型、指标、行数、类别数。

边界：

1. 两个任务都没有课程意义上的正式 test 划分，任何数字都只到我们自己划的验证集，且该验证集已被看过一次。
2. Dry Bean 的 16 个特征是图像形态学衍生量（面积、周长、形状因子等），它是**数值表格**任务，不是图像任务；不要用它声称图像模态已覆盖。
3. SMS 去重后仍是同一语料的一次重新分组，不能当成多个独立任务或多个样本。
4. 生成的 CSV 保留了源数据的行内容，许可为 CC BY 4.0，再分发时须保留第 1 节的署名。
