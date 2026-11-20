# 课程指定验证集 · 2026 年 11 月 20 日

| 任务与方案 | 拟合时使用的开发行数 | 验证正确数 | 准确率 | 本地特征输入 token |
|---|---:|---:|---:|---:|
| 优惠券 · CatBoost 深度 6 | 6,494 | 1,511 / 2,030 | 74.43% | 0 |
| 标题 · 轻量默认 MiniLM-L6 | 6,608 | 682 / 940 | 72.55% | 37,448 |
| 标题 · 可选双编码器 MiniLM-L12 与 L6 | 6,608 | 682 / 940 | 72.55% | 74,896 |

两个标题方案共同答对 643 对，共同答错 219 对。剩下 78 对中，轻量方案和双编码器各答对 39 对。训练内五折中观察到的双编码器准确率优势，在这 940 对标题上没有出现。轻量路线继续作为默认，双编码器保留为可选项。[准确率和逐行对照计数](scores.json)

对照此前仓库的四次搜索实验，优惠券从 1,527 降到 1,511，标题从 654 升到 682。该次训练使用完整的 8,117 和 8,301 行，这次使用开发阶段已冻结的模型，另外保留了课程训练集中的 1,623 和 1,693 行。两版的训练数据量和搜索配置都有变化，这组差值无法单独衡量某一种模型的贡献。课程验证集已有历史评估记录。[四次搜索实验](../../docs/search_budget_replication_results_zh.md) · [本次模型的训练来源](../../research/development_pool_models.json)

## 数据和费用

两个任务均使用相同的准确率定义，每行预测与发布标签相同计一分。输入来自原始 ZIP 中的 `C_val.npy` 和 `val.csv`。标题输入只投影 `headline_A` 和 `headline_B` 两列。三个模型都通过项目的标准预测入口导出，逐行预测的校验值先写入记录，独立评分阶段再使用 `y_val.npy` 和 `label_pairwise`。预测文件只含行号与预测类别。正式测试成员没有被打开。[验证前冻结的模型与文件指纹](protocol.json) · [预测冻结记录及运行环境](predictions_frozen.json)

三个模型的付费 API 请求、API 输入和输出 token、API 金额均为零。表中的 token 是本次冷缓存预测在 CPU 上用本地 MiniLM 提取特征的实际用量。这次使用已有的固定版本权重。[优惠券预测](task1/validation_predictions.csv) · [标题轻量模型预测](task2/validation_predictions.csv) · [标题双编码器预测](task2-wide/validation_predictions.csv)

## 复跑与核对

从仓库根目录运行。实测 Python 3.11.15，依赖见 `requirements-pretrained.txt`。发布前在此环境通过 329 项自动测试，并逐行独立复算了三份预测。准备课程原始 ZIP 和训练时固定版本的两个权重目录，替换命令里的示例路径。首次预测使用一个新的输出目录。

```bash
python -m tools.mainline_validation predict \
  --packages /path/to/course-zips --out runs/designated-validation \
  --minilm-weights /path/to/all-MiniLM-L6-v2 \
  --deep-minilm-weights /path/to/all-MiniLM-L12-v2
python -m tools.mainline_validation score \
  --packages /path/to/course-zips --out runs/designated-validation
```

仅核对已发布预测时，可跳过模型推理，使用同一环境和下面的命令。在原样检出本版本时，输出的逐行计数和 `scores.json` 一致。

```bash
python -m tools.mainline_validation score \
  --packages /path/to/course-zips --out validation/mainline-2026-11-20
```
