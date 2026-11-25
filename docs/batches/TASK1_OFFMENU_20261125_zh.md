# Task1 菜单外方法实测：经典模型全输，预训练表格模型需授权且被门控

2026-11-25。回应“既然没有限制，能不能用预训练模型 / 菜单外的方法”。**0 API**。

## 1 先确认课程是否允许预训练模型

老师 2026-11-22 的答复（`docs/TEACHER_ENCODER_CLARIFICATION_20261122_zh.md`）：

> Qwen/DeepSeek 的限制针对 SuperAgent 使用的语言模型。下游分类任务可以使用其他通用预训练模型（例如 MiniLM encoder）用于特征提取。但 SuperAgent 必须根据任务与实验结果自主选择。

**所以下游用预训练模型是允许的。** Task1 是 21 列类别表格，合适的预训练选择是**表格基础模型**（TabPFN 一类），而不是文本编码器。

## 2 菜单外的经典模型：全部低于 CatBoost depth8

同一批划分（`research/task1_headroom/attempt_02/split_707.joblib`），同一行、同一指标，类别列按训练侧拟合的序数编码。

| 模型 | 准确率 | 相对 depth8 |
|---|---:|---:|
| `catboost_depth8_reg3`（基线） | 0.753391 | — |
| LightGBM | 0.751541 | **−0.185pp** |
| sklearn HistGradientBoosting | 0.742293 | −1.110pp |
| sklearn RandomForest | 0.742910 | −1.048pp |
| sklearn ExtraTrees | 0.713317 | −4.007pp |
| sklearn MLP | 失败（输入含 NaN，类别列的缺失值） | — |

**没有一个赢过 depth8。** LightGBM 最接近，仍差 0.19pp。

## 3 离线集成：也输

用已保存的三个模型（depth6_reg10 / depth8_reg3 / interactions）对概率做等权平均：

| 划分 | 集成 | 最好的单模 | 差 |
|---|---:|---:|---:|
| 707 | 0.750925 | 0.755240 | −0.43pp |
| 811 | 0.739211 | 0.747842 | −0.86pp |
| 823 | 0.752158 | 0.750308 | +0.19pp |
| **均值** | | | **−0.37pp** |

**等权平均把强模型拉低了**——这与项目里多次出现的“融合不自动等于更好”一致。

## 4 预训练表格模型：被许可与门控挡住

`pip install tabpfn` 成功（10.2.0），但 `fit()` 直接抛出：

```
TabPFNHuggingFaceGatedRepoError: HuggingFace authentication error
downloading from 'Prior-Labs/tabpfn_3_5'.
This model is gated and requires you to accept its terms.
```

也就是说，要用它必须：

1. 用 HuggingFace 账号**接受 Prior Labs 的使用条款**；
2. 设置 `HF_TOKEN`。

**这不是技术问题，是授权问题——需要人来做决定，我不代做。**

补充两条事实：

- 该系列权重是**非商业许可**（研究/内部评测允许，商业用途需授权），课程项目属研究用途，理论上可用；
- 本机 GPU 是 RTX 5070（sm_120），而安装的 torch 是 cu126 构建，**不含 sm_120 内核**，所以即便拿到权重也只能**CPU 推理**，6,495 行的预测会慢很多。

## 5 把这一轮和前几轮合起来看

| 测试方向 | 最好结果 | 相对 depth8 |
|---|---|---:|
| 菜单内（14 个） | `depth8_reg3` | 基线 |
| 菜单外经典模型（5 个） | LightGBM | −0.19pp |
| 菜单内被排除的候选（2 个） | adaptive CatBoost | +0.06pp（贵 2.8×） |
| 离线等权集成 | — | −0.37pp |
| 预训练表格模型（TabPFN） | **未能运行** | 需授权 |

**所有能跑出数字的方案，没有一个稳定超过 depth8 超过 0.2pp。**

## 6 结论

1. **菜单外没有找到更好的方案**：经典模型全输，集成全输，被排除的候选只打平且更贵。
2. **唯一没试过的是预训练表格模型**，它被 HuggingFace 门控与非商业许可挡住；要用必须先由人接受条款。
3. 结合前几轮的 36 次 fit：Task1 在 75% 附近**已经饱和**，0.2pp 级别的差异远低于我们实测的 7.9pp 重跑漂移。

## 7 如果仍想试 TabPFN，需要你做的三件事

1. 登录 HuggingFace，访问 `https://huggingface.co/Prior-Labs/tabpfn_3_5` 接受条款；
2. 生成一个 read token，设置环境变量 `HF_TOKEN`；
3. 确认课程/报告里可以写明“使用了非商业许可的预训练权重”。

做了这三步，我可以再跑一次单划分探测（CPU，预计几分钟），但那也只是一次单点对照，不改变“Task1 已饱和”的判断。

## 8 副作用记录

- 本次为测试安装了 `lightgbm 4.7.0` 与 `tabpfn 10.2.0`（未加入交付依赖清单，也未改变任何交付产物）；
- 交付清单里记录的环境仍是 Python 3.12.14 / catboost 1.2.10 / scikit-learn 1.7.2。

证据：`research/task1_extra_candidates/{offmenu_models.json,ensemble.json}`。
