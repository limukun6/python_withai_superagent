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

## 4 预训练表格模型：拿到了，但明显更差

`pip install tabpfn` 成功（10.2.0）。默认 v3.5 权重在 HuggingFace 上**受门控**：

```
TabPFNHuggingFaceGatedRepoError: HuggingFace authentication error
downloading from 'Prior-Labs/tabpfn_3_5'.
```

正确地址是 **`https://huggingface.co/Prior-Labs/tabpfn_3_5`**（此前文档误写成 `tabpf_3_5`，所以打开是 404）。

绕开门控的办法是改用 **v2 检查点**（`Prior-Labs/TabPFN-v2-clf`，29 MB）。本机直连 HuggingFace 会被重置，改用镜像 `hf-mirror.com` 下载成功，SHA-256 `cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49`。

CPU 上默认拒绝超过 1,000 行，需设 `TABPFN_ALLOW_CPU_LARGE_DATASET=1`。实测（split 707，训练 6,495 行）：

| 编码方式 | 准确率 | 相对 depth8 | 耗时 |
|---|---:|---:|---:|
| 序数编码（21 维） | 0.712084 | **−4.13pp** | 40 秒 |
| one-hot（111 维） | 0.704069 | **−4.93pp** | 127 秒 |
| `catboost_depth8_reg3`（基线） | 0.753391 | — | 约 100 秒 |

**两种编码都明显更差**，one-hot 还更慢。可能原因：

1. Task1 的 21 列**全是类别**，TabPFN v2 主要在小规模数值/合成数据上预训练，没有原生类别处理；
2. 8,117 行卡在它舒适区边缘（v2 建议 ≤10,000 行）。

**结论：预训练表格模型在这份数据上不占优，差 4 个百分点以上。**

### 若仍要试更强的 v3.5 权重

1. 登录 HuggingFace 访问 `https://huggingface.co/Prior-Labs/tabpfn_3_5` 接受条款并配置 `HF_TOKEN`；
2. 确认报告里可以写明"使用了预训练表格权重"及其许可（该系列为非商业许可，研究用途可用）；
3. 本机 RTX 5070（sm_120）与已装的 torch cu126 内核不匹配，**只能 CPU 推理**。

## 5 把这一轮和前几轮合起来看

| 测试方向 | 最好结果 | 相对 depth8 |
|---|---|---:|
| 菜单内（14 个） | `depth8_reg3` | 基线 |
| 菜单外经典模型（5 个） | LightGBM | −0.19pp |
| 菜单内被排除的候选（2 个） | adaptive CatBoost | +0.06pp（贵 2.8×） |
| 离线等权集成 | — | −0.37pp |
| 预训练表格模型（TabPFN v2，两种编码） | 0.7121 / 0.7041 | **−4.13 / −4.93pp** |

**所有跑出数字的方案，没有一个超过 depth8 超过 0.2pp。**

## 6 结论

1. **菜单外没有找到更好的方案**：经典模型全输，集成全输，被排除的候选只打平且更贵，预训练表格模型差 4pp 以上。
2. **预训练路线技术上可跑**（镜像拿到 v2 权重），但在这个"全类别、中等规模"的任务上不占优。
3. 结合前几轮的 36 次 fit：Task1 在 75% 附近**已经饱和**，0.2pp 级别的差异远低于我们实测的 7.9pp 重跑漂移。

## 7 副作用记录

- 为测试安装了 `lightgbm 4.7.0` 与 `tabpfn 10.2.0`，并在 `%APPDATA%\tabpfn` 下载了 29 MB 的 v2 检查点；
- **未加入交付依赖，未改变任何交付产物**；
- 交付清单记录的环境仍是 Python 3.12.14 / catboost 1.2.10 / scikit-learn 1.7.2。

证据：`research/task1_extra_candidates/{offmenu_models.json,ensemble.json,tabpfn_v2.json,tabpfn_v2_onehot.json}`。

