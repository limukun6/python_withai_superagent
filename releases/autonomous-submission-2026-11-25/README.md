# 可交付的全自主 SuperAgent

运行源码来自 `d501807e61b5df1ac431d8fdf300c4bf4aaa4c2f`。两份课程任务均在空目录自主训练，10 次实际拟合包含最终重训。每一步的选型、复核及停止时机由规划器依据当时的训练反馈决定。课程指定验证在两份预测都封存后评分。[完整源码](https://github.com/limukun6/python_withai_superagent/tree/d501807e61b5df1ac431d8fdf300c4bf4aaa4c2f) · [两份成绩与模型指纹](verification/designated-results.json) · [自主决策验收](verification/autonomy.json)

| 任务 | 指定验证准确率 | 模型 API 调用 |
| --- | ---: | ---: |
| Task1 优惠券 | 1533/2030，75.52% | 0 |
| Task2 标题偏好 | 684/940，72.77% | 0 |

## 从这个目录运行

使用 Python 3.11。要复现两次验收时的候选范围，准备四个公共特征编码器。已有权重时可运行 `python prepare_encoders.py /path/to/weights --check-only`，再把下面的目录改为实际位置。下载与校验均使用固定发布版本和文件指纹。[编码器清单](verification/weights.json)

```sh
python -m pip install -r requirements-ab-encoder-tested.txt
python prepare_encoders.py weights
export SUPERAGENT_MINILM_WEIGHTS="$PWD/weights/all-MiniLM-L6-v2"
export SUPERAGENT_DEEP_MINILM_WEIGHTS="$PWD/weights/all-MiniLM-L12-v2"
export SUPERAGENT_E5_SMALL_WEIGHTS="$PWD/weights/e5-small-v2"
export SUPERAGENT_BGE_SMALL_WEIGHTS="$PWD/weights/bge-small-en-v1.5"
python -m unified_agent solve TASK.zip --prediction-input FEATURES.zip --run-dir runs/new --output output/predictions.csv
```

用本机任务包和只含特征的待预测输入替换命令中的文件名，使用新的运行目录。交付后可在独立进程运行 `python -m unified_agent predict FEATURES.zip --run-dir runs/new --output replay/predictions.csv`。指定验证结果仅适用于上述训练数据、环境和运行记录。

## 团队提交材料

仓库提供可重新训练的源码、依赖、权重获取办法和验收凭据。团队线下工程包 `SuperAgent-autonomous-d501807.zip` 包含两份已训练模型、四个编码器和运行日志，大小 496,424,643 字节，SHA-256 为 `48dbfddee80f15bc056eb1daa9c3f9f306cbba19e3d2236a308ace22c260a2f5`。两份模型从该包中独立重载后的预测与原交付逐字节一致。课程报告、演讲与最终评测材料由团队汇编。[工程包验收凭据](verification/portable.json)
