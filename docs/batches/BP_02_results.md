# BP-02 · 两套环境使用同一批训练样本

标题任务和 Dry Bean 合计 6 份入口文件，在两套本地环境中从各自的数据源独立生成后，逐个文件的 SHA-256 全部相等。把它们交给 A 的分区写入函数后，两边的 6 对工作文件都包含相同顺序的训练行、评估行和标签。重排训练输入或篡改清单时，两个任务均在训练前拒绝执行。

| 任务 | 用于优化的训练行 | 留在优化进程外的行 | 输入文件逐字节一致 | 内部训练与评估内容一致 |
|---|---:|---:|---:|---:|
| Released Task2 标题偏好 | 6,629 | 1,672 | 3/3 | 3/3 |
| UCI Dry Bean | 8,714 | 2,174 | 3/3 | 3/3 |

三份交接文件是优化进程读取的 `optimizer/train.zip`、`optimizer/inner_splits.json`，以及交付时单独使用的无标签 `delivery/features.zip`。两次执行分别使用 Python 3.11.15 和 3.14.7，scikit-learn 1.7.2 和 1.9.0，joblib 1.5.2 和 1.5.3，均在 macOS 运行。12 个本机工作文件逐一通过完整性检查。跨环境直接比较相应的 joblib 文件，6 对的二进制都不同。A 可用交接文件摘要核对两边的输入，使用各自实际工作文件的摘要验证本地执行。

## A 端复验

在固定源码版本下，执行下列命令可从标题原始发布包及 Dry Bean 训练包重新生成输入。每次指定新的输出目录。登记行号与参考结果在仓库的 `research/bc1_preparation` 下，两套环境都读取同一份登记。

```sh
python -m scripts.materialize_b_c1_rows --headlines <标题原始发布包> --bean <Dry-Bean训练包> --output <全新目录>
python -m scripts.compare_b_portable_inputs <第一套完整输出目录> <第二套完整输出目录> --check-files
```

输出清单的预登记摘要见下表。A 可将相应摘要与清单路径传给 `tools.ab_registered_rows.registered_partitioner(task, manifest_file, expected_sha256)`，获得一个与现有分区函数签名一致的行号选择器。正式入口仍由 A 维护。本批在实验进程内接入选择器并实际调用 A 的分区写入函数。Windows 正式入口上的复验待 A 执行。

| 任务 | inner_splits.json SHA-256 |
|---|---|
| 标题 | `110ed7d8c03c904c0d58ff00c932189efedc5cc5db79ffc1fd06452a3e8de5fd` |
| Dry Bean | `2548bb80bfa86314536f9f4be182699326d77994314ba5b46d28df1709e20548` |

第二套环境首次在导入 A 的运行模块时缺少 requests，尚未生成数据。复验使用该进程私有的一份已安装 HTTP 依赖副本。两次成功运行均为零训练、零预测、零模型 API 请求。仅使用课程发布包中的训练成员，保留的课程验证和测试集未参与。

[预登记](BP_02_plan.md) · [两环境实物核对](../../research/bp_02_materialization/comparison.json) · [首次导入失败记录](../../research/bp_02_materialization/first_import_failure.json)
