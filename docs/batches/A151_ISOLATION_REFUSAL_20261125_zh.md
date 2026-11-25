# 第四批补充二：声明式实体/时间隔离在入口上确实拒绝

2026-11-25。补齐第四批报告第 8 节第 3 项。**0 API、0 训练**（拒绝发生在训练之前）。

## 1 结论

包在 `evaluation_protocol` 里声明 **temporal** 或 **entity** 隔离时，入口**明确拒绝**，不会退化成随机分组：

```
status: failed
error:  Declared temporal/entity evaluation protocol needs a supported explicit splitter
attempted_classifier_fits: 0
partial_trial_models: []
delivery_status: unavailable
```

**0 次 fit、0 个模型、0 次 API。** 这是"时间不足时先准确拒绝不支持的协议"这条要求的可复现证据。

## 2 测试（5 项，`tests/test_declared_isolation.py`）

| 用例 | 断言 |
|---|---|
| temporal 协议 | `grouped_partition` 抛错且信息含 `temporal/entity evaluation protocol` |
| entity 协议 | 同上 |
| 协议不是对象（`"temporal"` 字符串） | 抛错 |
| **端到端**：带 temporal 协议的包跑入口 | 退出码非 0、标准输出含该错误、**没有 `trial_1.joblib`**、`run_outcome.status=failed`、`attempted_classifier_fits=0` |
| 未声明协议（`{}`） | 正常划分，训练集与评价集**类域都完整** |

## 3 一个口径细节（值得记住）

**运行期拒绝写进 stdout 的结果 JSON，不写 stderr。** 我第一版测试断言 `stderr` 含该错误，失败了——因为只有 argparse 级别的参数错误才走 stderr。现在测试同时检查 stdout 与 `run_outcome.json`。

这个区别对调试有影响：看到退出码 1 且 stderr 为空时，应该去看 stdout 的结果 JSON。

## 4 仍未做

1. **真实任务迁移**：需另立数据资格/划分/预算计划；
2. **受支持的显式 splitter 本身**：目前只拒绝，未实现按时间列或实体列切分；要做得先定"时间列如何声明、实体列如何声明、切分后类域不完整怎么办"。

## 5 资源

0 API、0 fit、¥0。账本未写：仍 457 请求 / known ¥7.925792 / unknown 16 / pending 0。

测试：新增 5 项，宽口径 **224 项通过**（25 个测试文件）。
