# 第五轮：重复画像树的容量与类别统计

第四轮双塔两折平均 −2.5563pp，融合 −0.5851pp，均不晋级。第三轮统一 Agent 选中 catboost_recurring_context，指定验证 1549/2030、76.3054%，10fit、171.016秒、0API，模型跨进程重载一致。80% 还差 75 题。

## 预登记配置

在相同无标签结构发现与类别组合上，比较当前默认树和两个固定配置。

| 名称 | 深度 | 轮次 | 学习率 | L2 | boosting | 类别组合复杂度 |
|---|---:|---:|---:|---:|---|---:|
| incumbent | 6 | 600 | .05 | 3 | Plain | 默认 |
| deeper | 8 | 2400 | .035 | 8 | Plain | 2 |
| ordered | 6 | 2400 | .035 | 8 | Ordered | 2 |

每次模型只在拟合侧发现画像与字段交叉。训练内 18643 五折中的前两个折，合计 6 个分类器拟合。该数据已在前四轮开发过，结论是开发证据。每条路径只含一拟合，不带额外分类器或外部数据。预计 CPU 3–12 分钟。API 0，课程费用 0。

挑战者须两折都更好，平均提升至少 .5pp。都通过时取平均增益较高者。未过线保留现有 76.31% 起点。过线后另做一次整组新画像压力测试和统一入口验收，生产候选按同一反馈规则选择。指定验证在本批封存，后续评分仍先冻结模型和预测。正式 test 留到最终版本评价。

```powershell
D:/codex/python_withai/runs/a79_encoder_assets/venv/Scripts/python.exe -m scripts.run_task1_context_capacity --task D:/codex/python_withai/data/released_task1.zip --output research/task1_context_capacity_01
```
