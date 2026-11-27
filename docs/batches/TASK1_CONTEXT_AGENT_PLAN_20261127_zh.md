# 第三轮：重复画像候选的真实 Agent 验收

输入版本 12e9eb4。前两轮共 17 次拟合，第二轮 T 对同折基线平均 +1.4805 个百分点，3/3 同向，仅 T 晋级。保持树与表示配置。统一入口在主训练折内无标签发现重复结构，存在足够支持并有变化情境时开放这个候选。所有候选继续由原调度器选型，不指定赢家。CPU、学费 API 0。

从新目录调用 solve 读取原包训练数据和说明，另供仅包含 C_val 的预测包。阻止训练子进程打开原包 val/test。预算 10 次拟合含重训。检查真实 fit 记录、完整选模、新进程重载以及逐行顺序。如 Agent 交付非 T，可额外用 1 次固定 T 全训练拟合作为方法诊断；两份预测须在一次读取 y_val 之前全部封存。正式 test 不开。仅在选型和重载均通过后评分。记录历史 1533/2030 和目标 1624/2030 的差。

```powershell
D:/codex/python_withai/runs/a79_encoder_assets/venv/Scripts/python.exe -m scripts.run_task1_context_acceptance --task D:/codex/python_withai/data/released_task1.zip --output research/task1_context_agent_01 --private runs/task1_context_agent_01
```
