# A2-02：为history-evidence-v3重新冻结真实状态

运行前协议，源码提交1d57fd4。沿用已审查scripts/run_a2_state_study.py prepare，不变更算法/选择政策，不调用ask。
四任务task1/task2/DryBean/SMS，每任务固定前三次搜索，prefix1/2/3共12状态；不按错误发生或得分筛选状态。训练内reserve41101/outer45101/search707，error_feedback和feature_cache同旧协议；外层不进入planner，不访问正式validation/test。
预算12真实fit/API0，加原流程35fit共47，仍在三小时实验49fit上限内；回归38fit另记，预期窗口总85fit。每任务time_limit1200/trial90为原协议上限，实际旧prepare约50秒。失败停止保留，不自动重试。
任务仍按原scope累计历史，11条未知用量未对账；不创建空账本、不触发真实ask。
新状态两组共享字段逐项一致，checked只额外checks/memory。冻结完整源码及上下文指纹。源码后续修改即失去直接ask身份，必须另记版本。结果仅状态准备，不是反馈效果实验证据。

执行（在integration工作树）：

```powershell
D:\codex\python_withai\.venv\Scripts\python.exe scripts/run_a2_state_study.py prepare --output D:\codex\python_withai\runs\a2_02_v3_states
```

完成后保留本地完整run、顶层状态与脱敏汇总；不提交模型/训练输入joblib。真实调用仍需按既定账本preflight，不在本批执行。
