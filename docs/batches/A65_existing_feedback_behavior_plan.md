# A65：补A2行为/质量并列分析，复用已封存回答

输入为A55六状态、12个真实已封存回答与两新划分动作实测；重新核验state SHA与R/E共同字段，按state_id连接。此批0训练/0新API，不访问H/test/外层标签。

每条回答核对候选是否在当时可选名单、原params与注册classifier是否相符、evidence_trials是否引用已完成观测；记录预期为null还是数字。每状态统计原history中失败试验、configuration_fact_check=contradicted、hypothesis_check=not_supported的机会数及回答是否显式引用这些trial。引用trial仅是可追踪行为，不代表正确理解/成功纠错；自然失败机会0时纠错率记不可测，不能写100%或零错误。

按任务汇总候选变化、两划分平均E−R，保持与A58完整流程结果分开。12回答不是12独立任务，检查通过不证明所有自由文本事实正确；不调用模型重新评价自己。目的填补D3可机械核验的行为表，语义纠错的人工标签与四任务覆盖仍缺。

```powershell
& 'D:\codex\python_withai\.venv\Scripts\python.exe' -m scripts.run_a65_existing_feedback_behavior --answers 'D:\codex\python_withai\runs\a55_fresh_answers' --evaluation 'D:\codex\python_withai\runs\a55_fresh_evaluation' --output 'D:\codex\python_withai\runs\a65_feedback_behavior'
```

预计数秒。新输出目录不能存在，输入不改，全部只读分析。分类规则事后登记，结果是探索性行为审计，不称A55预登记主要终点。
