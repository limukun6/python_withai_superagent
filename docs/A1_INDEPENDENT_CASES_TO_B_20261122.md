# A1独立48例接入说明（给B侧Codex）

A已实现统一案例runner，并在8类×6=48个A侧开发例发现并修复4个跨范围错误确认。原96案例中的独立48例尚未补齐，不能把A例换数字当独立封存。B可继续自己的方法实验，同时按此接口提供案例，无需重写检查器。

每类6例，共48例：task_semantics、configuration、trial_identity、metric_partition、negation_future、slice_scope、memory_state、failure_comparison。每类包括合法、错误、信息不足。JSON数组各项有：id、category、operation、subset=B_independent_reserved、author=B、expected=confirmed|contradicted|unknown、expectation_basis（独立判定理由）、payload。

操作与payload格式参考本仓库`research/a67_cases/A_development.json`和`scripts/run_a67_case_registry.py`，但应独立写不同表述、取值与场景，覆盖同一接口契约；不要看被测输出再反推expected。A的模型配置与原数据已供检查，例子须数据-only，不嵌可执行代码。未知证据不能标签为错误事实，不把“supported numerical expectation”当因果已证明。

先提交JSON与SHA、编写/标注来源、冻结前是否运行过检查器。A在冻结源上统一执行，保存逐类错误确认、合法未确认、未知误确认与原始状态。运行后修规则时该批成为回归，不能继续称独立未见验证。若B也已看过A修复输出/代码，应如实记录可见信息，独立贡献不等于严格盲测。

运行命令：
```powershell
python -m scripts.run_a67_case_registry run --cases B_CASES.json --output NEW_OUTPUT
```

本次不要求B操作者把真实标签、数据集或密钥写入案例。A67是受控结构测试，不测自然错误率；A2自然反馈/错误恢复和B1完整运行分别报告。

执行前先验证48例/八类各六、三种判定都有、ID与输入不重复、作者/子集和预期依据完整：`python -m tools.validate_a1_case_pack --cases B_CASES.json --role B --output NEW_VALIDATION.json`。B例ID用B_前缀，避免合并后与A重复。格式检查不验证真实作者身份或盲测过程，需另外附来源记录；检查器不会自动生成expected。
