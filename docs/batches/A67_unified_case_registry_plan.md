# A67：统一A1案例格式与A侧48例开发验收

用户要求持续补D3。本批0训练/0API，A累计1007不变。先prepare落盘48个开发案例、逐例预期及依据，再run冻结源上的检查。八类各6例，含合法/错误/信息不足；都是A侧新编受控开发例，不宣称原独立封存48已完成。B侧独立48例留空、ready=false，不用同模板另改数字冒充封存集。

八类：任务语义、实际配置、trial身份、metric/split身份、否定/未来语态、整体/切片作用域、记忆状态、失败/比较状态。每例输入与期望来源可审查，归一判定confirmed/contradicted/unknown。检查器输出不参与生成期望；适配器只选择被测字段/映射已有状态，原输出保留。

含预先设计的反例：overall/slice比较若明确来自不同split/payload或不同primary metric，不得以数值涨分标supported；失败结果的残留数字不得作为有效观测。规则只检查事实与作用域，不认证自由文本因果。

初次失败完整保留。如发现生产检查器问题，先保存baseline和错误分组，再对通用比较规则修复；用原案例作回归，不能把修复后旧开发集称独立验证。每次输出新目录，源/案例哈希保存；不修改预期以追通过。

```powershell
python -m scripts.run_a67_case_registry prepare --output NEW_CASE_DIRECTORY
python -m scripts.run_a67_case_registry run --cases NEW_CASE_DIRECTORY/A_development.json --output NEW_RESULT_DIRECTORY
```

预计数秒；总48开发例不等于原96完整矩阵。原封存集合待队友独立贡献，B收到schema可直接交JSON，不需要重新写运行器。原baseline版本只能在操作API兼容时比较，本批首先验收当前源和修复差异，不拼成旧版总体收益。
