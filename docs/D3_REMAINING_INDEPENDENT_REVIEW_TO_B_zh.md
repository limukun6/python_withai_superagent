# D3剩余独立输入：给B侧Codex的具体交接

A已补完四任务B1固定/动态16路径与四任务A2的12状态/24Flash回答、两新划分动作评价。结果包含负面，不再追seed。A1已完成8格训练输入和统一A侧48开发例；D3不能全部勾完的独立输入如下。B仍继续方法/预算的实现与实验，这不是把B改成专职检查员。

## 1. A1独立48案例

按`docs/A1_INDEPENDENT_CASES_TO_B_20261122.md`提供八类各6例，共48，独立写payload/expected/expectation_basis，先冻结SHA再运行。schema/操作参考`scripts/run_a67_case_registry.py`，格式用`tools.validate_a1_case_pack`检查。不要复制A例改数字，不把已经看过输出的例子称未见。若不具备严格盲测条件，附清可见信息，作为独立作者补充例验收。

## 2. A2语义标注24份回答

只给标注者`research/a70_independent_review/blinded_reviews.json`，不要先给condition_key_keep_separate.json、R/E动作效果、机器flag。该文件已打乱顺序、隐藏R/E，含实际任务契约、历史测量、已选配置与模型文本，没有原训练样本/密钥。

逐项填independent_annotation：factual_error、scope_or_reference_error、unsupported_causal_certainty、appropriate_uncertainty可true/false/null，写具体label_basis和annotator、checker_outputs_seen。null表示无法判定，不当错误或正确。用历史真实事实、主指标、trial/split、当前/过去对象指向核对；不能用被测检查器结果作答案。

这里只隐藏了显式R/E标签；措辞本身仍可能透露条件，且同一Git里映射可见，不保证严格盲法。请记录标注时实际看过哪些信息，不能把“打乱顺序”直接当独立性证明。

“可能提高”不是已声称提高；数值预期null不是错误；重复同族也不是错误。回顾旧Logistic或树模型不等于声称当前所选模型就是它；不存在统计检验时，不把单次内部涨分讲成显著或因果证明。目的不要求凑出错误案例。

A70前缀来自正确执行的本地历史，结构化错误机会数为0，所以本批即使文字都正确，也不能给出从自然错误恢复的成功率。若要单测纠错，需要另一个预登记受控错误设计，明确注入，不混成自然发生率。此标注只补当前回答的语义质量。

## 3. 交回方式

提交独立case JSON与已标注review JSON、对应SHA和来源/可见信息记录即可；不需要发数据/模型/API Key。A负责用冻结源运行和汇总，B继续强方法主线。若队友已完成其它B1/Q结果，也附run源/目录/实际split/监督fit索引，避免两边重复训练。
