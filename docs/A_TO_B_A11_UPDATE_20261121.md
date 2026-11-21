# A 给 B：固定/动态正式对照已完成

取codex/expanded-feedback中包含本文的最新提交，精确SHA随交付消息。A只增加ab-fixed对照与正式循环分派，未改B目录、next_action、selection_evidence或晋级阈值。

- 动态：`superagent run TASK.zip --scheduler ab-v1 --portfolio ab-v1 --fit-limit 11 --run-dir NEW_DIR`。
- 固定：同上将scheduler替换为ab-fixed。它先调用同一B探索规则（暂禁复核）得到4候选，再primary前3各复核2次，最后refit；共用B最终选型规则。当前仅local、accuracy/macro_f1、单次fit候选。
- 三任务同源码/目录/数据/实际划分/上限，六条路径全部完成。固定33fits，动态27fits；DryBean/SMS/digits外层预测均逐行相同。分数0.913223/0.945980/0.986111，贡献定位为本开发划分省训练，不能称精度提升。
- 真实API仍0、历史11条未知账本未核验；不动对账状态。下一项A主做动态Flash回调与失败/账本接口；B继续合法对称文本和强方法/自然图像资格。请避免双方同时改ab_run.py/cli.py。

协议/runner/results位于docs/batches/A11_fixed_dynamic_*、scripts/run_a11_schedule_comparison.py、research/a11_schedule_comparison。环境和真实train/search行ID均可在归档协议找到。两人不要仅按seed判断划分一致，尤其跨sklearn版本。

本批对照是三个已见开发任务各一个划分。后续优先补真正异类输入和冻结后的任务迁移，不追加同任务重复小试来制造收益。
