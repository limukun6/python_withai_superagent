# A→B：AUC证据接入与可选统一复核

基于faabc65之后的A60–A63增量，精确交接SHA以本次发布提交为准。A60/A62各自source_files见归档，不能把旧源结果当成最终整合版C1。

## A完成了什么，B接哪里

- 正式CLI增加可选`--scheduler ab-finalists`，A自有`superagent/ab_finalists.py`，B的ab-v1动作策略不改，默认仍原方案。已跑两真实任务4路径/26fit/0API；一持平、一+0.1064pp，不能宣布稳定更好。7fit小试未决定最终11fit调度规则。
- AUC支持从worker延伸到正式AB状态、重算核验、最终选择/重训、概率导出、partial与tools/export_b_handoff。对你维护的ab_scheduler.py，仅在_observations补充AUC概率绑定与本地重算，不改原排序、探索、配对胜出规则。此共享证据字段请按独立提交合并，不用旧文件整体覆盖。
- 新score_evidence.py由A维护；accuracy/F1原证据字段不变。probability记录只供本地可信组件，ab_flash.context_for仍不发送score向量或标签。

## 状态新增字段

仅score-required指标的每条prediction_evidence有`score_record`：

```text
schema_version: 1
candidate_id, metric_spec
scores: rows × classes（已归一概率）
class_order: 各列实际class
scores_sha256, class_order_sha256
prediction_sha256, label_sha256, evaluation_payload_sha256
```

外层item仍有trial/candidate_id/split_id/row_ids/predictions及其原SHA。调用`checked_score_record(record,item,metric,candidate_id,trusted_labels)`先核验，再`metric.evaluate(labels,predictions,matrix,class_order)`重新算主指标。行ID和payload必须与本地登记分区一致。

**AUC next_action/selection_evidence必须传trusted_partitions**，结构与正式AB相同：split→row_ids/labels/evaluation_payload_sha256。labels只留可信本地。终态B导出仍reserved_final_refit=0、executable_for_continuation=false，不当可重启请求；可用已运行A62的tools.export_b_handoff导出。字段整体复制不等于已核验数值。

## 可用入口

```powershell
python -m superagent run TASK.zip --scheduler ab-finalists --portfolio ab-v1 --fit-limit 7 --seed 707 --partial-delivery checkpoint
python -m tools.export_b_handoff --run-dir RUN_DIR --output NEW_STATE.json
```

AUC由任务manifest声明`metric: roc_auc`与需要时的positive_label，不能从任务名猜。无概率候选仍从AUC目录排除；新增方法需给出真实probability/class_order/fit/refit成本。当前统一finalists仅单fit候选，不能直接塞多组件方法。

## 下一步双方工作

A负责目标/证据/导出接入与统一入口；B继续强文本/图像/数值路线及资源调度、自己的完整实验。双方共同确定最终S，避免临时把新方案加成正式第七组。先合并真实可执行方法清单与依赖，跑不训练的接口核验，冻结共同源/目录/协议后再开C1/H。当前A开发训练累计999，下一阶段预算见D4_freeze_and_formal_budget_20261122_zh.md；不要重复A60/A62或把已开开发任务称未见。
