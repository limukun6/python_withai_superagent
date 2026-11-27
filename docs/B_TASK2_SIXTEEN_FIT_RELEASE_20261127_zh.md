# Task2 自主选择的 16 次拟合交付

在统一版本的普通入口上，只给课程训练原包、完整资源目录、预测输入和数值预算。Agent 最终选择 BGE-small 与 GTE-base 等权融合，在课程指定验证集答对 **701/940（74.57%）**。旧默认设置答对 700/940，对比有 27 题改对、26 题改错。两版均已独立进程断网重载。

启动时使用任务包实际路径、特征包实际路径和本机通用权重目录。训练允许 16 次实际拟合（含全量重训），墙钟上限 1,800 秒；本次模型 API 为 0 次、0 元。

```sh
python -m unified_agent solve "$TASK_PACKAGE" --weights-dir "$WEIGHTS_DIR" \
  --fit-limit 16 --time-limit 1800 --prediction-input "$FEATURES_PACKAGE" \
  --run-dir "$RUN_DIR" --output "$PREDICTIONS"
```

| 选模时可见的训练内留出 | 单模型 GTE-base | Agent 选出的融合 |
| --- | ---: | ---: |
| 首轮 1,693 行 | 1,228 | 1,232 |
| 第二份 1,645 行 | 1,179 | 1,188 |
| 第三份 1,651 行 | 1,176 | 1,191 |

候选名称换成三套匿名编号并逆序后，15 步合计 45 次决定一致。把模型、两份固定权重和程序搬到新目录后，断网复现 940 行。资源目录由开发者准备；实验、确认及最终选谁由运行中的 Agent 决定。

另取以前封存的两套训练集分组划分，每套包含五折、完整覆盖相同的 8,301 行。不增加拟合，仅检查这次冻结的融合是否也能超过普通默认。两套分别比 GTE-base 多答对 37 行（6,088 比 6,051）和 44 行（6,083 比 6,039），十折中八折获胜。另一种已验证的双 GTE 融合在两套上分别为 6,104 和 6,071，两个融合各赢一套。因此目前有支持 BGE-small 加入融合的开发证据，陌生任务的具体收益仍未知。正式测试集和 H2 封存。

[启动记录](../research/paired_encoder_search_1127/ordinary_sixteen_launch.json) · [对手和各折选择](../research/paired_encoder_search_1127/ordinary_sixteen_finalists.json) · [指定验证与逐行变化](../research/paired_encoder_search_1127/ordinary_sixteen_score.json) · [匿名重放](../research/paired_encoder_search_1127/ordinary_sixteen_decisions.json) · [交付包回执](../research/paired_encoder_search_1127/ordinary_sixteen_portable.json) · [跨划分结果](../research/paired_encoder_search_1127/ordinary_sixteen_transfer_results.json)
