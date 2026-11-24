# A60：预算内统一最终复核，先做真实本地小试

## 研究问题与边界

A58观察到早期赢家与后续挑战者所需证据不同；这不证明它造成外层退步。增加A自有可选`--scheduler ab-finalists`，不修改B的ab-v1算法，也不改变默认行为。比较的是整个预算分配/选型策略，不能将差异单独归因于公平复核。

现有A累计970实际fit，原总计划建议上限1000。本批仅两个已见真实任务coupon/preference × 原ab-v1/新ab-finalists × 每条最多7fit=28fit，API=0，累计最多998。旧ab-fixed要求11fit，本批不运行、不冒充7fit公平对照；完整11fit三组矩阵另需重新审视总预算，不能自动扩跑。无训练的契约检查不占fit；不运行会额外训练的宽回归。

## 新结构，运行前固定

- 仅允许已注册单fit候选，至少5fit预算，保留1fit最终refit。
- primary探索上限`min(目录数, fit_limit-3)`，小试为4个候选。探索仍调用原共同路线排序，关闭探索阶段的动态复核；不用任务名字选方法。
- 固定primary前两名（平分按candidate_id），各自重新训练并在同一个`confirmation_1`评价。不能用primary模型直接预测另一有训练重叠的划分。确认结果不能改短名单。
- 两候选都必须有primary与confirmation_1的哈希、行序、标签与主指标核验；缺一不算完整统一复核。赢家按两划分主指标均值，平分按candidate_id。只有一个合法候选时也必须复核它。
- 失败保留原始错误，无训练/API自动重试；可用已核验checkpoint作明确partial交付，不能标成完成。现阶段仅fit预留，时间硬保证尚未实现，不声称已解决时间预留。

## 实验控制

训练包与开发外层评价沿用A47冻结输入；只读既有train.zip准备，绝不打开H1/H2与课程test。每个任务两臂共用候选全集、seed707及三份训练内分区、时限600秒/每worker60秒、CPU2、profile=off。顺序coupon动态→统一，preference统一→动态。新源两臂都重新运行；A47/A50成绩仅历史动机，不能直接当同版本控制。

四个模型全部冻结之后才使用既有外层开发标签评分；每个模型独立进程导出预测、检查行ID/主指标/冻结SHA。记录每条实际physical fit、API0、wall time、primary候选数、双方复核覆盖、最终选型与得分。即便不涨分也保留。单开发划分/两个已见任务，不作为未见迁移或统计显著性证明。

## 可复跑

在本分支根目录，先0fit检查：

```powershell
& 'D:\codex\python_withai\.venv\Scripts\python.exe' -m unittest tests.test_ab_finalists tests.test_ab_scheduler tests.test_ab_fixed_schedule tests.test_ab_confirmation_budget
& 'D:\codex\python_withai\.venv\Scripts\python.exe' -m scripts.run_a60_matched_finalists --reference 'D:\codex\python_withai\runs\a47_catalog_Q' --output 'D:\codex\python_withai\runs\a60_matched_finalists' --execute
```

去掉`--execute`只校验准备，输出目录必须新建；已有结果不覆盖，不要重复执行同目录。运行器禁止网络，调用正式`python -m superagent run`，保存源码/计划/输入指纹、protocol/checkpoint/results/completion。中心估计2–6分钟，外层进程上界4×660秒约44分钟；慢模型首次失败就停止，不用重复训练凑完整矩阵。
