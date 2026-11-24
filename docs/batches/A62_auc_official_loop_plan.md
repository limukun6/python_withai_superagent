# A62：AUC正式入口与概率交付，单条真实小试

前置：A60完成26fit，A累计996；A61五模型概率/列序审计0fit通过。此次最多新增3fit/API0，累计最多999，不扩正式矩阵，不取H/test。

## 先实施与检查

正式worker保存概率矩阵、类列序及其指纹，绑定candidate、metric、payload、labels和硬预测。AB状态将绑定分数交给可信本地调度器；比较必须重算声明的AUC，缺分数/列序错误/行序或身份破坏就拒绝。Flash公共提示不携带原始概率或标签，仅保留主指标观测。概率列不得当hard-label证据，也不能用accuracy代替AUC。部分交付须复现分数和硬预测；A/B导出保留同样证据。原accuracy/F1不强制生成分数，保留默认行为。

使用人工案例0fit验证：相同硬预测但相反AUC、同步交换概率列/类名不改变AUC、只改分数/列名/metric/candidate/payload指纹会拒绝、没有可信标签不能确认AUC。此前A60源码保持其历史指纹；后续防御性缺失证据修复在新源记录，不将旧成绩冒充新源正式结果。

## 真实运行

从A47已见SMS的train-only包复制输入，manifest明确metric=roc_auc、positive_label=spam，其余数据不变；新包是同任务换目标，不是未见迁移。使用正式CLI、本地ab-v1、full候选、seed707、fit-limit3（两次探索加一次完整refit）、600秒/worker60秒。训练期间无外层验证。

模型冻结后，以新AUC任务规格从独立进程导出既有开发features，检查CSV概率与类顺序、row_id、新进程预测一致；再用同一开发holdout计算诊断AUC。所有步骤禁网、fit观察器开启。失败即停保留，实际fit仍计入996基础。成功标准是声明目标贯穿状态/主指标重算/选型/refit/导出，而不是成绩超过旧macro-F1系统。

```powershell
& 'D:\codex\python_withai\.venv\Scripts\python.exe' -m unittest tests.test_score_evidence tests.test_ab_finalists tests.test_ab_scheduler tests.test_ab_fixed_schedule tests.test_ab_confirmation_budget
& 'D:\codex\python_withai\.venv\Scripts\python.exe' -m scripts.run_a62_auc_official_loop --reference 'D:\codex\python_withai\runs\a47_catalog_Q' --output 'D:\codex\python_withai\runs\a62_auc_official_loop' --execute
```

先落盘计划、runner、protocol与源码指纹；输出目录不得覆盖。预计真实路径10–30秒，外层660秒上限。仅一任务一分区、少量方法，不能证明多分类AUC完整训练/Flash-AUC行为/未见迁移/效果收益。后续完整实验预算先根据新进度重新制定，不能悄悄超过建议1000fit停止线。
