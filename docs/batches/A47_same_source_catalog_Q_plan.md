# A47：五真实开发任务的同版候选库Q对照

在训练/评分前登记。网络当前DNS失败，停止真实Flash，但仍推进完整Agent的可执行方法与模型交付。A44–A46新结构的合同验证不能代替方法效果，本批单独检查“方法库是不是瓶颈”。不修改B的方法/调度实现。

## 问题和比较

同一新生产源码、同一AB本地动态调度、相同707/808/909划分和11实际fit上限（含refit），比较--catalog-scope baseline/full。其他开关都off/strict-v1，不能把差异归因于训练摘要或诊断契约。五个已见真实开发任务：coupon、preference、SMS、photo、DryBean。正式test/H保持封存。

先在只含训练成员的既有包上重新核验TaskSpec和每个候选的可执行元数据，按当前源码冻结。若某任务baseline/full字典完全相同，只执行一条shared路径，并明确这是两组同策略去重，不能把同一结果当两个独立样本。0fit预检发现preference也是两目录完全相同（22/22，额外冻结文本编码器没有本地权重，未激活），原先预计8路径已拒绝、未训练；现在在标签未开前改为coupon/photo各两条，preference/SMS/Bean各一条，共7物理路径。不能将没有激活的新方法写成已对照。

顺序预登记：coupon baseline→full；preference shared；SMS shared；photo baseline→full；Bean shared。每条最多11监督fit、600秒路径/worker60/外层660，CPU2，API=0。最多77实际fit，不凭成绩追seed或重复训练。另一次0fit预检由PowerShell把ISO日期自动转成区域格式而拒绝，改用--window-config让Python读原始JSON字符串；原错误目录保留。

所有物理模型封存后，才统一读取原指定val/已登记开发holdout。正式predict和独立重载逐行一致，再计算主指标与预测差；写来源与模型SHA。不得把旧A20/A29不同源成绩当本批baseline，也不因中途评分调整候选。全批失败停、原产物保留。

## 报告与界限

逐任务报告baseline/full候选数、实际训练/耗时、选中方法、主指标/行数/预测差。目录相同的任务显示deduplicated_same_policy。对照仅支持这个固定开发划分上增加现有方法的边际收益；未见迁移、Flash自主决策、摘要/诊断契约收益均不能由本批证明。真实ML结果和A46人工故障验证分开。

代码scripts/run_a47_catalog_Q.py默认只注册和校验，--execute才训练。新目录a47_catalog_Q，运行器与生产源码哈希冻结；账本不修改，0HTTP。预计15–25分钟；若恢复窗口剩不足30分钟则只完成当前注册，不启动整批。
