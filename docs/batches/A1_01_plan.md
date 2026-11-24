# A1-01：任务语义与训练标签一致性

状态：运行前计划，2026-11-20。总计划 A1 的第一小批，负责人 A。工作目录为仓库下 runs/integration；不改队友 B 的算法实现。

问题：semantic 目录的四个 pair recipe 未声明 swap_augmentation，但实际训练会拼接负特征并翻转标签。原过滤器把它们放行到 symmetric 任务。交换探针还会把全 0.5 的预测优先判为 flips。

假设：依据实际训练入口识别 recipe 的翻转规则，并对未知能力保留 unknown、在执行前验证，能阻止这条错误路径；双重满足时输出 ambiguous 能避免错误语义确认。

范围：修复语义过滤和 worker 搜索/确认/refit 的执行边界；不新增真正的对称算法，本批结束后 symmetric 无合法候选时应诚实拒绝。后续再把合法对称动作接入。

对照：先运行 baseline 保存当前反例，再修改代码运行 fixed。同一 runner 保存源码哈希、Python/依赖、逐项观察、完成状态、耗时。病例覆盖四个实际 recipe 的训练输入、symmetric 放行数、preference 兼容性、0.5/近 0.5 歧义、非法概率、未知语义声明状态、未知候选能力与 worker 防绕过。

资源：零 API、零完整分类器拟合。训练输入在 LogisticRegression.fit 入口拦截后立即终止，检查实际特征/标签；这是执行路径诊断，不是训练成功或性能实验。回归测试中的小模型拟合另记，不与本 runner 的零拟合混淆。预计机器 1–3 分钟。

验收：四个 recipe 仍如实标识 flips；symmetric 不再放行；preference 不被误拒；两误差均达阈值时 ambiguous；未知声明返回 null；执行器在解析动态候选后、fit 前执行同样契约；相关回归通过。

复跑（仓库根目录 PowerShell，输出名必须全新）：

```powershell
& .\.venv\Scripts\python.exe .\runs\integration\scripts\run_a1_semantics_batch.py --phase baseline --output .\runs\a1_01_baseline
# 实现修复后运行；baseline 必须用修复前代码，不能事后伪造旧结果。
& .\.venv\Scripts\python.exe .\runs\integration\scripts\run_a1_semantics_batch.py --phase fixed --output .\runs\a1_01_fixed
& .\.venv\Scripts\python.exe .\runs\integration\scripts\run_a1_regression.py --output .\runs\a1_01_regression
```

回归 runner 自动切换到 runs/integration，保存 tests.log 与 completion.json，最长 600 秒，不覆盖输出。测试中的小模型拟合不独立计数，只报告测试条数；本批机制 runner 的完整拟合数仍为 0。

给 DeepSeek 的续跑说明：先读此文件与 docs/EXPERIMENT_PROGRESS.md；检查已有 completion.json，完成批次不要覆盖或重复计费。运行未完成则用新输出目录，保留失败记录。先检查当前源码是否已经修复，不再从当前代码生成 baseline。下一批 A1-02 为配置事实与证据核验，不擅自扩大到付费 A2。
