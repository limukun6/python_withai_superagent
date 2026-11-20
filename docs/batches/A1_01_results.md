# A1-01 结果：阻止对称任务使用翻转标签训练

2026-11-20。这是 A1 的一个小批次，属于工程机制验证。没有测任务性能或泛化收益。

## 瓶颈与结构

原候选过滤器只读 swap_augmentation 标志，漏掉四个在实际 fit 内执行翻转增强的 recipe。现从已知 recipe 的实际训练入口识别 flips；没有声明且不能识别的能力为 unknown，不能被当成适用于 symmetric。

过滤后没有合法候选时明确拒绝。worker 在解析静态或注册候选后、调用 fit 前再次核验；搜索、确认和最终重训的数据载荷都传递 pair_semantics。新增合法对称模型属于后续算法能力工作，本批未实现。

交换探针两种误差同时达标时记为 ambiguous，不再优先称 flips。概率越界被拒绝，未声明语义或歧义结果不确认与任务一致。

## 对照结果

| 检查 | 修复前 | 修复后 |
|---|---|---|
| 四个 recipe 实际传入 fit 的后半段标签 | 均为 1-y | 均为 1-y：训练方法没有被暗改 |
| symmetric 是否排除这四个 recipe | 0/4 | 4/4 |
| symmetric + semantic 目录 | 放行四个错误候选 | 无合法候选，拒绝运行 |
| preference + semantic 目录 | 正常放行 | 仍正常放行 |
| 相同 0.5 与相同 0.52 的交换预测 | 均判 flips | 均判 ambiguous |
| 非法概率 1.1 | 接受 | 拒绝 |
| 本批九项固定要求 | 1/9 | 9/9 |

这些是针对已知故障预先登记的回归案例，不能解释成在独立测试集达到 100% 检查准确率。额外单元测试覆盖 unknown callable、实际注册 recipe 绕过静态目录、search/refit 两种 worker 模式，以及未声明/歧义的一致性状态。

机制 runner baseline 用时 1.248 秒，fixed 用时 1.218 秒（不含 Python 进程启动）。两轮各拦截四次分类器训练入口并立即终止，完整分类器拟合均为 0，API 请求均为 0，API 费用为 0。回归测试另有小模型拟合，未独立计数，不混入零拟合的机制实验。

七个实际存在的回归测试模块共 91 项通过，测试执行 40.742 秒，回归 runner 墙钟 42.418 秒。第一次手工命令多列了不存在的 test_recipe_models 模块而失败；之后修正模块清单完整重跑，最终通过。没有付费请求失败或重试。

## 证据与复跑

- 原始产物：主仓库 runs/a1_01_baseline、runs/a1_01_fixed。
- 可共享摘要：research/a1_semantics_guard/baseline、fixed，含计划身份、源码哈希、逐例观察与 completion。
- 回归日志：主仓库 runs/a1_01_regression/tests.log 与 completion.json。
- 续跑入口：docs/EXPERIMENT_PROGRESS.md；计划与命令见 A1_01_plan.md。

本批实际使用工作区文件快照，尚未整理成新的 Git 提交。源码身份以 protocol 的内容哈希为准，不把旧 HEAD 当作本次实现版本。

## 下一步

A1-02 先制定小批计划与 runner，再修分类器真实对象误映射，以及证据中的试验/指标/划分绑定；之后接通检查进入下一请求。新增真实 symmetric 方法需与 B 的候选接口协同。未接通正式 CLI、未做真实任务性能验证前，不宣称通用流程已全部完成。
