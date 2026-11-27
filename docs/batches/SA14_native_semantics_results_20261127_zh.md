# SA14：无manifest数组／图像，正式自主流程2/2通过

基线dee7140。A改动是任务语义接入，不改B方法目录、排名或算法。新生产代码superagent/native_semantics.py，统一解释器、预测加载器和语义缓存接线。

## 瓶颈 → 结构 → 实验 → 结论

| 瓶颈 | 结构 | 验证 | 支持到哪一步 |
|---|---|---|---|
| 全局LLM契约只能描述标量记录，数组/图像仍需手写manifest | semantic-native-v1：LLM声明任意安全成员名、数组特征/标签键、图像根目录、指标和像素处理；已有reader执行 | 16项新检查；真实NPZ与图像目录各一条正式solve | 两个给定开发包无需改文件名/补manifest即可接入 |
| 每次预测重新猜结构，容易标签/顺序/尺寸不一致 | native_input_recipe冻结来源、键、处理和指标，预测重放并检查shape/decoder | 两模型各新进程predict，CSV与原输出逐字节相同 | 本批54行预测跨进程可复现，无新增训练/API |
| 把LLM格式正确等同于理解正确 | 文档引用、实际训练/预测特征与标签指纹、类别/指标/导出语义外部核对 | 每任务9项契约检查全部通过 | 本批契约与预先封存期望一致，引用存在本身不是语义真值证明 |
| 缓存没有绑定新类型的输入含义 | 新reader加入17项源码身份，task摘要加入native recipe，恢复重新编译 | 图像artifact恢复与新进程数组重放检查通过 | 当前源码/同文件内容可以0API恢复；旧reader版本仍需显式重验证 |

```mermaid
flowchart LR
  A[原始任务包和文档] --> B[LLM联合解释]
  B --> C[有来源的JSON契约]
  C --> D[现有数组或图像读取器]
  D --> E[训练与预测结构预检]
  E --> F[B自主选模与预算内训练]
  F --> G[模型＋冻结输入与输出契约]
  G --> H[独立进程预测与CSV校验]
```

## 正式运行结果

两个包均无manifest，stdin关闭，从正式`python -m unified_agent solve --task-interpretation proactive --task-analysis local`启动。没有注入client、resolver、算法或预写模型回答；外部oracle仅在运行结束后核对，不能修复提案或选算法。两任务都是**第一次真实回复即被接受**，未激活字段工具、没有语义修复调用。

| 公开开发封装 | LLM正确解释的结构 | B自主选型 | 训练含refit | 真实API | 课程费 | 秒 | 预测 |
|---|---|---|---:|---:|---:|---:|---:|
| N1 Iris，96训练/24预测 | 非标准evidence.pack/future.pack，sensors→state，预测readings；balanced_accuracy；分号/BOM | ExtraTree | 2 | 1 | ¥0.006995 | 31.906 | 24行 |
| N2 Digits，90训练/30预测 | library/类目录，arrivals/无标签图；10×10 RGB双线性；macro_f1；路径排序 | 梯度直方图＋逻辑回归 | 2 | 1 | ¥0.008725 | 22.078 | 30行 |

合计**4fit、2真实API、输入6580/输出2285token、¥0.015720、批次56.141秒**。两任务均完整refit，新进程输出54行；独立审计5.344秒，新增0fit/0API。实际usage的请求ID/费用和canonical ledger逐条匹配，两个已训练模型的新进程CSV逐字节一致。

当前安装的本地读取器在不配置LLM时对这两个原包完整接入**0/2**；新语义契约＋真实Flash为**2/2**。这是输入覆盖的能力检查：没有做完整等预算策略消融，不据此声称准确率提升或普遍100%成功率。

## 离线与失败记录

完整相关检查**245/245、12.735秒、0fit/0API**，含16新＋原229项；两个旧Dummy.fit夹具仍显式排除。范围含NPY/NPZ非标准文件名与键、数值图像数组、图像目录/尺寸/模式、目标与指标、eval源/伪造引用/多余字段/显式manifest覆盖拒绝、预测shape和decoder漂移、pickle拒绝、缓存与新进程。

预检脚本首轮复制了前一天计划文件名，尚未启动测试即失败，记录research/sa14_native_checks_01；修正日期后_02通过。准备脚本首轮括号语法错误，尚未准备数据或调用模型，保留research/sa14_native_prepare_01；修正后_02两包全部准备通过。均0fit/0API，原目录不覆盖。准备阶段使用开发者预写提案检查读取器，不是真实LLM成绩。

## 账本、限制和下一步

真账本475→477，已知课程累计¥8.067821→¥8.083541；旧16未知原行SHA不变，待处理0、新未知0。所有历史null保留，供应商实际扣费和全团队完整账本仍未核验。本自动窗口SA7–SA14累计22fit/16API/¥0.117146。

本批没有读取包外预测答案、没有新评分。Iris/Digits均为公开开发数据，Digits来源以前用过；重新封装不是独立任务迁移，不能把格式接入改善当算法效果提升。

支持边界：现有NPY/NPZ数值数组、split/class/files形式的图像目录；未新增任意文件解码、顶层类目录/所有目录布局、图像文件名ID与任意sidecar ID接口、未知数组元数据探查。显式manifest继续权威；路径中的eval关键词仍有保守拒绝边界。默认fallback对专用数组reader的NativeContract阻断仍需单独处理，本批正式验收使用明确proactive。后续应审计默认入口是否存在类似“无显式契约却不让LLM接手”的分支，再做独立新任务验收；当前不冻结。

## 可复跑交接

- 计划：docs/batches/SA14_native_semantics_plan_20261127_zh.md。
- 无fit检查：scripts/run_sa14_native_checks.py，证据research/sa14_native_checks_02。
- 准备/实时：scripts/run_sa14_native_delivery.py（--mode prepare或live）；case builder scripts/sa14_native_cases.py。
- 实时证据：research/sa14_native_live_01；核对器scripts/audit_sa14_native_delivery.py，证据research/sa14_native_audit_01。
- 数据/模型：runs/sa14_cases、runs/sa14_native_live_01；不进Git。已尝试scope不能原样重跑或补花费；新批先登记协议。
