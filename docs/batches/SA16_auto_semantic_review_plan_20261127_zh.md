# SA16：本地可读不等于理解完成，CLI自动语义复核

基线b620b83。先计划/脚本/基线，再改代码再验收。A修改语义入口和CLI，不改B算法/排名；用户已授权自动下一步。

## 假设和结构

默认fallback在原生输入可读时直接返回，因此文档里非默认列名、ID、类别映射、分隔符或概率列可能从未进入系统。本批先用既有开发微案例（普通CSV/显式manifest）复现“输入接入成功，输出契约缺失”，不记为模型质量或独立迁移结果。

人用CLI train/solve与superagent unified新增auto并设为默认：已有有效artifact直接恢复；已有模型client且有文档，则自动走全局解释（显式manifest只准输出overlay）；未配置模型或没有文档则沿用现有本地路径，日志说明未做模型复核。blocked/耗尽模型不当成成功语义复核；已配置但预检失败时，不得把未知语义标记完成。自动模式不改变权限或创建账本。

显式fallback/proactive保留；程序化train/resolve_training_task等API的旧默认保留，避免悄悄改变历史实验。显式local/control评审维持既有无付费语义入口，若要求LLM读题可显式proactive；legacy模式保持旧路线。cache模式不初始化隐式计费client；auto与B开题共用原client/最多3请求，不另外加预算。

## 实验和预算

1. 零fit/API基线：标准CSV与显式manifest各一例，fallback实际读入输入但不调用开发模拟client，输出规范不完整。日志/原输入冻结后保留，不回填成功。
2. 新增0fit/API检查：auto修复两例、manifest输入不被覆盖、本地无client/无文档/显式fallback兼容、blocked/耗尽/配置失败、两个CLI默认auto及内部子进程接线、缓存无计费、legacy保留。加原261相关回归；不通过不实跑。
3. 新公开开发封装：从SA14已用Iris训练/预测特征做带显式manifest的标准CSV，要求自定义管道分隔/BOM、原ID、三类映射与指定类概率。数据重封装不是新任务迁移；manifest不写输出，原文是输出依据。先落盘代码、注册表与期望。
4. 正式standalone solve不加解释/开题覆盖参数，stdin关闭；最多2实际fit含refit、3Flash共用、¥.25估计停止阈值、240秒。允许B默认付费决策，按角色报告；不注入模型/工具/算法/外部oracle，oracle仅结束后核对。
5. 起点真账本480/known8.103240/unknown16/pending0，旧行SHA/费用null保留，新unknown/pending停批次，不自动重试。原输出/代码/失败版本不覆盖，结果须解释未复核的路径和其他格式限制。

预计实现/离线20–45分钟，真实小试1–4分钟。独立新任务验收、flat uncertainty的作用域和未知数组元数据探查仍另行登记，本批不声称一并解决。
