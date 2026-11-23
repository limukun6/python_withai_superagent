# A99：将likely/unlikely to干预假设与当前对象事实分开

A97结束后发现R1 trial4原句“increasing logistic C is unlikely to help ... while extra trees ...”被标logistic与当前extra_trees矛盾。这是另一个开发审查认定的误报，不改写原机器日志。

先保存10条定期望开发句子和只读重放脚本，记录A98修复后的基线；再只将likely to/unlikely to作为现有不确定表达语法的一部分。裸likely不豁免明确的当前错误；不同句子的确定事实仍单独核对。复用现有unknown政策，不让假设被确认，也不改结构化配置守卫。

同10用例修复前后检查；A97原12提案原文/原history后验重放，保留原/新标签；运行prose、measurement、review、audit、comparison回归。全部0fit/API，预期<10秒检查、15分钟实现分析。小批先落盘，原始产物不覆盖，不以修改用例期望凑通过。

本批不是新端到端性能实验；开发标签不替代独立48/24语义标注。当前C1仍锁定c64bee4的旧生产内容，未来登记新版本；不拿后验标签当修复前纠错成功率。
