# A98：修复自然日志中的历史基线误报，隔离开发

真实发现：A89 R1 trial2 reason中的“It is a reasonable next primary-split probe after a strongly regularized one-hot logistic baseline.”描述此前logistic基线，却因当前对象是CatBoost被标contradicted。该项经本轮开发审查是引用范围误报，不能算真实AI错误机会。

在独立codex/prose-baseline-scope-20261123工作树开发，不改活动A97源码。先保存16条有明确期望的开发句子和只读审计runner，记录修复前结果；再修窄语法：after/following的历史baseline名词短语只记提及，遇到明确的current/selected/this model主语时重新绑定，不能让一句历史描述把后半句真实错误也豁免。保持否定、预测、不确定和无法绑定的review为unknown。

运行前后重放相同16条开发用例及A89原12提案的原文/原history；比较逐条machine标签，不重训、不重新调用、不改原实验产物。运行相关prose/measurement/审查回归测试，不跑含训练的性能测试。0fit/API，预计开发20–40分钟、检查<1分钟。

A97全部结束后才允许将通过检查的改动整合；A89/A96/A97仍按各自原source解释，修复后的标签是另行后验审计，不能改写旧completion或宣称新端到端收益。保留初次失败，不反复改期望凑通过。16例和本次开发者对自然句子的标注不替代独立48/24语义标注，不认证自由语言理解。
