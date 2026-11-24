# A98：检查器也会串错对象，真实误报在隔离副本中修复

瓶颈：检查器只看算法名和前面的比较词，容易把“之前的logistic基线”当成“当前CatBoost”；反方向也会把历史句子的豁免延伸到后面的当前模型错误。AI的审查层需要先弄清每句话在说谁，不能见到算法名就判真伪。

真实例子来自A89 R1 trial2：`It is a reasonable next primary-split probe after a strongly regularized one-hot logistic baseline.` 其中logistic是此前的基线，原系统却标contradicted。开发者本轮复核认定这是机器误报，不能写成自然AI错误率或纠错成功率。

结构改动很窄：识别after/following后紧邻的baseline名词短语，把该算法词记成历史提及；遇到明确current/selected/this model等当前主语时重新开始引用范围，后面的错误分类器仍会被核对。没有放宽结构化配置校验；否定、猜测、复杂测量和无法绑定的review继续unknown，不认证整段推理。

证据：运行前先保存16条开发用例和期望，原实现6/16，新实现16/16；包括真实原句、历史与当前对象混写、后半句当前SVM错误、基线词单独出现、不确定表达等反例。重放A89原12条Flash提案的原文/原决策前history，机器矛盾标记1→0，消失的正是上述误报。30项prose/measurement回归及另17项review/audit/scope回归全部通过，共47项，不训练、不调用API。

资源与边界：前后重放各约0.031秒，0fit/API，A累计不变。检查是在独立工作树完成，未改运行中的A97；原A89日志、source、completion不覆盖。此处是后验开发审计，不能用修复后标签改写旧实验成绩，也不证明重新运行Flash会选得更好。16开发例和本轮开发者判断不是独立48例/24语义标注。after/following之外的复杂指代、中文和自由文本仍未全面解决。

整合门槛：A97全批结束后才可并入A；B本轮同源C1仍固定c64bee4的生产内容，不把后续修复HEAD和旧组静默拼表。修复后的版本用于下一次明确登记的运行或独立语义验收。
