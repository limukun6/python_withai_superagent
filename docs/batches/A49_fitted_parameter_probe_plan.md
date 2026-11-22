# A49：实际参数事实的独立0fit原型

瓶颈：现有observed_classifier主要确认分类器家族，LogisticRegression/CatBoost的parameters记录常为空。配置metadata写C=某值不等于已经从拟合对象核验C；错误的C声明仍可能通过“分类器正确”检查。

当前A47与A48生产源冻结，故本批只在tools实现独立观察原型，不改正式提示、worker或B方法。复用A44最终五任务完整模型，禁止fit/网络，不打开任何外层标签。按已知终端分类器读取get_params(deep=False)的白名单：LR/LinearSVC/SVC的C等，树的深度/叶子阈值，CatBoost的构造参数；另将实际tree_count_与配置iterations分开，不把训练迭代数当成相同概念。

对每个模型，比较已声明且可观测的参数；representation、特征工程超参、未知包装器或不支持的字段明确unverified，不能因为没检查就confirmed。再注入一条错误的已可观测参数声明（不改实际模型），验证旧的家族检查仍confirmed而新原型给出contradicted。五任务各一真实模型+一人工错误声明，0新训练/请求。无可观测参数则记unsupported，不换模型挑成功案例。

输出观察源/展开路径、参数、逐字段核对/未知、人工故障结果及模型SHA。只证明可补事实核验，不证明实际历史LLM犯了这些错误，不证明下轮选择或精度收益。报告明确未接生产。后续若纳入正式worker，需新版本合同及同信息对照，不能反改已有A47结果。
