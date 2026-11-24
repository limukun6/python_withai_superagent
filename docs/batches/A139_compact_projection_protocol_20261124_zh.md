# A139：有来源、可无损还原的checked反馈去重

固定三份真实运行state_3.json（恰好前2个完成试验，不按错误选例）：A135 Coupon、A122 SMS、A122 Flowers。它们来自local历史，proposal/audit可能为空，不能把本批写成已纠正自然发生的旧LLM错误。原历史原文件SHA保持，预算视图统一3请求/已用0，其余字段不改。三种视图raw/full checked/compact checked共用task、history、候选、预算及calibrated-v2输出契约。

结构不是扩NLP规则。只对checked附加的checks/memory去重：重复的完整值引用已有common字段或先前checks，以明确JSON指针绑定；状态/unknown/contradiction/失败/未验证解释全部保留。提供expand函数逐项恢复原full context并验证完全相同。引用不能访问上下文外或形成循环；不执行数据或指针。完整参数定义没有在full里就不额外给compact，不混入信息增量。显式--feedback-view full|compact，默认full，compact仅checked-history允许；不改变动作验证、候选、训练及预算规则。

离线资格：三份expand(compact)==full；七个共同字段及可选training_profile保持原样；每份compact序列化UTF8不比full大；三份合计的checked附加开销至少减少20%。开销定义full_bytes-raw_bytes；只字节资格，不等同token或实际费。没有达到就保留负结果、不跑A140来追效果。正常case、含unknown/contradiction/failed与否定文本故障夹具分开；不修改旧真实历史。

最多0分类fit/0API。A138已12fit、累计1614。本批协议和runner先落盘，新增生产源码94文件与93源A138记录分开，旧Task1/H1不重标。达到资格后才按总计划A140最多9真实Flash，三任务轮换raw/full/compact，答案全部封存后独立评价机器可核对事实和覆盖率。费用从实际usage计算，不能把字节减少当省钱结论。
