# A151堵点二：免费复核与无损菜单编码小试

先登记后执行。此批仅离线诊断，不重放失败调用，不读取密钥，不训练，不改生产默认或真实账本。

1. 检查本机requests/urllib3及新建Session的retry策略，验证connect/read错误在零重试策略下立即抛出；不能据此断言学校网关内部无重试。
2. 检查SSE行缓冲和空行统计：证明data_events=0不能推出连接没有收到字节，当前other_events也漏空行。此批不改流式生产实现。
3. 读取A151已存llm_request，离线把菜单的重复字段与同结构对象做可逆编码。保持全部候选、参数、事实、history、checks、memory与原输出契约；对解码后完整payload做严格相等断言。记录实际字符/字节，不以字符估计当token实测。
4. 输出脱敏protocol/results/completion；不导出原始messages、数据、响应或密钥。输入只读，输出必须新目录。旧失败与费用记录不动。

上限：0fit、0模型API、0课程新增费用；预计10分钟内完成。脚本scripts.review_a151_blocker2，测试tests.test_a151_blocker_review。成功仅代表程序无损和本机传输配置核实，不代表LLM理解/合法提案率/时延恢复。

后续建议：先独立实现显式menu_view开关并离线验收，再新登记一个最多1次Flash、0fit、原3次请求上限内的机制探测，不自动执行。原始A151不重试、不覆盖。若要压缩输出，定义独立新协议，不自动补全grounded断言或伪造模型意图。
