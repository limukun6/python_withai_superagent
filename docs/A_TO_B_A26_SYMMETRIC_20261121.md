# A → B：A已按用户要求补齐对称方法，请勿重复开发同一基线

用户2026-11-21明确要求A直接补齐。取包含本文的codex/expanded-feedback提交；新增superagent/symmetric_text.py，注册点ab_methods/ab_catalog和语义守卫task_semantics。B调度逻辑没有修改。

四候选：symmetric_word_C0.3 / symmetric_word_C3.0 / symmetric_word_char_C0.3 / symmetric_word_char_C3.0。共享训练内TF-IDF，两侧绝对差+逐元素积，LogisticRegression，每评估/重训均1监督fit。目录route=symmetric_word或symmetric_word_char，支持概率，swap_label_rule=preserves；语义按已知fit函数身份核验，非凭flag放行。沿用origin=b_port扩充目录桶，新增实现作者为A。

preference目录不受新增方法污染，关系/语义不明任务仍无相应候选。116项回归通过，匿名正式CLI 5fit/8.570秒，新进程原/搬迁/交换输入预测24行一致，最大概率交换差0。仅合成工程验收，不是PAWS语义效果。

封存注册Spec检查：H1从0到4合法候选，H2仍8；H原始包未打开、模型未训练未评分。请继续Q/方法效果与预算实验，不再等待或重复这一词法基线。更强对称语义路线可以作为另登记扩展，但不能先看H分数再针对它调方法。

Flash已新试一次，在DNS getaddrinfo11001失败，校园IP443也超时；不是历史账单等待。错误/本机测试命令CAMPUS_FLASH_A27_FAILURE_AND_TEST_20261121.md。不要同步提交密钥或SQLite；真实两人调用要串行共享最新累计账本。

另修A09回归fit硬编码：四个旧schema测试每轮实际训练4个小模型，原10轮回溯补40，本轮26fits含回归21/CLI5；累计本A窗口449（含回溯更正，非全项目）。原始失败/计数未覆写，见batches/A26_A27_results.md。性能矩阵的物理fit计数不受这次旧回归补记影响。
