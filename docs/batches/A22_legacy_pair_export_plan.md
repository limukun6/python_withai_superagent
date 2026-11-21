# A-22：课程文本对语义契约的预测格式闭合（2026-11-21）

启动A20前静态检查发现：A19给legacy pair任务新增input_schema，预测端因此进入native prediction_manifest分支，但旧spec缺input_format/columns。若直接跑可能先训练再导出失败。

本批0fit/0API，在任务声明识别后补齐已由legacy输入列验证的CSV格式、headline_A/B文本列、target=label_pairwise。声明来自README的语义不变；不是猜新字段。新增train-only spec直接加载无标签test.csv的往返检查，确认不读取任何y/test标签或重推断模式；保留逆向/否定/manifest优先检查。通过后重建五任务就绪源指纹再启动A20，模型不会在此补丁之前训练。H无改动。
