A线演讲材料｜2026-11-24

主讲5页，150秒；另有2页技术问答/证据附页（已设hidden，不计主讲）。
讲稿：A_LINE_2M30_SCRIPT_ZH.txt，含逐页时段、完整中文讲稿、评分对应及技术问答。
演示文稿：已保存于用户个人Spaces，Page ID与artifactRef见delivery_manifest.json。
在演示文稿查看器可导出PowerPoint .pptx；本材料没有额外生成未经验证的本地PPTX副本。
不需要运行项目训练或学校API；打开文稿后直接演讲，逐页备注含讲稿。

内容说明
材料聚焦A线：任务契约、证据核验、反馈压缩和中断交付。
A145/A146的受控核验、A140的token节省、A141的恢复工程证据分别标明批次。
A147的负结果与A146新增核验成本保留，不能声称所有模块都提分或节费。
指标来自Git 500983f之前已有结果，各批源码身份各自保留。
演讲稿含全部来源路径与公平性/通用性/错误率/失败重试问答。
最新97源完整grounded路径和独立交付仍未验证，不用旧模型成绩替代。

构建代码
build_a_line_deck.mjs导出Artifact Session callback build({presentation,input})。
用于已授权Spaces页面：create_spaces_artifact后connect_spaces_artifact，复用返回artifactRef。
新空文稿依次调用phase cover、architecture、evidence、efficiency、conclusion、appendices、notes。
notes需input.sources、input.qa、input.boundaries，来源/问答可从讲稿对应分隔段读取。
不要在已有成稿上重复执行add页阶段；编辑现有页或在新的个人文稿重建。
图表与表格为原生对象，可在文稿中继续修改；字体为Microsoft YaHei。
对当前成稿做修改后仍需逐页渲染并复核。
