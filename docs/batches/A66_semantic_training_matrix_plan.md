# A66：补齐A1八格实际训练输入探针

起点c1514db，A累计999fit。按用户“开始”继续D3，预登记新增最多8真实fit/API0，累计最多1007；1000为原建议线，新增8次专用于缺格验收，不扩性能或提示小试。H/test不访问。

## 固定矩阵

preference/symmetric × primary-search/full-refit × split seed19411/19421=8格。相同32行匿名合成双文本，标签为二分类；每个语义的两个分区分层取24训练/8查询。primary只训24，refit训全32并以mode=refit走正式worker重训路径。两refit即使数据相同也分别实训并计数，作为两条路径验收而非独立统计样本。数据是人工性质探针，不能称真实性能或未见迁移。

候选固定：preference word_C0.05；symmetric symmetric_word_C0.3。同一原生产候选、experiment_worker、fit_guard及metric。用观察器在LogisticRegression.fit调用前落盘实际稀疏矩阵和y，不改矩阵/标签/模型配置。每worker限1监督fit，核对观察器调用数与admission账本。预处理fit不计监督fit。

## 验收（执行前固定）

- preference实际矩阵必须是训练内TF-IDF左右差与相反数拼接，实际标签必须是原标签与1-y拼接，实际分类器行数为2N。
- symmetric实际矩阵必须是训练内共享TF-IDF的绝对差与逐元素积拼接，实际标签严格等于原y、不复制或翻转，分类器行数为N。
- 从已拟合vectorizer手算预期矩阵，与实际捕获矩阵比较，误差≤1e-12；同时测交换概率：preference正类概率互补，symmetric概率相同，容差1e-8。
- primary查询独有单词不得进入训练词表；refit全32行进入词表是合法重训，不把它当泄漏。保存每条train/query行ID与输入指纹。
- 同一格将候选改为相反语义候选再调用worker：必须在任何分类器fit前拒绝；8受控非法尝试不能选入模型，也不计成训练0成本的性能路径。保存returncode/原始错误，预期拒绝不自动重试。

全部结果均留存，失败即停，不能改seed/候选/阈值追通过。正式任务入口的完整自主链已有A26等验收，本批专测共用worker的primary/refit边界，不伪装完整8条自主搜索。

```powershell
& 'D:\codex\python_withai\.venv\Scripts\python.exe' -m scripts.run_a66_semantic_matrix --output 'D:\codex\python_withai\runs\a66_semantic_matrix'
```

脚本与观察器先保存。预计8训练+8拒绝子进程约30–60秒，串行上界16×60秒；不请求API、不打开课程数据。输出独占新建，源码/观察器指纹运行期不变。归档仅统计与指纹，实际矩阵/模型/合成输入留本机。
