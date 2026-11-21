# A34/A36：跨输入类型与紧预算的路线门槛检验

在A32外层结果出来前预登记，属于23:24–02:24三小时窗口。A32运行期间只写计划/代码、prepare数据，不并发训练，不改其生产源码/runner/观察器。共享A31后的同一生产源码、方法与能力注册，无H/test访问。

## A34：已见单文本与照片开发流程

任务SMS（macro-F1）、Olivetti照片（accuracy），原已登记训练/留出包。raw动态、门槛off/routes各一条，顺序SMS off→routes、照片routes→off，共4完整路径，最多44实际fit/12请求；每路径11fit/3请求、600秒/worker60，估计10–20分钟。目录分别只有6/8候选，可能都能遍历，因此本批主要检验通用结构接入与无明显退化，不以零分差判定门槛无价值。两个模型对每任务全封存后统一留出评分；为简化一致地等全部4模型冻结。它们已是开发任务，不能改称未见迁移。

## A36：条件紧预算对照

只有A32完整通过、门槛确实增加预期覆盖且窗口余时≥45分钟才启动，不以是否涨分作为条件。四开发任务Coupon/偏好对/SMS/照片，raw动态、门槛off/routes，fit-limit=5含refit，仍最多3请求。每任务顺序交错off→routes / routes→off，8路径最多40fit/24请求，估计15–30分钟。先不重做完整11fit；与A32/A34同版本11fit路径只作预登记的跨批预算参考，清楚标注批次/顺序，不能声称所有不同预算模型都在首次查看val之前封存。A36自己的8模型全封存后才评分，参数/任务/seed不依已有val修改。

预算减少后路线门槛仍尊重原allocator；若预算使确认/停止先发生，覆盖目标可能未完成，必须报告实际而非强称保底覆盖。检验的是有限预算下是否少浪费重复路线、是否影响效果/资源，而不是给每组无限方法。

## 资源、隔离与代码

统一同一真实累计ledger、规范task1/task2/sms_spam/olivetti_faces；每paid臂.30元估计阈值，A34总1.20、A36总2.40，每任务每批.60，非平台扣费硬上界；原11历史未知保留、总预算未核验。失败停止批次、无重试，剩余窗口<660秒不启动新路径。

prepare包含四任务的当前source/TaskSpec/能力/707、808、909实际划分和原始源/留出文件摘要；只训练包进入CLI。原课程val、其他holdout labels都到各批全部模型封存后才反序列化；正式test不读取。使用同一独立worker fit观察器、相同CPU线程2、正式predict与新进程验证。原数据/模型/payload/SQLite/Key不进Git。

scripts/run_a34_transfer_coverage.py prepare与execute（默认仅校验，--execute才训练/请求）；--task-set multimodal --fit-limit 11为A34，--task-set all-four --fit-limit 5为A36。要求--after-matrix已completed且源码一致；每批新目录。每小批完成先分析，不为凑三小时重复收费或空等，余时不足就标未启动。
