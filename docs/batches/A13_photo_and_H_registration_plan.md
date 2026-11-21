# A-13：真实照片流程与迁移任务登记（2026-11-21）

本批在获取数据前登记。源码先等待A12验收结束再冻结；不与模型运行同时改生产代码。真实API0。

## 图像开发任务

采用AT&T/Olivetti照片作为图像开发任务，官方Cambridge档案说明使用时署名AT&T Laboratories Cambridge；sklearn提供400张、40类、64×64版本。来源 https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html 与 https://scikit-learn.org/stable/datasets/real_world.html#the-olivetti-faces-dataset 。下载 https://ndownloader.figshare.com/files/5976027 ，用本地sklearn登记的SHA256 b612fb967f2dc77c9c62d3e1266e0c73d5fca46a4b8906c18e454d41af987794 校验，不安装依赖、不覆盖既有缓存。

只读解码MAT数据，固定灰度像素0–255，不从留出数据拟合归一化。外层grouped_partition(seed45101)，内层707/808/909。保存实际行ID、图像内容哈希；以PNG文件＋CSV manifest包装训练和无标签预测包，走现有image_path解码而非仅NPZ适配。固定64×64 L，stretch_bilinear。以身份0..39作闭集类别；这不是识别未见人物、真实监控或自然场景泛化。

正式本地动态入口一条路径，最多11实际fits（含refit）、300秒/worker60秒，无API。开始前保存协议、完整目录、划分与任务契约；不得调参或改下一模型。模型封存后预测外层无标签包，迁移目录新进程CSV一致，独立进程评分。单一开发路径只补输入/执行/交付覆盖，不比较架构准确率。不要求成绩过线，失败保留且停止，不自动重跑。

## H1：PAWS-Wiki同义判定，仅登记不跑模型

源仓库 https://github.com/google-research-datasets/paws ，原始LICENSE允许数据自由使用并鼓励署名Google；不误写成CC授权。只用Wiki Labeled Final，下载 https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz 。标签1=同义、0=不同义；交换两文本保持标签。只读取train.tsv用于准备，官方dev/test不开；固定seed61401以规范化文本组成的连通组抽取约2000行（不拆组，可略多于目标），确保两类和可划分性。登记原文件/选定行/标签分布与两个训练内划分61411/61421。保存原始训练TSV与schema草案，无模型拟合、无性能读取。A适配器/B合法对称方法仍需实现和工程验收；数据准备不能冒充H1模型可跑。

## H2：Rice Leaf Diseases彩色照片，仅登记不跑模型

源 https://archive.ics.uci.edu/dataset/486/rice%2Bleaf%2Bdiseases ，CC BY4.0；署名Shah/Prajapati/Dabhi，DOI10.24432/C5R013。下载 https://archive.ics.uci.edu/static/public/486/rice+leaf+diseases.zip 。预计120张、3类；每张原图/解码内容摘要登记，图像固定32×32 RGB、确定性bilinear。不按分类效果挑图；完整数据登记两个61411/61421划分及实际行号，检查重复图跨分区。不开模型/不预测/不看性能，待部署策略冻结后H2才评分。

仅以上来源用于本批，下载失败停止对应准备、保留原错误，不自动轮换镜像或下载其他任务。普通数据HTTP每请求60秒、无自动重试，不涉及学校LLM。数据与照片仅忽略的runs目录保留，Git仅记录许可、来源、摘要和协议，不提交原图/模型。

## 预算与产物

准备0fit，照片开发路径最多11fits/0API，预计1–4分钟训练；下载时间按网络另记。runner scripts/prepare_a13_photo_and_holdouts.py 与scripts/run_a13_photo_pilot.py。protocol/registration/results/completion均保留；H未运行成绩为null。不能将PAWS/Rice预登记看作独立测试结论。

首次受限进程在figshare域名解析阶段失败（runs/a13_data_registration，未下载/0fit）。按环境网络权限限制诊断后，改为获准的普通网络进程在全新a13_data_registration_network目录执行；不更换数据源、不改系统代理/防火墙，不是付费API重试。失败记录保留。

正常网络进程已完成photo_dev，H1的Google存储返回403后停止。保留其有效照片与失败记录，runner增加按dataset独立准备；H1使用已配置的127.0.0.1:7897代理作一次受控网络复核（仅该数据HTTP请求，不修改系统配置/学校API），H2独立准备，避免已成功照片重复下载。CSV适配器将照片类ID读为字符串，外层scorer仅将同一数字ID转为字符串对齐，不重抽样、不合并类别。

H1经代理仍403，不再请求原始存储。已只读核验Google Research发布者的Hugging Face仓库文件清单；明确新增来源 https://huggingface.co/datasets/google-research-datasets/paws/resolve/main/labeled_final/train-00000-of-00001.parquet （8433884字节、SHA256 8dc9ad3e5f30ad9a86b290fe236d528ef23a5751fec9a35d99cbacf68ba277cf）。只读此train文件、官方validation/test不下载；与原tar未做字节等价声明。Parquet解码如需pyarrow，仅安装在忽略的runs/a13_preparation_dependencies侧目录，不改主venv或生产依赖，记录实际版本。H1转换与2000行连通组抽样/划分协议不变。这是来源审阅后的显式修订，不是自动轮换任意镜像。

H2直连持续传输但速率不足（数分钟仅约6MB/36.7MB），人为停止本次下载，保留部分文件；使用既有代理在新目录a13_H2_proxy请求同一UCI源，不改数据/抽样规则。旧目录若因进程被终止没有completion，归档时标记interrupted_slow_download，0fit/0API。

数据准备pyarrow下载亦因直连缓慢而手工停止；已从包元数据确认26.0.1版本，改为经现有代理安装到独立a13_preparation_dependencies_proxy，不改主环境。该路径仅是数据解码准备依赖，不加入Agent执行候选或课程模型依赖。
