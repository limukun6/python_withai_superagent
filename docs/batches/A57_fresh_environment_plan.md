# A57：全新依赖环境中的真实预测验收

在安装/执行前登记，0训练/0模型API。只有A50主/后续训练队列全部结束、A54六包导出及隔离源预测通过，且窗口剩≥20分钟时才启动，否则保留待办。不能与训练抢CPU导致worker超时。

在本工作区runs/a57_fresh_environment内新建独立venv，**不使用system-site-packages**，不修改当前项目.venv或系统Python。用当前pip的--python目标解释器功能，在新env按已记录的7个直接依赖精确版本安装，安装源明确PyPI，only-binary、no-cache、retries=0，安装最长10分钟且为汇报保留至少7分钟；安装失败保存原错，不换版本或反复试源。过程配置仅对子进程生效，不改代理/DNS/全局pip设置，不写工作区之外。

安装成功后，以新env的python -I从另一个cwd运行A54六个bundle内predict.py。模型/源/依赖SHA与版本检查由包内entry执行，禁网络/fit，输出CSV必须与A54原已验收输出逐字节相同。使用的都是features.zip，不读held labels；真实API账本不变。记录安装耗时、每包预测结果、0fit/API、是否从原仓库源导入（必须false）。

本批如果成功，才可写“已验证从零安装直接依赖的独立环境预测”；仅A54成功时仍只能说源码隔离、依赖复用。这里的新env同一Windows/Python大版本，不等于跨操作系统或另一台硬件。若公共wheel不可用或网络安装失败，这就是交付限制，应交给队友复现/提供合规wheelhouse，不伪称新环境验收通过。
