# 第六轮：已有 TabPFN-v2 权重的离线对照

两次树方法之外，检查通用预训练表格模型能否提供互补。使用本机已有 tabpfn 10.2.0 与 29MB 的 v2 权重，固定 model_path，禁止下载和网络。文件 SHA256 必须等于 cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49。官方称此 finetuned 别名与 v2 默认权重内容相同。[官方模型卡](https://huggingface.co/Prior-Labs/TabPFN-v2-clf) · [权重文件与哈希](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt)

零拟合 GPU 探针发现 RTX5070 Laptop 为 sm120，现有 torch 2.13.0+cu126 的内核最高到 sm90，实际分配后求和报 no kernel image。该诊断未训练、未调用模型 API。本轮使用 CPU，不改队友正在使用的环境。

同一 18643 的前两折，在每折拟合侧以固定种子 26471 分层抽取 2048 行；评价折完整使用。词表只从抽出的拟合行建立。两种表示分别为原始类别字段，和额外加入无标签自动发现的完整画像/情境 token。各运行 2 个预处理成员，不自动增加成员；共 4 个 fit API 调用、保守计 8 个分类器成员。推理时分批 128 行，CPU 6 线程。固定 0.5 阈值，不搜索。学习资源上限是本批登记，无课程 API 和语言模型用量。

首次 attempt_01 在任何 fit 之前停止，2048 行子集达不到现有重复结构检测器的支持门槛。原 completion 保留，fit/API 都为 0。修正后 raw 路线无须结构检测；with_context 的无标签结构只从外层拟合侧约 6500 行发现，词表和模型依然只用上述 2048 行，评价侧完全不参与发现。fresh attempt_02 按此明确版本执行，模型配置不变。

重用第二轮同折基线和 token 树的正确数，报告本模型单独得分。由于旧概率未保存，本批不事后伪造融合对照。至少两折更好且平均增益 ≥.5pp 才能考虑下一轮全训练/融合的独立实验。本批只读 C_train/y_train，指定验证与 test 封存。预计 CPU 2–10 分钟。错误保留，停止当前配置链，不自动改数据量或权重继续。

```powershell
D:/codex/python_withai/runs/a79_encoder_assets/venv/Scripts/python.exe -m scripts.run_task1_tabpfn --task D:/codex/python_withai/data/released_task1.zip --output research/task1_tabpfn_01
```
