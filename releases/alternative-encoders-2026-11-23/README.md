# Task2 的替代编码器与按反馈选组合

进入本目录即可使用同一套 SuperAgent 模块。原十二层、E5 菜单、BGE 菜单三条标准路径在训练内保留区均为 **1,224/1,693**。后续等权组合在相同训练量下分别为 **MiniLM + E5 1,234/1,693（72.89%）**、**MiniLM + BGE 1,232/1,693（72.77%）**。组合由独立的后续研发路径和历史调度回放验证。23 次新增本地拟合，模型 API 0 次。 [完整结果、决策来源与配对不确定性](docs/batches/B_alternative_encoders_results_1123_zh.md)

快速核对已发布回执可运行 `python3 scripts/verify_b_encoder_study.py`。检查全包源文件与证据的哈希可运行 `python3 -c "import json,hashlib,pathlib; m=json.load(open('SOURCE.json'))['sha256']; assert all(hashlib.sha256(pathlib.Path(k).read_bytes()).hexdigest()==v for k,v in m.items()); print(len(m))"`。[文件清单](SOURCE.json)

## 使用两个可选的正式目录

按已有十二层方案准备 Python 环境、标准六层与十二层 MiniLM 权重及匿名训练包。把两个新模型保存到工作区的 `work/pretrained_weights` 下。以下为 Bash 示例，用自己的工作区和数据路径替换变量。下载工具核对官方版本的大小与内容指纹，训练和预测在本机执行。[原方法和输入重建步骤](docs/batches/B_verified_preference_1123_zh.md) · [E5 模型卡](https://huggingface.co/intfloat/e5-small-v2) · [BGE 模型卡](https://huggingface.co/BAAI/bge-small-en-v1.5)

```bash
study_workspace=/path/to/workspace
training_packet=/path/to/anonymous.zip
python scripts/prepare_b_alternative_encoders.py e5 "$study_workspace/work/pretrained_weights"
python scripts/prepare_b_alternative_encoders.py bge "$study_workspace/work/pretrained_weights"
export SUPERAGENT_MINILM_WEIGHTS="$study_workspace/work/pretrained_weights/all-MiniLM-L6-v2"
export SUPERAGENT_DEEP_MINILM_WEIGHTS="$study_workspace/work/pretrained_weights/all-MiniLM-L12-v2"
export SUPERAGENT_E5_SMALL_WEIGHTS="$study_workspace/work/pretrained_weights/e5-small-v2"
export SUPERAGENT_BGE_SMALL_WEIGHTS="$study_workspace/work/pretrained_weights/bge-small-en-v1.5"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python -m superagent run "$training_packet" --portfolio ab-v1 --scheduler ab-v1 --planner local --preference-catalog deep-e5 --seed 8123 --fit-limit 11 --trial-timeout 300 --time-limit 1200 --run-dir runs/e5
```

BGE 使用 `--preference-catalog deep-bge` 及另一运行目录，每条预算均含最终重训。原始匿名训练、保留区特征、内部评分及分组文件可用 `scripts/rebuild_b_preference_inputs.py` 从课程发布包重建。具有已冻结普通十二层对照运行目录的队友，可以使用 `scripts/probe_b_alternative_encoders.py`、`scripts/run_b_alternative_encoders.py` 及其 `--help` 复现标准双路径。后续的组合、分区复核与调度回放沿用同一批工件。[预登记方案](docs/batches/B_alternative_encoders_protocol_1123_zh.md) · [23 次拟合的来源及已完成回执](research/b_alternative_encoders_1123/)
