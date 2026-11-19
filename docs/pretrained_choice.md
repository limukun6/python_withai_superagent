# Optional pretrained features for text and named categories

The local interleaved agent can spend one search slot on a frozen Qwen representation. At trial five, after four original actions, it reads their predictions on the same inner search rows. With text weights configured, single texts and ordered binary preference pairs qualify if at least two trials completed and every completed candidate misclassified at least max(8, ceil(5% of the search rows)) of the same rows. A separately enabled option covers named categories below. The deterministic policy and its thresholds are developer choices. Its only observations come from the current training partition. Confirmation ranks the contenders across three training splits. The best of the four original models has a guaranteed place among the two finalists.

Use Python 3.11 and the optional dependencies in `requirements-pretrained.txt`. The runtime uses the publisher's tokenizer, architecture configuration and full safetensors weights, pinned by revision, SHA-256 and byte count in `superagent/pretrained_choice.py`. It downloads missing verified assets only after a positive decision. Each forward processes at most the first 64 tokens per text, without a chat template or generated output. The mean of unmasked final-layer hidden states is normalized and combined with word/character TF-IDF. Ordered pairs also use the original style features, left-minus-right representations and swap augmentation. Only the logistic classifier and classical preprocessing learn from the fold's training rows.

For `superagent run`, add `--pretrained-config /absolute/path/config.json --trials 5 --portfolio expanded --policy interleaved_routes`. A configuration holds the absolute local weight folder, existing project budget ledger, stable task scope and incremental allowance. For example:

```json
{"weights":"/path/to/Qwen3-0.6B","ledger":"/path/to/project-ledger.sqlite3","scope":"task2","cap_rmb":0.15}
```

The local forward counts nonpadding input tokens and zero generated tokens. Each actual batch reserves and settles its charge in the existing ledger. Saved vectors can be reused within one run without executing another forward. The gate decision, artifact downloads, hardware, batch calls, padding and cost are recorded beside the model. A relocated winning contextual model needs access to the same pinned assets, an appropriate ledger and a writable run folder. Set `SUPERAGENT_PRETRAINED_RUNTIME` to the relocated `pretrained_runtime.json` and update its local paths; keep the same task scope and cumulative run ID when continuing its accounting. Prediction on new text also consumes the course token budget.

The paired driver is `python -m tools.pretrained_study --help`. It uses only training members of the released ZIPs, preserves the previous reserve, checks group overlap, and freezes every model before outer scoring. Both arms have five search slots, two finalists, three measurements per finalist including the search score, and one refit. Each arm has at most 10 fits, 2,000 seconds of training including model acquisition, 1,500 seconds per individual fit, and ¥0.075 of new model usage. Including the preceding development pilot the combined experiment ceiling is ¥0.15 per released task. Outer scoring is recorded separately and uses the same monetary ceiling. When a previously documented historical deferral is supplied, the study matches its task ID and dataset hash, retains those unknown calls and a ¥3 contingency per such request, and rejects any new unknown or outstanding reservation.

The official release was April 29, 2025. Alibaba Cloud's March 17, 2026 standard API table lists ¥0.30 per million input tokens and ¥1.20 per million non-thinking output tokens. No cache discount or free credit is applied. Course cost and actual local API debit are separate fields. [Qwen release](https://qwenlm.github.io/blog/qwen3/) · [Alibaba Cloud tariff](https://help.aliyun.com/zh/document_detail/3023262.html) · [Publisher weights](https://modelscope.cn/models/Qwen/Qwen3-0.6B)

## Named categories

Set `tabular_weights` to the directory for the official Qwen3-Embedding-0.6B checkpoint. An example configuration for a registered task follows.

```json
{"tabular_weights":"/path/to/Qwen3-Embedding-0.6B","ledger":"/path/to/project-ledger.sqlite3","scope":"task1","cap_rmb":0.02}
```

The policy inspects the fitting rows for columns with 3–128 distinct nonmissing levels, at least two observations per level on average, and word-like text in at least half the distinct names. It ranks eligible columns by cardinality and keeps up to eight. It counts shared mistakes on search rows containing an unseen level or a level at or below its column's median training support and below the maximum. At least four such mistakes and at least 10% of all shared mistakes are required, in addition to the common-error threshold for text. The fifth action also requires a compatible incumbent and sufficient token allowance. Every skip leaves the ordinary fifth experiment available.

The procedure encodes each distinct training name once per run with the frozen publisher checkpoint. It uses the last unmasked token including the tokenizer's end-of-text marker, L2 normalization and a maximum input length of 64 tokens. Up to three principal coordinates per column are fitted on that fold's distinct training levels. The candidate adds those numeric features to the original categorical columns and uses the current incumbent's classifier and hyperparameters. The supported backbones are CatBoost, extra trees, and one-hot logistic regression. The saved classifier contains its lookup tables. Missing or previously unseen levels receive zero added coordinates. Prediction on new rows runs offline. The permuted assignment in the study rotates level-to-vector assignments within each column as a negative control.

Qwen released this embedding model on June 5, 2025. The Shanghai OpenSearch input rate for the exact 0.6B version is RMB 0.005 per thousand tokens in the first 500 pricing units. The ledger counts every actual nonpadding input token, including the end marker. Embedding inference has zero generated tokens. The pre-cutoff indexed table and current table agree on this rate. A dated October 21 invoice or archived daily tariff is not part of the evidence. [Qwen release](https://qwenlm.github.io/blog/qwen3-embedding/) · [Publisher pooling and tokenizer usage](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) · [OpenSearch API model listing](https://help.aliyun.com/zh/open-search/ai-search-open-platform/text-embedding-api-details) · [Earlier Alibaba Cloud tariff page](https://help.aliyun.com/zh/open-search/search-platform/product-overview/billing-methods-and-billing-items) · [Current tariff](https://help.aliyun.com/zh/open-search/ai-search-open-platform/billing-methods-and-billing-items)

The training-only comparison uses three disjoint grouped outer folds after removing the project's reserved partition. Each of the original, semantic, and permuted policies attempts five searches, three measurements for each of the top two, and a full refit. The first search measurement counts toward confirmation, giving ten classifier fits per run. The driver freezes all nine winners before any outer scoring and checks that all three arms saw identical first-four feedback within each fold.

```sh
python -m tools.tabular_encoder_study \
  --task1 "/path/to/Released_Task1-In_Vehicle_Coupon_Recommendation.zip" \
  --output "/path/to/new-study" --ledger "/path/to/project-ledger.sqlite3" \
  --weights "/path/to/Qwen3-Embedding-0.6B" \
  --historical-protocol "/path/to/project-historical-protocol.json"
```

To reproduce the follow-up, run `python -m tools.tabular_encoder_replication` with the same arguments, replacing `--output` with `--first-stage-root`. It checks the original source and archive digests, then scores the other two folds without changing the policy. Both stages draw from a single experiment allowance. After both finish, `python -m tools.tabular_encoder_diagnostics --task1 /path/to/task1.zip --first-stage-root /path/to/new-study` replays the stored search models on the outer rows to compare the three representations at the same backbone and training sample size.
