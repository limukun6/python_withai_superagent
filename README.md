# General Algorithm SuperAgent

## Current execution core

`python -m superagent run` and `python -m superagent.benchmark` now share the
same audited runtime. It builds a task contract from the supplied instructions,
records the actual preprocessing/model pipeline for every candidate, trains only
on an inner fit split, selects on an inner search split, independently reloads
and accepts the frozen model, then evaluates the supplied validation split.

The controlled comparison entry point supports `random`, `deterministic`,
`oneshot`, `feedback`, `reviewed`, and the structured-claim `scientist`
strategy. The new runtime is deliberately binary-only and supports the released
categorical-table and ordered-headline-pair schemas. It does not claim support
for arbitrary files, multiclass tasks, images, or unrestricted code generation.

See [the comparison protocol](docs/agent_experiments.md), [the evidence-driven
search plan](docs/evidence_driven_agent_design.md), and [the overall architecture
plan](docs/system_agent_architecture.md). `legacy-run` retains the old exhaustive
portfolio baseline for historical artifacts only.

For Windows PowerShell compatibility, the wrapper reads task packages from
`data/released_task1.zip` and `data/released_task2.zip`, and reads the API key
from an ASCII-named `api_key.txt` in the repository root. Both are ignored by
Git. Copy them into place before running the wrapper.

Price note: values transcribed from platform screenshots are provisional
platform estimates. They have not been verified against the course's required
historical official standard prices; formal results require that verification.

Each new run writes a task contract, pipeline registry, split indices, decisions,
trial results, model hash, acceptance report, and usage ledger. A run is marked
complete only after the frozen artifact reloads in a separate process and passes
prediction count, label, batch/single-row, and row-order checks.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Development run

The command below reads **only** `train` and `val`; it does not load test
features or test labels.

```powershell
python -m superagent run "D:\\学校\\python with ai\\Released_Task1-In_Vehicle_Coupon_Recommendation.zip"
python -m superagent run "D:\\学校\\python with ai\\Released_Task2-Headlines_CTR.zip"
```

Each command creates a timestamped directory below `runs/`. The default
`deterministic` strategy makes no API calls. To use an LLM policy, choose a
strategy and supply a key:

```powershell
python -m superagent run "TASK.zip" --strategy scientist `
  --api-key-file ".\api_key.txt" --model DeepSeek-V4-Pro
```

The `scientist` policy must cite exact recorded measurements for a follow-up
hypothesis. A malformed, invented, or unavailable claim triggers a logged
recovery and the run is not counted as a valid pure-policy run.

## Candidate portfolios

Candidate selection depends on detected data schema rather than a task or
dataset name. Categorical tables use a majority baseline, regularised One-hot
logistic regression, ExtraTrees with two leaf sizes, and CatBoost with two
depth/regularisation settings. Pair-text inputs use a majority baseline plus
word and word/character TF-IDF difference classifiers with swap augmentation
and several regularisation strengths.

## Frozen-model prediction

Only after model selection is frozen **and independent acceptance passes**, write
test predictions without opening test labels:

```powershell
python -m superagent predict "D:\\学校\\python with ai\\Released_Task1-In_Vehicle_Coupon_Recommendation.zip" `
  --run-dir runs\\categorical_tabular_YYYYMMDD_HHMMSS `
  --output final_predictions\\task1_predictions.csv
```

The output directory also receives `prediction_manifest.json`, which records
that no test labels were read or evaluated.

## Optional LLM planning

The shared client records actual input/output token counts. If a request fails
before usage is known, it stops further paid calls rather than automatically
retrying a potentially billed request. The screenshot prices below remain
development estimates; formal reports require the course-mandated historical
official price verification.

The current audited price configuration is in `superagent/pricing.py`, based
on the HPC AI Computing Platform table supplied on 2026-11-13:

| Model | Input RMB / 1M Tokens | Output RMB / 1M Tokens | Availability |
| --- | ---: | ---: | --- |
| DeepSeek-V4-Pro | 2.25 | 6.75 | enabled |
| DeepSeek-V4.1-Flash-1017 | 0.50 | 2.00 | enabled |
| Qwen3.8-27B-w8a8 | 0.10 | 0.34 | held until release-date evidence is recorded |

For example:

```powershell
python -m superagent run "TASK.zip" --strategy feedback `
  --api-url "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions" `
  --api-key-file "D:\\path\\to\\api_key.txt" `
  --model DeepSeek-V4.1-Flash-1017
```

## Design boundary

`run` is a development command. It never loads test features or labels. The
separate `predict` command reads test features only after the artifact has
passed acceptance; it has no test-label evaluator.

