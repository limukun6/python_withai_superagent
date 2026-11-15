# Agent architecture experiments

## Research questions and fixed protocol

Compare random search, one-shot planning, feedback planning and planning with
a reviewer. All policies share the same 14 configurations per task. Default
budget is three candidate attempts (failures count), 120 seconds per training
process, and a 600-second search deadline. Planner calls have a bounded HTTP
timeout; one in-flight call can overrun the search deadline. Final export and
holdout scoring occur after search and are included in total elapsed time.

One-shot gets the profile once and submits its whole plan. Feedback gets prior
training/search metrics before each choice. Reviewed uses the same model in
two calls: proposal, then critique/final decision. It does not yet have a third
revision call or generated-code capability. LLM failure activates random
recovery; recovered runs are labeled invalid for pure-policy comparisons.

Each repeat uses a grouped, stratified 80/20 split of the supplied training
set. Identical feature rows or unordered text pairs remain in one fold.
The grouping does not detect all semantic duplicates or shared entities.
All policies receive the same split for the same seed. Classifier seeds stay
42, so repeated seeds measure split/search variability rather than independent
training seeds. Official validation is scored once after selection and never
included in planner context. No refit is done in this protocol.

The supplied official validation sets were already used during early project
development, so they are not pristine held-out benchmarks. Never use test data
to choose an architecture. Subsequent transfer evaluation needs new datasets.

## Commands

After disconnecting VPN, run the PowerShell wrapper from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_agent_architecture_experiment.ps1 -ApiKeyFile "D:\\学校\\python with ai\\api\\api_key.txt" -Mode Pilot
```

`Pilot` uses seed 11 and caps the shared batch estimate at RMB 2. Check that
the run has no `planner_fallback`, `unresolved_call` or `missing_usage` event
before starting the main comparison. Then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_agent_architecture_experiment.ps1 -ApiKeyFile "D:\\学校\\python with ai\\api\\api_key.txt" -Mode Main -BatchBudgetRmb 10
```

`Main` fixes seeds 11, 22 and 33. The shared budget setting caps this entire
benchmark batch; it does not replace the course's per-task RMB 20 limit.

Equivalent direct commands, useful when changing an explicit task path, are:

```powershell
.\.venv\Scripts\python.exe -m superagent.benchmark "TASK1.zip" "TASK2.zip" --strategies random --seeds 11 22 33

.\.venv\Scripts\python.exe -m superagent.benchmark "TASK1.zip" "TASK2.zip" --strategies random oneshot feedback reviewed --seeds 11 --api-key-file "PATH_TO_KEY" --model DeepSeek-V4-Pro --batch-budget-rmb 2
```

Start with one seed to verify feasibility. Then predeclare seeds 11, 22, 33
for the main 4 x 2 x 3 comparison. Do not keep rerunning until a preferred
architecture wins. Small seed counts do not establish statistical significance.

The benchmark shares one estimated-cost ceiling across all tasks, policies
and seeds in a batch. It uses the supplied platform screenshot prices, not
verified course-standard historical prices. Preserve raw usage for repricing.
Unknown usage and ambiguous transport errors open a circuit and prevent further
batch API calls; they are never counted as zero-cost successes. There is no
automatic billing retry. No API credentials are stored in run artifacts.

## Artifacts and experimental conclusions

`protocol.json` records configuration and a source hash. Every run stores inner
split indices, candidate metrics, hypotheses, raw planning requests/responses,
usage, the frozen selection, predictions and summary. `summaries.json` retains
all runs; `comparison.md` reports valid-policy accuracy means/SD and all-run
time/token totals. The per-round best score in `trials.json` supports search
curves. Invalid/recovered runs must not silently enter architecture averages.

Interpret outcomes in terms of: final accuracy, time to a predeclared target,
search trajectory, successful delivery, failed trials, fallback rate, and
token expenditure. A one-seed pilot is not a design-selection result. All
policies are compared under equal trial limits; reviewer benefits must also
be checked under equal token limits before attributing gains to architecture.

After the main comparison, ablate feedback or review using the same source
version and configuration flags. Freeze the selected design, then evaluate
transfer on untouched tasks and conduct final test evaluation separately.
All formal report runs must be regenerated from the final submitted version.

This framework studies constrained experiment selection. It is not yet a
general agent that writes algorithms or understands arbitrary task formats.
