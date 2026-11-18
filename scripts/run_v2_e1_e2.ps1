[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$ApiKeyFile,

    [ValidateRange(0.01, 20.0)]
    [double]$BatchBudgetRmb = 2.0,
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$apiKey = (Resolve-Path -LiteralPath $ApiKeyFile).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$task1 = Join-Path $root "data\released_task1.zip"
$task2 = Join-Path $root "data\released_task2.zip"
foreach ($path in @($python, $task1, $task2)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file missing: $path"
    }
}
Write-Host "E1: 2 tasks x 5 seeds (41/52/63/74/85) x d/o/f/r = 40 runs, 6 trials each."
Write-Host "Flash with a shared provisional budget of RMB $BatchBudgetRmb. E2 reuses frozen models offline."
if ($ValidateOnly) { exit 0 }

$stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffffffZ")
$output = Join-Path $root "runs\v2_e1\$stamp"
$budget = $BatchBudgetRmb.ToString([Globalization.CultureInfo]::InvariantCulture)

Push-Location $root
try {
    & $python -m superagent.benchmark $task1 $task2 --strategies d o f r --seeds 41 52 63 74 85 --max-trials 6 --trial-timeout 180 --time-limit 3600 --reserve-confirmation --stop-on-unknown-usage --model DeepSeek-V4.1-Flash --api-key-file $apiKey --batch-budget-rmb $budget --output-dir $output
    $e1Exit = $LASTEXITCODE
    $batch = Get-ChildItem -LiteralPath $output -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($batch -and (Test-Path -LiteralPath (Join-Path $batch.FullName "protocol.json"))) {
        & $python (Join-Path $root "scripts\analyze_agent_architecture.py") $batch.FullName --out (Join-Path $batch.FullName "analysis.md")
        if ($LASTEXITCODE -ne 0) { throw "E1 report generation failed: $($batch.FullName)" }
    }
    if ($e1Exit -ne 0) { throw "E1 stopped (exit $e1Exit). Preserved records under $output" }
    & $python -m superagent.replay_confirmation $batch.FullName
    if ($LASTEXITCODE -ne 0) { throw "Offline E2 replay failed: $($batch.FullName)" }
} finally {
    Pop-Location
}
