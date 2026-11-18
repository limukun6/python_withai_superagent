[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$ApiKeyFile,

    [string]$Task1 = "",
    [string]$Task2 = "",
    [string]$ApiUrl = "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions",
    [ValidateSet("DeepSeek-V4-Pro", "DeepSeek-V4.1-Flash")]
    [string]$Model = "DeepSeek-V4-Pro",
    [ValidateRange(0.01, 20.0)]
    [double]$BatchBudgetRmb = 8.0,
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$dataRoot = Join-Path $repoRoot "data"
if (-not $Task1) { $Task1 = Join-Path $dataRoot "released_task1.zip" }
if (-not $Task2) { $Task2 = Join-Path $dataRoot "released_task2.zip" }

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python virtual environment not found: $python. Run setup first."
}
foreach ($task in @($Task1, $Task2)) {
    if (-not (Test-Path -LiteralPath $task -PathType Leaf)) {
        throw "Task package not found: $task. Copy it to the ASCII-named data folder or pass -Task1/-Task2."
    }
}
if ($ValidateOnly) {
    Write-Host "Preflight passed. All paths are ASCII-safe and required inputs exist." -ForegroundColor Green
    exit 0
}

# P0 is deliberately small and predeclared: 2 tasks x one seed x D/O/F/R.
# It is a feasibility pilot, not an accuracy conclusion. Do not add seeds after
# looking at results; the formal E1 protocol must be frozen separately.
$outputRoot = Join-Path $repoRoot "runs\v2_pilot"
$arguments = @(
    "-m", "superagent.benchmark", $Task1, $Task2,
    "--strategies", "d", "o", "f", "r",
    "--seeds", "101",
    "--max-trials", "6",
    "--time-limit", "3600",
    "--trial-timeout", "180",
    "--confirmation",
    "--stop-on-unknown-usage",
    "--output-dir", $outputRoot,
    "--api-url", $ApiUrl,
    "--api-key-file", $ApiKeyFile,
    "--model", $Model,
    "--batch-budget-rmb", $BatchBudgetRmb.ToString([Globalization.CultureInfo]::InvariantCulture)
)

Push-Location $repoRoot
try {
    Write-Host "P0 feasibility pilot: 2 tasks x seed 101 x d/o/f/r; 6 training starts per run." -ForegroundColor Cyan
    Write-Host "Price data are provisional platform estimates; no formal cost conclusion is produced." -ForegroundColor Yellow
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark exited with code $LASTEXITCODE. Preserve the output directory for diagnosis."
    }
} finally {
    Pop-Location
}
