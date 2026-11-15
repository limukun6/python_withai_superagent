[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$ApiKeyFile,

    [ValidateSet("Pilot", "Main")]
    [string]$Mode = "Pilot",

    [string]$Task1 = "D:\学校\python with ai\Released_Task1-In_Vehicle_Coupon_Recommendation.zip",
    [string]$Task2 = "D:\学校\python with ai\Released_Task2-Headlines_CTR.zip",
    [string]$ApiUrl = "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions",
    [ValidateSet("DeepSeek-V4-Pro", "DeepSeek-V4.1-Flash-1017")]
    [string]$Model = "DeepSeek-V4-Pro",
    [ValidateRange(0.01, 20.0)]
    [double]$BatchBudgetRmb = 2.0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python virtual environment not found: $python. Run setup first."
}
foreach ($task in @($Task1, $Task2)) {
    if (-not (Test-Path -LiteralPath $task -PathType Leaf)) {
        throw "Task package not found: $task"
    }
}

if ($Mode -eq "Pilot") {
    $seeds = @("11")
    $budget = [Math]::Min($BatchBudgetRmb, 2.0)
} else {
    # Predeclared seeds: do not rerun additional seeds after seeing results.
    $seeds = @("11", "22", "33")
    $budget = $BatchBudgetRmb
}

$outputRoot = Join-Path $repoRoot "runs\agent_architecture"
$arguments = @(
    "-m", "superagent.benchmark", $Task1, $Task2,
    "--strategies", "random", "oneshot", "feedback", "reviewed",
    "--seeds"
) + $seeds + @(
    "--max-trials", "3",
    "--time-limit", "600",
    "--trial-timeout", "120",
    "--output-dir", $outputRoot,
    "--api-url", $ApiUrl,
    "--api-key-file", $ApiKeyFile,
    "--model", $Model,
    "--batch-budget-rmb", $budget.ToString([Globalization.CultureInfo]::InvariantCulture)
)

Push-Location $repoRoot
try {
    Write-Host "Mode: $Mode" -ForegroundColor Cyan
    Write-Host "Model: $Model | Shared batch ceiling: RMB $budget" -ForegroundColor Cyan
    Write-Host "Seeds: $($seeds -join ', ') | Test labels are not read." -ForegroundColor Cyan
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark exited with code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
