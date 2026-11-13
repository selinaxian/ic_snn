param(
    [string]$Python = "",
    [string]$Output = "results/new_first_fold",
    [switch]$AllFolds
)
$ErrorActionPreference = "Stop"
if (-not $Python) {
    $bundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    if (Test-Path -LiteralPath $bundledPython) { $Python = $bundledPython }
    else { $Python = 'python' }
}
Push-Location $PSScriptRoot
try {
    $runArguments = @('-X', 'utf8', 'run.py', '--output', $Output)
    if (-not $AllFolds) { $runArguments += @('--folds', '1') }
    & $Python @runArguments
    if ($LASTEXITCODE -ne 0) { throw "SNN run failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
