param([string]$Output='results/baseline',[string]$Python='')
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
$env:OPENBLAS_NUM_THREADS='4'
if (-not $Python) {
    $bundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    if (Test-Path -LiteralPath $bundledPython) { $Python=$bundledPython } else { $Python='python' }
}
& $Python -X utf8 run.py --output $Output
if ($LASTEXITCODE -ne 0) { throw 'Run failed' }
