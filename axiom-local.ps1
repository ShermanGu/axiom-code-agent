param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AxiomArgs
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Candidates = @(
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    (Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe")
)
$Python = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

if (-not $Python) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) {
        $Python = $PythonCommand.Source
    }
}
if (-not $Python) {
    throw "Python 3.11+ was not found. Install Python, then run: py -3.12 -m venv .venv"
}

$PreviousPythonPath = $env:PYTHONPATH
$SourcePath = Join-Path $ProjectRoot "src"
$env:PYTHONPATH = if ($PreviousPythonPath) {
    "$SourcePath;$PreviousPythonPath"
} else {
    $SourcePath
}

Push-Location $ProjectRoot
try {
    & $Python -m axiom_agent.cli @AxiomArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
    $env:PYTHONPATH = $PreviousPythonPath
}
