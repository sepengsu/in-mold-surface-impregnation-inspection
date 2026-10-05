param(
    [string]$Checkpoint = '',
    [string]$Output = '',
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Push-Location $projectRoot
try {
    $pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
    $arguments = @('-X', 'utf8', '-m', 'src.inference.report_nine_points', '--device', $Device)
    if ($Checkpoint) { $arguments += @('--checkpoint', $Checkpoint) }
    if ($Output) { $arguments += @('--output', $Output) }
    & $pythonPath @arguments
    if ($LASTEXITCODE -ne 0) { throw "Nine-point inference failed: $LASTEXITCODE" }
} finally {
    Pop-Location
}
