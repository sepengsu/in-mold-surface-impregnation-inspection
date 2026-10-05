[CmdletBinding()]
param(
    [string]$Checkpoint = '',
    [string]$Data = '',
    [string]$Images = '',
    [string]$Output = '',
    [ValidateSet('test', 'val', 'train', 'all')][string]$Split = 'test',
    [ValidateSet('all', '000', '180')][string]$Position = 'all',
    [ValidateRange(1, 1024)][int]$BatchSize = 16,
    [ValidateRange(0.000001, 0.999999)][double]$Threshold = 0.5,
    [ValidateRange(0, 10000000)][int]$Limit = 0,
    [ValidateSet('cuda', 'cpu', 'auto')][string]$Device = 'cuda',
    [switch]$NoAmp,
    [switch]$CentralThree,
    [switch]$Open
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $project '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Project Python missing. Install training dependencies using src/training/run_train.ps1 -InstallDependencies -PrepareOnly.'
}
if ($Data -and $Images) { throw 'Use either -Data (paired tiles) or -Images (unlabeled tiles).' }
if ($Images -and $Position -ne 'all') { throw '-Position requires a paired tile manifest.' }
if ($Images -and $CentralThree) { throw '-CentralThree requires a paired ROI tile manifest.' }
if (-not $Output) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $Output = Join-Path $project "outputs\inference\prediction_$stamp"
}
$Output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
if (Test-Path -LiteralPath $Output) { throw "Use a new output directory: $Output" }
$predictArgs = @('-u', '-m', 'src.inference.predict', '--output', $Output, '--split', $Split,
    '--position', $Position, '--batch-size', $BatchSize.ToString(), '--device', $Device, '--limit', $Limit.ToString())
foreach ($entry in @(@('--checkpoint', $Checkpoint), @('--data', $Data), @('--images', $Images))) {
    if ($entry[1]) {
        $resolvedInput = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($entry[1])
        if (-not (Test-Path -LiteralPath $resolvedInput)) { throw "Input not found: $resolvedInput" }
        $predictArgs += @($entry[0], $resolvedInput)
    }
}
if ($PSBoundParameters.ContainsKey('Threshold')) {
    $predictArgs += @('--threshold', $Threshold.ToString([System.Globalization.CultureInfo]::InvariantCulture))
}
if ($NoAmp) { $predictArgs += '--no-amp' }
if ($CentralThree) { $predictArgs += '--central-three' }
Push-Location $project
try {
    & $python @predictArgs
    if ($LASTEXITCODE -ne 0) { throw "Inference failed: $LASTEXITCODE. Check the input and checkpoint paths." }
    $gallery = Join-Path $Output 'index.html'
    Write-Host "Open this gallery in your browser: $gallery"
    if ($Open) { Invoke-Item -LiteralPath $gallery }
}
finally {
    Pop-Location
}
