[CmdletBinding()]
param(
    [string]$Data = '',
    [string]$Output = '',
    [ValidateSet('augmentation', 'improvement')][string]$Suite = 'augmentation',
    [ValidateSet('all', 'bd')][string]$Experiments = 'all',
    [ValidateRange(1, 100000)][int]$Epochs = 100,
    [ValidateRange(1, 1024)][int]$BatchSize = 16,
    [ValidateRange(0.0000001, 1.0)][double]$LearningRate = 0.001,
    [ValidateRange(1, 100000)][int]$Patience = 20,
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [ValidateRange(0, 2147483647)][Nullable[int]]$SplitSeed = $null,
    [ValidateRange(0.000001, 0.999999)][double]$TverskyAlpha = 0.3,
    [ValidateRange(0.000001, 0.999999)][double]$TverskyBeta = 0.7,
    [ValidateRange(0.000001, 1.0)][double]$LargeRegionFraction = 0.05,
    [ValidateRange(1.0, 1000.0)][double]$LargeRegionWeight = 3.0,
    [ValidateRange(0, 64)][int]$Workers = 0,
    [ValidateSet('all', '000', '180')][string]$Position = 'all',
    [ValidateSet('cuda', 'cpu', 'auto')][string]$Device = 'cuda',
    [switch]$InstallDependencies,
    [switch]$SmokeTest,
    [switch]$NoAmp,
    [switch]$Open
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $project '.venv\Scripts\python.exe'
if (-not $Data) { $Data = Join-Path $project 'data\raw\rectangle_roi_tiles_256' }
$Data = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Data)
if (-not (Test-Path -LiteralPath (Join-Path $Data 'manifest.json') -PathType Leaf)) { throw "Tile manifest not found: $Data" }
if (-not $Output) {
    $prefix = if ($Suite -eq 'improvement') { 'comparison_improvement_' } else { 'comparison_' }
    $Output = Join-Path $project ('outputs\training\' + $prefix + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
$Output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
if (Test-Path -LiteralPath $Output) { throw "Use a new output directory: $Output" }
if ($Suite -eq 'improvement' -and $Experiments -ne 'all') { throw 'The improvement suite requires -Experiments all.' }
if ([Math]::Abs($TverskyAlpha + $TverskyBeta - 1.0) -gt 0.000001) { throw 'TverskyAlpha + TverskyBeta must equal 1.' }
Push-Location $project
try {
    if ($InstallDependencies) {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv is required for installation.' }
        $syncArgs = @('--cache-dir', (Join-Path $project '.cache\uv'), 'sync', '--extra', 'training', '--inexact', '--frozen')
        if (Test-Path -LiteralPath $python -PathType Leaf) { $syncArgs += @('--python', $python) }
        & uv @syncArgs
        if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed: $LASTEXITCODE" }
    }
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Run with -InstallDependencies first.' }
    $compareArgs = @('-u', '-m', 'src.training.compare', '--data', $Data, '--output', $Output,
        '--suite', $Suite,
        '--experiments', $Experiments, '--epochs', $Epochs.ToString(), '--batch-size', $BatchSize.ToString(),
        '--learning-rate', $LearningRate.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        '--patience', $Patience.ToString(), '--seed', $Seed.ToString(), '--workers', $Workers.ToString(),
        '--tversky-alpha', $TverskyAlpha.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        '--tversky-beta', $TverskyBeta.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        '--large-region-fraction', $LargeRegionFraction.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        '--large-region-weight', $LargeRegionWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        '--position', $Position, '--device', $Device)
    if ($null -ne $SplitSeed) { $compareArgs += @('--split-seed', $SplitSeed.ToString()) }
    if ($SmokeTest) { $compareArgs += '--smoke-test' }
    if ($NoAmp) { $compareArgs += '--no-amp' }
    & $python @compareArgs
    if ($LASTEXITCODE -ne 0) { throw "Comparison failed: $LASTEXITCODE. Check the per-experiment logs under $Output." }
    Write-Host "Comparison table: $(Join-Path $Output 'comparison.csv')"
    if ($Open) { Invoke-Item -LiteralPath (Join-Path $Output 'test_inference\index.html') }
}
finally { Pop-Location }
