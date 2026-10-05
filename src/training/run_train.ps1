[CmdletBinding()]
param(
    [string]$Data = '',
    [string]$Output = '',
    [ValidateRange(1, 100000)][int]$Epochs = 100,
    [ValidateRange(1, 1024)][int]$BatchSize = 16,
    [ValidateRange(0.0000001, 1.0)][double]$LearningRate = 0.001,
    [ValidateRange(8, 256)][int]$BaseChannels = 32,
    [ValidateSet('none', 'basic', 'strong')][string]$Augmentation = 'basic',
    [ValidateSet('bce_dice', 'bce_tversky')][string]$Loss = 'bce_dice',
    [ValidateSet('uniform', 'large_region')][string]$Sampling = 'uniform',
    [ValidateRange(0.000001, 0.999999)][double]$TverskyAlpha = 0.3,
    [ValidateRange(0.000001, 0.999999)][double]$TverskyBeta = 0.7,
    [ValidateRange(0.000001, 1.0)][double]$LargeRegionFraction = 0.05,
    [ValidateRange(1.0, 1000.0)][double]$LargeRegionWeight = 3.0,
    [ValidateRange(0, 64)][int]$Workers = 0,
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [ValidateRange(0, 2147483647)][Nullable[int]]$SplitSeed = $null,
    [ValidateRange(1, 100000)][int]$Patience = 20,
    [ValidateSet('all', '000', '180')][string]$Position = 'all',
    [ValidateSet('cuda', 'cpu', 'auto')][string]$Device = 'cuda',
    [switch]$InstallDependencies,
    [switch]$PrepareOnly,
    [switch]$SmokeTest,
    [switch]$SkipTest,
    [switch]$NoAmp
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $project '.venv\Scripts\python.exe'
if (-not $Data) { $Data = Join-Path $project 'data\raw\rectangle_roi_tiles_256' }
$Data = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Data)
if ($Output) { $Output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output) }
if (-not (Test-Path -LiteralPath (Join-Path $Data 'manifest.json') -PathType Leaf)) {
    throw "Tile manifest not found: $Data"
}
if ($PrepareOnly -and $SmokeTest) { throw 'Choose either PrepareOnly or SmokeTest.' }
if ($BaseChannels % 8 -ne 0) { throw 'BaseChannels must be divisible by 8.' }
if ([Math]::Abs($TverskyAlpha + $TverskyBeta - 1.0) -gt 0.000001) { throw 'TverskyAlpha + TverskyBeta must equal 1.' }
if ($Output -and (Test-Path -LiteralPath $Output)) { throw "Use a new output directory: $Output" }

Push-Location $project
try {
    if ($InstallDependencies) {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv first: https://docs.astral.sh/uv/' }
        $syncArgs = @('--cache-dir', (Join-Path $project '.cache\uv'), 'sync', '--extra', 'training', '--inexact', '--frozen')
        if (Test-Path -LiteralPath $python -PathType Leaf) { $syncArgs += @('--python', $python) }
        & uv @syncArgs
        if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed: $LASTEXITCODE" }
    }
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw 'Project Python is missing. Run this script with -InstallDependencies.'
    }
    if (-not $PrepareOnly) {
        & $python -c "import torch, cv2, numpy; print('PyTorch:', torch.__version__)"
        if ($LASTEXITCODE -ne 0) { throw 'Training dependencies are missing. Run with -InstallDependencies.' }
    }
    $trainArgs = @('-u', '-m', 'src.training.train', '--data', $Data,
        '--epochs', $Epochs.ToString(), '--batch-size', $BatchSize.ToString(),
        '--learning-rate', $LearningRate.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        '--base-channels', $BaseChannels.ToString(), '--augmentation', $Augmentation, '--workers', $Workers.ToString(),
        '--loss', $Loss, '--sampling', $Sampling,
        '--tversky-alpha', $TverskyAlpha.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        '--tversky-beta', $TverskyBeta.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        '--large-region-fraction', $LargeRegionFraction.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        '--large-region-weight', $LargeRegionWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        '--seed', $Seed.ToString(), '--patience', $Patience.ToString(),
        '--position', $Position, '--device', $Device)
    if ($Output) { $trainArgs += @('--output', $Output) }
    if ($null -ne $SplitSeed) { $trainArgs += @('--split-seed', $SplitSeed.ToString()) }
    if ($PrepareOnly) { $trainArgs += '--prepare-only' }
    if ($SmokeTest) { $trainArgs += '--smoke-test' }
    if ($SkipTest) { $trainArgs += '--skip-test' }
    if ($NoAmp) { $trainArgs += '--no-amp' }
    & $python @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE. See result.json in the run directory." }
}
finally {
    Pop-Location
}
