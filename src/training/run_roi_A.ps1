[CmdletBinding()]
param(
    [string]$Data = '',
    [string]$Output = '',
    [ValidateRange(1,100000)][int]$Epochs = 500,
    [ValidateRange(1,1024)][int]$BatchSize = 16,
    [ValidateRange(1,100000)][int]$Patience = 20,
    [ValidateSet('cuda','cpu','auto')][string]$Device = 'cuda',
    [switch]$PrepareOnly,
    [switch]$SmokeTest
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $project '.venv\Scripts\python.exe'
if (-not $Data) { $Data = Join-Path $project 'data\raw\rectangle_roi_tiles_256' }
$Data = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Data)
$manifest = Get-Content -LiteralPath (Join-Path $Data 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.summary.settings.alignment -ne 'roi_centerline' -or $manifest.summary.settings.selection -ne 'all') {
    throw 'A ROI run requires the ROI-centerline full interior dataset.'
}
if (-not $Output) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    if ($SmokeTest -or $PrepareOnly) {
        $Output = Join-Path $project "outputs\training\checks\roi_A_$stamp"
    } else {
        $Output = Join-Path $project "outputs\training\unet_roi_A_$stamp"
    }
}
$Output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
Write-Host "A: U-Net 32, basic augmentation, uniform sampling, BCE + Dice"
Write-Host "Dataset: $Data"
Write-Host "Output: $Output"
Write-Host "ROI review flags retained: $($manifest.summary.roi_review_flagged_images) images"
& (Join-Path $PSScriptRoot 'run_train.ps1') -Data $Data -Output $Output `
    -Epochs $Epochs -BatchSize $BatchSize -LearningRate 0.001 -BaseChannels 32 `
    -Augmentation basic -Loss bce_dice -Sampling uniform -Workers 0 `
    -Seed 42 -SplitSeed 42 -Patience $Patience -Position all -Device $Device `
    -PrepareOnly:$PrepareOnly -SmokeTest:$SmokeTest
if (-not $PrepareOnly) {
    Push-Location $project
    try {
        & $python -m src.training.report_impregnation --run $Output
        if ($LASTEXITCODE -ne 0) { throw 'Test impregnation report failed.' }
    } finally { Pop-Location }
}
Write-Host "Saved run: $Output"
