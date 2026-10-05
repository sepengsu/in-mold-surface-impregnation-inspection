[CmdletBinding()]
param(
    [string]$Output = '',
    [string]$PointsOutput = '',
    [string]$GalleryOutput = '',
    [string]$Rectangles = '',
    [switch]$CountOnly
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $project '.venv\Scripts\python.exe'
if (-not $Rectangles) {
    $Rectangles = Join-Path $project 'outputs\measurements\rectangle_roi_20261001\rectangles_reviewed.json'
}
$Rectangles = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Rectangles)
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
if (-not $Output) { $Output = Join-Path $project "data\raw\rectangle_roi_tiles_256_$stamp" }
if (-not $PointsOutput) { $PointsOutput = "$($Output)_points" }
if (-not $GalleryOutput) { $GalleryOutput = Join-Path $project "analysis\p123_rectangle_roi_$stamp" }
$Output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
$PointsOutput = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($PointsOutput)
$GalleryOutput = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($GalleryOutput)
if (-not $CountOnly) {
    foreach ($path in @($Output, $PointsOutput, $GalleryOutput)) {
        if (Test-Path -LiteralPath $path) { throw "Use a new output directory: $path" }
    }
    if ($Output -eq $PointsOutput -or $Output -eq $GalleryOutput -or $PointsOutput -eq $GalleryOutput) {
        throw 'Data, measurement tiles, and gallery require separate directories.'
    }
}
Push-Location $project
try {
    if ($CountOnly) {
        & $python -X utf8 -u -m src.preprocess.rectangle_dataset --stage count --rectangles $Rectangles
        if ($LASTEXITCODE -ne 0) { throw 'Rectangle ROI count failed.' }
    } else {
        & $python -X utf8 -u -m src.preprocess.rectangle_dataset --stage export --rectangles $Rectangles --data $Output --points $PointsOutput
        if ($LASTEXITCODE -ne 0) { throw 'Rectangle ROI dataset export failed.' }
        & $python -X utf8 -u -m src.preprocess.visualize_roi_points --data $PointsOutput --output $GalleryOutput
        if ($LASTEXITCODE -ne 0) { throw 'Measurement gallery generation failed.' }
        Write-Host "Training data: $Output"
        Write-Host "Measurement points: $PointsOutput"
        Write-Host "Gallery: $GalleryOutput\index.html"
        Write-Host "Train using: .\src\training\run_rectangle_A.ps1 -Data `"$Output`""
    }
} finally { Pop-Location }
