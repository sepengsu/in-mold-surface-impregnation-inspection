[CmdletBinding()]
param(
    [string]$Source = '',
    [string]$OutputRoot = '',
    [string]$RoiOverrides = '',
    [string]$BoundaryConfig = '',
    [string]$LabelExposure = '10000us',
    [ValidateRange(0, 1000)][int]$FusionExposures = 0,
    [ValidateRange(0, 1000000)][int]$Limit = 0,
    [ValidateRange(0, 100000000)][double]$MinContourArea = 50,
    [ValidateSet('gaussian_otsu', 'gaussian_hysteresis', 'periodic_consensus')][string]$CandidateMethod = 'gaussian_otsu',
    [ValidateRange(0.01, 1.0)][double]$HysteresisLowRatio = 0.4,
    [ValidateRange(0, 101)][int]$HysteresisCloseKernel = 3,
    [string]$ConsensusExposures = '8000us,10000us,12000us',
    [ValidateRange(1, 100)][int]$ConsensusMin = 2,
    [ValidateRange(0.1, 1000)][double]$IlluminationSigma = 80,
    [ValidateRange(0.1, 100)][double]$SeedZ = 25,
    [ValidateRange(0, 100)][double]$SupportZ = 12,
    [ValidateRange(0, 1000)][int]$GrowthSteps = 2,
    [ValidateRange(0, 1000)][int]$BoundaryGuard = 64,
    [switch]$PreserveSmall,
    [switch]$SaveDebug,
    [switch]$NoOverlays
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Source) { $Source = Join-Path $project 'data\0918' }
if (-not $BoundaryConfig) { $BoundaryConfig = Join-Path $project 'config\roi_gabor_boundary.yaml' }
if (-not $OutputRoot) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $OutputRoot = Join-Path $project "data\roied\roi_pseudo_$stamp"
}
$Source = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Source)
$OutputRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputRoot)
$BoundaryConfig = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($BoundaryConfig)
if ($RoiOverrides) { $RoiOverrides = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RoiOverrides) }
if (-not (Test-Path -LiteralPath $Source -PathType Container)) { throw "Source not found: $Source" }
if (-not (Test-Path -LiteralPath $BoundaryConfig -PathType Leaf)) { throw "Config not found: $BoundaryConfig" }
if ($RoiOverrides -and -not (Test-Path -LiteralPath $RoiOverrides -PathType Leaf)) {
    throw "Overrides not found: $RoiOverrides"
}
if ($LabelExposure -ne 'all' -and $LabelExposure -notmatch '^\d+us$') {
    throw 'LabelExposure must be an exposure folder (such as 10000us) or all.'
}
$separator = [System.IO.Path]::DirectorySeparatorChar
$sourcePrefix = $Source.TrimEnd($separator) + $separator
$outputPrefix = $OutputRoot.TrimEnd($separator) + $separator
if ($sourcePrefix.StartsWith($outputPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
    $outputPrefix.StartsWith($sourcePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Source and OutputRoot must be separate directory trees, not nested.'
}
if (Test-Path -LiteralPath $OutputRoot) { throw "Use a new output directory: $OutputRoot" }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv was not found on PATH.' }

$roiOutput = Join-Path $OutputRoot 'roi'
$pseudoOutput = Join-Path $OutputRoot 'pseudo'
$roiArgs = @('run', 'python', '-m', 'src.roi.auto_roi_local_std',
    '--source', $Source, '--output', $roiOutput, '--reference-exposure', '10000us',
    '--mode', 'gabor_boundary', '--boundary-config', $BoundaryConfig, '--image-output', 'masked',
    '--fusion-exposures', $FusionExposures.ToString())
if ($RoiOverrides) { $roiArgs += @('--roi-overrides', $RoiOverrides) }
if ($SaveDebug) { $roiArgs += '--save-debug' }
if ($Limit -gt 0) { $roiArgs += @('--limit', $Limit.ToString()) }
$pseudoArgs = @('run', 'python', '-m', 'src.segmentation.roi_pseudo_labels',
    '--source', $Source, '--roi-report', (Join-Path $roiOutput 'roi_results.csv'),
    '--output', $pseudoOutput, '--mode', 'gabor_boundary', '--exposure', $LabelExposure,
    '--gaussian-kernel', '101', '--gaussian-sigma', '10',
    '--candidate-method', $CandidateMethod,
    '--hysteresis-low-ratio', $HysteresisLowRatio.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--hysteresis-close-kernel', $HysteresisCloseKernel.ToString(),
    '--consensus-exposures', $ConsensusExposures, '--consensus-min', $ConsensusMin.ToString(),
    '--illumination-sigma', $IlluminationSigma.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--seed-z', $SeedZ.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--support-z', $SupportZ.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--growth-steps', $GrowthSteps.ToString(), '--boundary-guard', $BoundaryGuard.ToString(),
    '--min-contour-area', $MinContourArea.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--contour-approximation', '0.002', '--review-policy', 'quarantine')
if ($PreserveSmall) {
    $pseudoArgs += '--preserve-small'
    $minAreaIndex = [Array]::IndexOf($pseudoArgs, '--min-contour-area') + 1
    $pseudoArgs[$minAreaIndex] = '0'
}
if (-not $NoOverlays) { $pseudoArgs += '--save-overlays' }

Push-Location $project
try {
    & uv @roiArgs
    if ($LASTEXITCODE -ne 0) { throw "ROI extraction failed with exit code $LASTEXITCODE; pseudo labels were not started." }
    & uv @pseudoArgs
    if ($LASTEXITCODE -ne 0) { throw "Pseudo labeling failed with exit code $LASTEXITCODE. Check pseudo_results.csv." }
    Write-Host "ROI output: $roiOutput"
    Write-Host "Pseudo output: $pseudoOutput"
    Write-Host 'All labels, including empty labels, require human review.'
}
finally {
    Pop-Location
}
