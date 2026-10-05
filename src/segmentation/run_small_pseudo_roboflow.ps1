[CmdletBinding()]
param(
    [string]$Source = '',
    [string]$OutputRoot = '',
    [string]$PseudoRoot = '',
    [string]$RoiReport = '',
    [string]$RoiOverrides = '',
    [string]$BoundaryConfig = '',
    [ValidateRange(1, 1000000)][int]$ExpectedCount = 87,
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
    [string]$Workspace = '',
    [string]$Project = '',
    [string]$BatchName = '',
    [switch]$SaveOverlays,
    [switch]$Upload
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
if (-not $Source) { $Source = Join-Path $projectRoot 'data\0918' }
if (-not $OutputRoot) { $OutputRoot = Join-Path $projectRoot "data\roboflow_upload\small_pseudo_$stamp" }
if (-not $BoundaryConfig) { $BoundaryConfig = Join-Path $projectRoot 'config\roi_gabor_boundary.yaml' }
if (-not $BatchName) { $BatchName = "small-pseudo-10000us-$((Get-Date).ToString('yyyyMMdd'))" }

$Source = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Source)
$OutputRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputRoot)
$BoundaryConfig = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($BoundaryConfig)
if ($PseudoRoot) { $PseudoRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($PseudoRoot) }
if ($RoiReport) { $RoiReport = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RoiReport) }
if ($RoiOverrides) { $RoiOverrides = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RoiOverrides) }

if (-not (Test-Path -LiteralPath $Source -PathType Container)) { throw "Source not found: $Source" }
if ($PseudoRoot -and $RoiReport) { throw 'Use either PseudoRoot or RoiReport, not both.' }
if ($PseudoRoot -and -not (Test-Path -LiteralPath $PseudoRoot -PathType Container)) {
    throw "PseudoRoot not found: $PseudoRoot"
}
if ($RoiReport -and -not (Test-Path -LiteralPath $RoiReport -PathType Leaf)) {
    throw "RoiReport not found: $RoiReport"
}
if (Test-Path -LiteralPath $OutputRoot) { throw "Use a new output directory: $OutputRoot" }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv was not found on PATH.' }
if ($Upload) {
    if (-not $Workspace) { throw 'Workspace is required with -Upload.' }
    if (-not $Project) { throw 'Project is required with -Upload.' }
    if (-not $env:ROBOFLOW_API_KEY) { throw 'Set ROBOFLOW_API_KEY before using -Upload.' }
}

Push-Location $projectRoot
try {
    if (-not $PseudoRoot) {
        if (-not $RoiReport) {
            $pipelineArgs = @(
                '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'run_roi_pseudo_labels.ps1'),
                '-Source', $Source, '-OutputRoot', $OutputRoot, '-BoundaryConfig', $BoundaryConfig,
                '-LabelExposure', '10000us', '-PreserveSmall', '-NoOverlays',
                '-CandidateMethod', $CandidateMethod,
                '-HysteresisLowRatio', $HysteresisLowRatio.ToString([System.Globalization.CultureInfo]::InvariantCulture),
                '-HysteresisCloseKernel', $HysteresisCloseKernel.ToString(),
                '-ConsensusExposures', $ConsensusExposures, '-ConsensusMin', $ConsensusMin.ToString(),
                '-IlluminationSigma', $IlluminationSigma.ToString([System.Globalization.CultureInfo]::InvariantCulture),
                '-SeedZ', $SeedZ.ToString([System.Globalization.CultureInfo]::InvariantCulture),
                '-SupportZ', $SupportZ.ToString([System.Globalization.CultureInfo]::InvariantCulture),
                '-GrowthSteps', $GrowthSteps.ToString(), '-BoundaryGuard', $BoundaryGuard.ToString()
            )
            if ($RoiOverrides) { $pipelineArgs += @('-RoiOverrides', $RoiOverrides) }
            if ($SaveOverlays) {
                $pipelineArgs = $pipelineArgs | Where-Object { $_ -ne '-NoOverlays' }
            }
            & powershell @pipelineArgs
            if ($LASTEXITCODE -ne 0) { throw "ROI/pseudo pipeline failed with exit code $LASTEXITCODE." }
            $PseudoRoot = Join-Path $OutputRoot 'pseudo'
        }
        else {
            $PseudoRoot = Join-Path $OutputRoot 'pseudo'
            $pseudoArgs = @(
                'run', 'python', '-m', 'src.segmentation.roi_pseudo_labels',
                '--source', $Source, '--roi-report', $RoiReport, '--output', $PseudoRoot,
                '--mode', 'gabor_boundary', '--exposure', '10000us',
                '--gaussian-kernel', '101', '--gaussian-sigma', '10',
                '--candidate-method', $CandidateMethod,
                '--hysteresis-low-ratio', $HysteresisLowRatio.ToString([System.Globalization.CultureInfo]::InvariantCulture),
                '--hysteresis-close-kernel', $HysteresisCloseKernel.ToString(),
                '--consensus-exposures', $ConsensusExposures, '--consensus-min', $ConsensusMin.ToString(),
                '--illumination-sigma', $IlluminationSigma.ToString([System.Globalization.CultureInfo]::InvariantCulture),
                '--seed-z', $SeedZ.ToString([System.Globalization.CultureInfo]::InvariantCulture),
                '--support-z', $SupportZ.ToString([System.Globalization.CultureInfo]::InvariantCulture),
                '--growth-steps', $GrowthSteps.ToString(), '--boundary-guard', $BoundaryGuard.ToString(),
                '--min-contour-area', '0', '--contour-approximation', '0.002',
                '--review-policy', 'quarantine', '--preserve-small'
            )
            if ($SaveOverlays) { $pseudoArgs += '--save-overlays' }
            & uv @pseudoArgs
            if ($LASTEXITCODE -ne 0) { throw "Pseudo labeling failed with exit code $LASTEXITCODE." }
        }
    }

    $dataset = Join-Path $OutputRoot 'roboflow_dataset'
    $zip = Join-Path $OutputRoot 'roboflow_dataset.zip'
    $packageArgs = @(
        'run', 'python', '-m', 'src.segmentation.package_pseudo_roboflow',
        '--source', $Source, '--pseudo-root', $PseudoRoot, '--output', $dataset, '--zip', $zip,
        '--exposure', '10000us', '--expected-count', $ExpectedCount.ToString(),
        '--image-source', 'masked-roi', '--include-review', '--require-preserve-small'
    )
    if ($RoiReport) { $packageArgs += @('--roi-report', $RoiReport) }
    & uv @packageArgs
    if ($LASTEXITCODE -ne 0) { throw "Roboflow packaging failed with exit code $LASTEXITCODE." }

    $uploadArgs = @(
        'run', 'python', '-m', 'src.segmentation.upload_roboflow_from_yaml',
        '--dataset', $dataset, '--expected-count', $ExpectedCount.ToString(),
        '--batch-name', $BatchName, '--zip-upload'
    )
    if ($Workspace) { $uploadArgs += @('--workspace', $Workspace) }
    if ($Project) { $uploadArgs += @('--project', $Project) }
    & uv @uploadArgs '--dry-run'
    if ($LASTEXITCODE -ne 0) { throw "Upload validation failed with exit code $LASTEXITCODE." }
    if ($Upload) {
        & uv @uploadArgs
        if ($LASTEXITCODE -ne 0) { throw "Roboflow upload failed with exit code $LASTEXITCODE." }
        Write-Host 'Roboflow upload completed.'
    }
    else {
        Write-Host 'Upload was not requested. Add -Upload after checking the ZIP and source_mapping.csv.'
    }
    Write-Host "Dataset: $dataset"
    Write-Host "ZIP: $zip"
}
finally {
    Pop-Location
}
