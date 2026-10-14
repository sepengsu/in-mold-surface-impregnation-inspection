[CmdletBinding()]
param(
    [string]$Output = '',
    [string]$OofSource = '',
    [string]$ReviewedManifest = '',
    [string]$TestReviewManifest = '',
    [ValidateSet('A','B','C','D')][string[]]$Arms = @('A','B','C','D'),
    [ValidateSet('prepare','validate-oof','oof','issues','auto','train','compare','all')][string]$Stage = 'all',
    [ValidateRange(2,20)][int]$Folds = 3,
    [ValidateRange(1,100000)][int]$OOFEpochs = 150,
    [ValidateRange(1,100000)][int]$Epochs = 500,
    [ValidateRange(1,100000)][int]$Patience = 20,
    [ValidateRange(1,1024)][int]$BatchSize = 16,
    [ValidateRange(8,256)][int]$BaseChannels = 32,
    [ValidateRange(1,1024)][int]$GradientAccumulationSteps = 1,
    [switch]$ActivationCheckpointing,
    [ValidateRange(0,1048576)][int]$VramBudgetMiB = 0,
    [ValidateRange(0,1048576)][int]$MinFreeVramMiB = 0,
    [ValidateSet('none','basic','strong')][string]$Augmentation = 'basic',
    [ValidateRange(0,2147483647)][int]$Seed = 42,
    [ValidateSet(0)][int]$Workers = 0,
    [ValidateRange(1,2147483647)][int]$PixelChunkSize = 262144,
    [ValidateRange(1,2147483647)][int]$SortChunkSize = 262144,
    [ValidateRange(128,1048576)][int]$MemoryBudgetMiB = 4096,
    [ValidateRange(0,1048576)][int]$MinFreeRamMiB = 1024,
    [ValidateSet('cuda','cpu','auto')][string]$Device = 'cuda',
    [switch]$InstallDependencies,
    [switch]$Resume,
    [switch]$CentralOnly,
    [switch]$SmokeTest,
    [switch]$NoAmp
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$data = Join-Path $project 'data\changed_v2'
$splits = Join-Path $data 'splits.json'
foreach ($required in @((Join-Path $data 'manifest.json'), $splits)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing changed_v2 input: $required" }
}
if (Test-Path -LiteralPath (Join-Path $data 'RUNNING.json')) { throw 'changed_v2 export is incomplete.' }
if ($Resume -and -not $Output) { throw 'Resume requires the original Output directory.' }
if (-not $Output) {
    $label = if ($CentralOnly -and $SmokeTest) { 'changed_v2_central_smoke_' }
             elseif ($CentralOnly) { 'changed_v2_central_' }
             elseif ($SmokeTest) { 'changed_v2_smoke_' } else { 'changed_v2_' }
    $Output = Join-Path $project ('analysis\confident\' + $label + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
$forward = @{
    Data = $data; Splits = $splits; Output = $Output; Stage = $Stage
    OofSource = $OofSource; ReviewedManifest = $ReviewedManifest; TestReviewManifest = $TestReviewManifest; Arms = $Arms
    PixelChunkSize = $PixelChunkSize; SortChunkSize = $SortChunkSize; MemoryBudgetMiB = $MemoryBudgetMiB; MinFreeRamMiB = $MinFreeRamMiB
    Folds = $Folds; OOFEpochs = $OOFEpochs; Epochs = $Epochs; Patience = $Patience
    BatchSize = $BatchSize; BaseChannels = $BaseChannels; Augmentation = $Augmentation
    GradientAccumulationSteps = $GradientAccumulationSteps; ActivationCheckpointing = $ActivationCheckpointing
    VramBudgetMiB = $VramBudgetMiB; MinFreeVramMiB = $MinFreeVramMiB
    Seed = $Seed; Workers = $Workers; Device = $Device
    InstallDependencies = $InstallDependencies; Resume = $Resume; CentralOnly = $CentralOnly; SmokeTest = $SmokeTest; NoAmp = $NoAmp
}
Write-Host "Data: $data"
Write-Host "Frozen split: $splits"
Write-Host "Output: $Output"
Write-Host 'CL audits TRAIN only. Existing image orientation and GT masks are preserved.'
if ($CentralOnly) { Write-Host 'CentralOnly: exact P1/P2/P3 tiles in OOF, CL, A/B/C/D training and Test.' }
& (Join-Path $PSScriptRoot 'run_confident.ps1') @forward
