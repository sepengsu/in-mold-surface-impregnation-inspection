[CmdletBinding()]
param(
    [string]$Data = '',
    [string]$Splits = '',
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
$project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $project '.venv\Scripts\python.exe'
if ($BaseChannels % 8 -ne 0) { throw 'BaseChannels must be divisible by 8.' }
if ($Resume -and -not $Output) { throw 'Resume requires an explicit Output directory.' }
foreach ($name in @('Data','Splits','Output','OofSource','ReviewedManifest','TestReviewManifest')) {
    $value = Get-Variable -Name $name -ValueOnly
    if ($value) { Set-Variable -Name $name -Value $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($value) }
}
Push-Location $project
try {
    if ($InstallDependencies) {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv first: https://docs.astral.sh/uv/' }
        $syncArgs = @('--cache-dir', (Join-Path $project '.cache\uv'), 'sync', '--extra', 'confident', '--inexact', '--frozen')
        # Preserve an existing Python version, even if .python-version differs.
        if (Test-Path -LiteralPath $python -PathType Leaf) { $syncArgs += @('--python', $python) }
        & uv @syncArgs
        if ($LASTEXITCODE -ne 0) { throw 'Confident Learning dependency installation failed.' }
    }
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Run with -InstallDependencies first.' }
    & $python -c "import cleanlab, torch, psutil; print('cleanlab:', cleanlab.__version__, '| torch:', torch.__version__, '| psutil:', psutil.__version__)"
    if ($LASTEXITCODE -ne 0) { throw 'Run with -InstallDependencies first.' }
    $runArgs = @('-X', 'utf8', '-u', '-m', 'src.Namrc.run', '--stage', $Stage,
        '--folds', $Folds.ToString(), '--oof-epochs', $OOFEpochs.ToString(), '--epochs', $Epochs.ToString(),
        '--patience', $Patience.ToString(), '--batch-size', $BatchSize.ToString(), '--base-channels', $BaseChannels.ToString(),
        '--gradient-accumulation-steps', $GradientAccumulationSteps.ToString(),
        '--vram-budget-mib', $VramBudgetMiB.ToString(), '--min-free-vram-mib', $MinFreeVramMiB.ToString(),
        '--augmentation', $Augmentation, '--seed', $Seed.ToString(), '--workers', $Workers.ToString(), '--device', $Device,
        '--pixel-chunk-size', $PixelChunkSize.ToString(), '--sort-chunk-size', $SortChunkSize.ToString(),
        '--memory-budget-mib', $MemoryBudgetMiB.ToString(), '--min-free-ram-mib', $MinFreeRamMiB.ToString())
    $runArgs += '--arms'; $runArgs += $Arms
    if ($Data) { $runArgs += @('--data', $Data) }
    if ($Splits) { $runArgs += @('--splits', $Splits) }
    if ($Output) { $runArgs += @('--output', $Output) }
    if ($OofSource) { $runArgs += @('--oof-source', $OofSource) }
    if ($ReviewedManifest) { $runArgs += @('--reviewed-manifest', $ReviewedManifest) }
    if ($TestReviewManifest) { $runArgs += @('--test-review-manifest', $TestReviewManifest) }
    if ($Resume) { $runArgs += '--resume' }
    if ($CentralOnly) { $runArgs += '--central-only' }
    if ($SmokeTest) { $runArgs += '--smoke-test' }
    if ($NoAmp) { $runArgs += '--no-amp' }
    if ($ActivationCheckpointing) { $runArgs += '--activation-checkpointing' }
    & $python @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Confident Learning stage failed: $LASTEXITCODE" }
}
finally { Pop-Location }
