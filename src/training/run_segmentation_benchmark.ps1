[CmdletBinding()]
param(
    [string]$Data = '',
    [string]$Output = '',
    [ValidateSet('models','prepare','repair-resume','memory-check','all')][string]$Mode = 'all',
    [string[]]$Models = @('unet_16','unet_32','unet_64',
        'unetpp_resnet18','unetpp_resnet34','unetpp_resnet50',
        'fpn_resnet18','fpn_resnet34','fpn_resnet50',
        'deeplabv3plus_resnet18','deeplabv3plus_resnet34','deeplabv3plus_resnet50',
        'segformer_b0','segformer_b1','segformer_b2'),
    [int[]]$Seeds = @(42),
    [ValidateRange(1,100000)][int]$Epochs = 500,
    [ValidateRange(1,100000)][int]$Patience = 20,
    [ValidateRange(0.0000001,10)][double]$LearningRate = 0.001,
    [ValidateSet(1,2)][int]$MicroBatchSize = 2,
    [ValidateRange(1,1024)][int]$EffectiveBatchSize = 16,
    [ValidateRange(128,1048576)][int]$MemoryBudgetMiB = 4096,
    [ValidateRange(0,1048576)][int]$MinFreeRamMiB = 2048,
    [ValidateRange(256,1048576)][int]$VramBudgetMiB = 8192,
    [ValidateRange(0,1048576)][int]$MinFreeVramMiB = 1024,
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [switch]$Resume,
    [switch]$SmokeTest,
    [switch]$InstallDependencies,
    [switch]$Open
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $project '.venv\Scripts\python.exe'
if (-not $Data) { $Data = Join-Path $project 'data\changed_v2' }
$Data = (Resolve-Path -LiteralPath $Data).Path
if ($EffectiveBatchSize % $MicroBatchSize) { throw 'EffectiveBatchSize must be divisible by MicroBatchSize.' }
if ($Resume -and -not $Output) { throw 'Resume requires the original Output directory.' }
if (-not $Models.Count -or -not $Seeds.Count) { throw 'Models and Seeds cannot be empty.' }
if (-not $Output) {
    $Output = Join-Path $project ('outputs\training\segmentation_benchmark_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
$Output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
$culture = [Globalization.CultureInfo]::InvariantCulture
$runArgs = @('-X','utf8','-u','-m','src.training.benchmark','--data',$Data,'--output',$Output,'--mode',$Mode,
    '--epochs',$Epochs.ToString(),'--patience',$Patience.ToString(),'--learning-rate',$LearningRate.ToString($culture),
    '--micro-batch-size',$MicroBatchSize.ToString(),'--effective-batch-size',$EffectiveBatchSize.ToString(),
    '--memory-budget-mib',$MemoryBudgetMiB.ToString(),'--min-free-ram-mib',$MinFreeRamMiB.ToString(),
    '--vram-budget-mib',$VramBudgetMiB.ToString(),'--min-free-vram-mib',$MinFreeVramMiB.ToString(),'--device',$Device,
    '--models') + $Models + @('--seeds') + @($Seeds | ForEach-Object { $_.ToString() })
if ($Resume) { $runArgs += '--resume' }
if ($SmokeTest) { $runArgs += '--smoke-test' }
$taskEnvironment = @{
    OMP_NUM_THREADS='4'; MKL_NUM_THREADS='4'; OPENBLAS_NUM_THREADS='1'
    PYTORCH_ALLOC_CONF='max_split_size_mb:128,garbage_collection_threshold:0.8'
    PYTHONIOENCODING='utf-8'; HF_HUB_OFFLINE='1'
}
$taskPrevious = @{}
foreach ($name in $taskEnvironment.Keys) {
    $taskPrevious[$name] = [Environment]::GetEnvironmentVariable($name,'Process')
    [Environment]::SetEnvironmentVariable($name,$taskEnvironment[$name],'Process')
}
Push-Location $project
try {
    if ($InstallDependencies) {
        & uv sync --cache-dir (Join-Path $project 'temp\uv_benchmark_cache') --extra training --extra confident --extra segmentation-benchmark --quiet
        if ($LASTEXITCODE -ne 0) { throw 'Benchmark dependency installation failed.' }
    }
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Python environment missing. Use -InstallDependencies.' }
    Write-Host "Ordinary supervised segmentation | $($Models.Count) models x $($Seeds.Count) seeds | Mode: $Mode"
    Write-Host "Data: $Data | Output: $Output"
    Write-Host "RAM budget $MemoryBudgetMiB MiB | CUDA allocator budget $VramBudgetMiB MiB | Effective batch $EffectiveBatchSize"
    & $python @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed ($LASTEXITCODE). See error.json and memory_log.csv/vram_log.csv under $Output." }
}
finally {
    Pop-Location
    foreach ($name in $taskEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name,$taskPrevious[$name],'Process') }
}
if ($Open -and (Test-Path -LiteralPath (Join-Path $Output 'index.html'))) {
    Start-Process -FilePath (Join-Path $Output 'index.html') -WindowStyle Hidden
}
