[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [string]$Data = '',
    [string]$Output = '',
    [ValidateRange(1,1024)][int]$BatchSize = 8,
    [ValidateSet('cuda','cpu','auto')][string]$Device = 'cuda',
    [ValidateRange(1,2147483647)][int]$PixelChunkSize = 262144,
    [ValidateRange(1,2147483647)][int]$SortChunkSize = 262144,
    [ValidateRange(128,1048576)][int]$MemoryBudgetMiB = 4096,
    [ValidateRange(0,1048576)][int]$MinFreeRamMiB = 1024,
    [switch]$NoAmp,
    [switch]$Resume,
    [switch]$Open
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $project '.venv\Scripts\python.exe'
if ($Resume -and -not $Output) { throw 'Resume requires an explicit Output directory.' }
if (-not $Data) { $Data = Join-Path $project 'data\changed_v3' }
if (-not $Output) { $Output = Join-Path $project ('analysis\confident\A_p123_clean_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')) }
foreach ($name in @('Checkpoint','Data','Output')) {
    $value = Get-Variable -Name $name -ValueOnly
    Set-Variable -Name $name -Value $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($value)
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Project Python environment is missing.' }
if ((Test-Path -LiteralPath $Output) -and -not $Resume) { throw "Choose a new output directory: $Output" }
$arguments = @('-X','utf8','-u','-m','src.inference.audit_clean_ratio',
    '--checkpoint',$Checkpoint,'--data',$Data,'--output',$Output,'--device',$Device,
    '--batch-size',$BatchSize.ToString(),'--pixel-chunk-size',$PixelChunkSize.ToString(),
    '--sort-chunk-size',$SortChunkSize.ToString(),'--memory-budget-mib',$MemoryBudgetMiB.ToString(),
    '--min-free-ram-mib',$MinFreeRamMiB.ToString())
if ($NoAmp) { $arguments += '--no-amp' }
if ($Resume) { $arguments += '--resume' }
Push-Location $project
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Inference/CL audit failed: $LASTEXITCODE" }
    Import-Csv -LiteralPath (Join-Path $Output 'by_point.csv') | Format-Table point,tiles,cl_clean_pct,cl_issue_pct,pred_impregnation_pct
    if ($Open) { Invoke-Item -LiteralPath (Join-Path $Output 'index.html') }
} finally { Pop-Location }
