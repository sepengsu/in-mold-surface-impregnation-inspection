[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Run,
    [string]$Output = '',
    [ValidateSet('cuda','cpu','auto')][string]$Device = 'cuda',
    [switch]$IncludeCase16
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $project '.venv\Scripts\python.exe'
$Run = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Run)
if (-not $Output) {
    $Output = Join-Path $project ('analysis\p123_inference_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
$Output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
if (Test-Path -LiteralPath $Output) { throw "Use a new output directory: $Output" }
$checkpoint = Join-Path $Run 'best.pt'
$config = Get-Content -LiteralPath (Join-Path $Run 'config.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$result = Get-Content -LiteralPath (Join-Path $Run 'result.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($config.smoke_test -or $result.status -ne 'completed' -or $result.test_is_partial) {
    throw 'Use a completed full training run.'
}
$provenance = Join-Path $project 'outputs\measurements\rectangle_roi_20261001\case16_provenance.json'
Push-Location $project
try {
    & $python -X utf8 -u -m src.inference.predict --checkpoint $checkpoint --data $config.data `
      --split all --position all --central-three --device $Device --output (Join-Path $Output 'tiles_all')
    if ($LASTEXITCODE -ne 0) { throw 'P1/P2/P3 inference failed.' }
    if ($IncludeCase16) {
        & $python -X utf8 -u -m src.inference.predict --checkpoint $checkpoint `
          --images (Join-Path (Split-Path $provenance) 'case16_tiles') `
          --device $Device --output (Join-Path $Output 'case16_unlabeled')
        if ($LASTEXITCODE -ne 0) { throw 'Case16 inference failed.' }
    }
    $arguments = @('-X','utf8','-u','-m','src.inference.report_p123','--run',$Run,'--output',$Output)
    if ($IncludeCase16) { $arguments += @('--case16-provenance',$provenance) }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw 'P1/P2/P3 report failed.' }
    $signedArguments = @('-X','utf8','-u','-m','src.inference.report_signed_error',
        '--inference',$Output,'--output',(Join-Path $Output 'signed_error'))
    if ($IncludeCase16) { $signedArguments += @('--case16-provenance',$provenance) }
    & $python @signedArguments
    if ($LASTEXITCODE -ne 0) { throw 'Signed error and Case05 overlap report failed.' }
    Write-Host "Report (actual minus predicted, pp): $Output\signed_error\index.html"
} finally { Pop-Location }
