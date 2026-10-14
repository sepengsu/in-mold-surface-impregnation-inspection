[CmdletBinding()]
param(
    [string]$SourceZip = '',
    [string]$Output = '',
    [ValidateRange(1,100000)][int]$ExpectedImages = 72
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $project '.venv\Scripts\python.exe'
$arguments = @('-X','utf8','-u','-m','src.segmentation.export_label_studio_brush','--expected-images',$ExpectedImages.ToString())
foreach ($entry in @(@('--source-zip',$SourceZip),@('--output',$Output))) {
    if ($entry[1]) { $arguments += @($entry[0],$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($entry[1])) }
}
Push-Location $project
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Label Studio PNG -> NumPy -> JSON conversion failed: $LASTEXITCODE" }
} finally { Pop-Location }
