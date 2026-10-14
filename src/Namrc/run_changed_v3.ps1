[CmdletBinding()]
param(
    [ValidateSet('prepare','oof','issues','audit-plan','audit-evaluate','freeze-policy','candidates','train','evaluate')]
    [string]$Stage = 'prepare',
    [string]$Config = '',
    [string]$Experiment = '',
    [string]$Output = '',
    [ValidateSet('A','B','C','D')][string[]]$Arms = @('A','B','C','D'),
    [string]$AuditReview = '',
    [ValidateSet('dev','check')][string]$Partition = 'dev',
    [string]$Policy = '',
    [string]$AuditScopeAuthorization = ''
)
$ErrorActionPreference = 'Stop'
$taskProject = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$taskPython = Join-Path $taskProject '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) { throw 'Missing project .venv. Install the confident optional dependencies first.' }
if ($Stage -ne 'prepare' -and -not $Experiment) { throw 'Execution stages require the exact prepared -Experiment path.' }
$taskArgs = @('-X','utf8','-u','-m','src.Namrc.v3','--stage',$Stage,'--partition',$Partition,'--arms') + $Arms
foreach ($entry in @(@('Config','--config'),@('Experiment','--experiment'),@('Output','--output'),
                     @('AuditReview','--audit-review'),@('Policy','--policy'),@('AuditScopeAuthorization','--audit-scope-authorization'))) {
    $taskValue = Get-Variable -Name $entry[0] -ValueOnly
    if ($taskValue) {
        $taskAbsolute = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($taskValue)
        $taskArgs += @($entry[1],$taskAbsolute)
    }
}
Push-Location -LiteralPath $taskProject
try {
    Write-Host "Stage: $Stage | changed_v3 fixed split | RAM/VRAM guards enabled"
    if ($Stage -eq 'prepare') { Write-Host 'Prepare only: no training, OOF, inference or policy fitting.' }
    & $taskPython @taskArgs
    if ($LASTEXITCODE -ne 0) { throw "changed_v3 stage failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
