[CmdletBinding()]
param([switch]$Memory, [string]$Output = '')
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $project '.venv\Scripts\python.exe'
Push-Location $project
try {
    # Keep temporary artifacts inside the workspace, including under sandboxed Windows.
    & $python -X utf8 -c "import tempfile,unittest; from pathlib import Path; p=Path('temp/confident_unit_tests').resolve(); p.mkdir(parents=True,exist_ok=True); tempfile.tempdir=str(p); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(['tests.test_confident','tests.test_confident_abcd','tests.test_confident_central','tests.test_training','tests.test_training_improvements','tests.test_augmentation'])); raise SystemExit(not r.wasSuccessful())"
    if ($LASTEXITCODE -ne 0) { throw 'Confident Learning tests failed.' }
    if ($Memory) {
        $arguments = @('-X','utf8','-u','-m','src.Namrc.memory_check')
        if ($Output) { $arguments += @('--output',$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)) }
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw 'Memory measurement failed.' }
    }
}
finally { Pop-Location }
