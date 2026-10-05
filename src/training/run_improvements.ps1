[CmdletBinding()]
param(
    [string]$Data = '',
    [string]$Output = '',
    [ValidateRange(1, 100000)][int]$Epochs = 100,
    [ValidateRange(1, 1024)][int]$BatchSize = 16,
    [ValidateRange(0.0000001, 1.0)][double]$LearningRate = 0.001,
    [ValidateRange(1, 100000)][int]$Patience = 20,
    [ValidateRange(0, 2147483647)][int]$Seed = 42,
    [ValidateRange(0, 2147483647)][int]$SplitSeed = 42,
    [ValidateRange(0.000001, 0.999999)][double]$TverskyAlpha = 0.3,
    [ValidateRange(0.000001, 0.999999)][double]$TverskyBeta = 0.7,
    [ValidateRange(0.000001, 1.0)][double]$LargeRegionFraction = 0.05,
    [ValidateRange(1.0, 1000.0)][double]$LargeRegionWeight = 3.0,
    [ValidateRange(0, 64)][int]$Workers = 0,
    [ValidateSet('all', '000', '180')][string]$Position = 'all',
    [ValidateSet('cuda', 'cpu', 'auto')][string]$Device = 'cuda',
    [switch]$InstallDependencies,
    [switch]$SmokeTest,
    [switch]$NoAmp,
    [switch]$Open
)

# Keep preprocessing/model/augmentation fixed; vary sampling and loss only.
$ErrorActionPreference = 'Stop'
$runOptions = @{
    Suite = 'improvement'; Experiments = 'all'; Data = $Data; Output = $Output
    Epochs = $Epochs; BatchSize = $BatchSize; LearningRate = $LearningRate; Patience = $Patience
    Seed = $Seed; SplitSeed = $SplitSeed; Workers = $Workers; Position = $Position; Device = $Device
    TverskyAlpha = $TverskyAlpha; TverskyBeta = $TverskyBeta
    LargeRegionFraction = $LargeRegionFraction; LargeRegionWeight = $LargeRegionWeight
    InstallDependencies = $InstallDependencies; SmokeTest = $SmokeTest; NoAmp = $NoAmp; Open = $Open
}
& (Join-Path $PSScriptRoot 'run_comparison.ps1') @runOptions
