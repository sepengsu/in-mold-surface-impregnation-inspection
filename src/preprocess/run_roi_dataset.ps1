# Compatibility entry point: current reviewed horizontal-rectangle ROI pipeline.
[CmdletBinding()]
param(
    [string]$Output = '',
    [switch]$CountOnly
)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'run_rectangle_dataset.ps1') -Output $Output -CountOnly:$CountOnly
