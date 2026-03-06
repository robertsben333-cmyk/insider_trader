[CmdletBinding()]
param(
    [string]$VmHost = "143.47.182.234",
    [string]$User = "opc",
    [string]$KeyPath = "$HOME/.ssh/oracle_insider.key",
    [string]$RemoteRepoPath = "/home/opc/insider_trader",
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path $repoRoot "live/data/vm_sync"
}

$resolvedKeyPath = [Environment]::ExpandEnvironmentVariables($KeyPath)
if (-not (Test-Path $resolvedKeyPath)) {
    throw "SSH key not found: $resolvedKeyPath"
}

if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
    throw "scp was not found on PATH. Install OpenSSH client or use Git Bash."
}

$files = @(
    "alert_candidate_history.csv",
    "latest_alert_candidates.csv",
    "live_predictions.csv",
    "insider_purchases.csv",
    "event_history_aggregated.csv"
)

New-Item -ItemType Directory -Force -Path $Destination | Out-Null

foreach ($file in $files) {
    $remote = "{0}@{1}:{2}/live/data/{3}" -f $User, $VmHost, $RemoteRepoPath, $file
    $local = Join-Path $Destination $file
    Write-Host "Pulling $file ..."
    & scp -i $resolvedKeyPath $remote $local
}

Write-Host ""
Write-Host "VM live CSV sync complete."
Write-Host "Files saved to: $Destination"
