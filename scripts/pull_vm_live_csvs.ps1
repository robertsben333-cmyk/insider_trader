[CmdletBinding()]
param(
    [string]$VmHost = "143.47.182.234",
    [string]$User = "opc",
    [string]$KeyPath = "$HOME/.ssh/oracle_insider.key",
    [string]$RemoteRepoPath = "/home/opc/insider_trader",
    [string]$Destination = "",
    [double]$AlertThreshold = [double]::NaN
)

$ErrorActionPreference = "Stop"
$invariantCulture = [System.Globalization.CultureInfo]::InvariantCulture

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
    "latest_alert_candidates.csv"
)

New-Item -ItemType Directory -Force -Path $Destination | Out-Null

$obsoleteFiles = @(
    "live_predictions.csv",
    "insider_purchases.csv",
    "event_history_aggregated.csv"
)
foreach ($obsolete in $obsoleteFiles) {
    $obsoletePath = Join-Path $Destination $obsolete
    if (Test-Path $obsoletePath) {
        Remove-Item -Force $obsoletePath
    }
}

foreach ($file in $files) {
    $remote = "{0}@{1}:{2}/live/data/{3}" -f $User, $VmHost, $RemoteRepoPath, $file
    $local = Join-Path $Destination $file
    Write-Host "Pulling $file ..."
    & scp -i $resolvedKeyPath $remote $local
}

$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("vm_live_sync_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
$tempPredictions = Join-Path $tempDir "live_predictions.csv"
$remotePredictions = "{0}@{1}:{2}/live/data/live_predictions.csv" -f $User, $VmHost, $RemoteRepoPath
Write-Host "Pulling live_predictions.csv for historical recommendation export ..."
& scp -i $resolvedKeyPath $remotePredictions $tempPredictions

$latestAlertPath = Join-Path $Destination "latest_alert_candidates.csv"
if ([double]::IsNaN($AlertThreshold) -and (Test-Path $latestAlertPath)) {
    $latestAlertRows = Import-Csv $latestAlertPath
    if ($latestAlertRows.Count -gt 0) {
        $thresholdText = [string]$latestAlertRows[0].raw_alert_threshold
        if (-not [string]::IsNullOrWhiteSpace($thresholdText)) {
            $parsedThreshold = 0.0
            if ([double]::TryParse($thresholdText, [System.Globalization.NumberStyles]::Float, $invariantCulture, [ref]$parsedThreshold)) {
                $AlertThreshold = $parsedThreshold
            }
        }
    }
}
if ([double]::IsNaN($AlertThreshold)) {
    $AlertThreshold = 0.7129917140924689
}

$predictionRows = Import-Csv $tempPredictions
$score3Lookup = @{}
foreach ($row in $predictionRows) {
    if ($row.horizon_days -eq "3") {
        $score3Lookup["{0}|{1}" -f $row.event_key, $row.scored_at] = $row.pred_mean4
    }
}

$recommended = foreach ($row in $predictionRows) {
    if ($row.horizon_days -ne "1") {
        continue
    }
    $predValue = 0.0
    if (-not [double]::TryParse([string]$row.pred_mean4, [System.Globalization.NumberStyles]::Float, $invariantCulture, [ref]$predValue)) {
        continue
    }
    if ($predValue -le $AlertThreshold) {
        continue
    }

    $lookupKey = "{0}|{1}" -f $row.event_key, $row.scored_at
    [pscustomobject]@{
        scored_at            = $row.scored_at
        event_key            = $row.event_key
        ticker               = $row.ticker
        company_name         = $row.company_name
        owner_name           = $row.owner_name
        title                = $row.title
        trade_date           = $row.trade_date
        buy_price            = $row.buy_price
        score_1d             = $row.pred_mean4
        score_3d             = $score3Lookup[$lookupKey]
        pred_mean4           = $row.pred_mean4
        market_type          = $row.market_type
        is_tradable          = $row.is_tradable
        raw_alert_threshold  = $AlertThreshold.ToString("G17", $invariantCulture)
    }
}

$historicalPath = Join-Path $Destination "historical_recommended_predictions.csv"
$recommended |
    Sort-Object @{ Expression = { $_.scored_at }; Descending = $true }, @{ Expression = { [double]$_.score_1d }; Descending = $true }, ticker |
    Export-Csv -NoTypeInformation -Encoding utf8 -Path $historicalPath

Remove-Item -Recurse -Force $tempDir

Write-Host ""
Write-Host "VM live CSV sync complete."
Write-Host "Files saved to: $Destination"
