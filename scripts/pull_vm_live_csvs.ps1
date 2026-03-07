[CmdletBinding()]
param(
    [string]$VmHost = "143.47.182.234",
    [string]$User = "opc",
    [string]$KeyPath = "$HOME/.ssh/oracle_insider.key",
    [string]$RemoteRepoPath = "/home/opc/insider_trader",
    [string]$Destination = "",
    [double]$AlertThreshold = [double]::NaN,
    [switch]$UseLocalLiveData
)

$ErrorActionPreference = "Stop"
$invariantCulture = [System.Globalization.CultureInfo]::InvariantCulture

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path $repoRoot "live/data/vm_sync"
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

$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("vm_live_sync_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
$tempPredictions = Join-Path $tempDir "live_predictions.csv"
$localLiveDir = Join-Path $repoRoot "live/data"

if ($UseLocalLiveData) {
    foreach ($file in $files) {
        $source = Join-Path $localLiveDir $file
        $local = Join-Path $Destination $file
        if (-not (Test-Path $source)) {
            throw "Local live-data file not found: $source"
        }
        Copy-Item -Force $source $local
    }

    $localPredictions = Join-Path $localLiveDir "live_predictions.csv"
    if (-not (Test-Path $localPredictions)) {
        throw "Local live_predictions.csv not found: $localPredictions"
    }
    Write-Host "Using local live/data files for VM-sync export ..."
    Copy-Item -Force $localPredictions $tempPredictions
} else {
    $resolvedKeyPath = [Environment]::ExpandEnvironmentVariables($KeyPath)
    if (-not (Test-Path $resolvedKeyPath)) {
        throw "SSH key not found: $resolvedKeyPath"
    }

    if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
        throw "scp was not found on PATH. Install OpenSSH client or use Git Bash."
    }

    foreach ($file in $files) {
        $remote = "{0}@{1}:{2}/live/data/{3}" -f $User, $VmHost, $RemoteRepoPath, $file
        $local = Join-Path $Destination $file
        Write-Host "Pulling $file ..."
        & scp -i $resolvedKeyPath $remote $local
    }

    $remotePredictions = "{0}@{1}:{2}/live/data/live_predictions.csv" -f $User, $VmHost, $RemoteRepoPath
    Write-Host "Pulling live_predictions.csv for historical recommendation export ..."
    & scp -i $resolvedKeyPath $remotePredictions $tempPredictions
}

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
    $AlertThreshold = 0.6091125803233034
}

$predictionRows = Import-Csv $tempPredictions
$score3Lookup = @{}
foreach ($row in $predictionRows) {
    if ($row.horizon_days -eq "3") {
        $score3Lookup["{0}|{1}" -f $row.event_key, $row.scored_at] = $row.pred_mean4
    }
}

$alertHistoryPath = Join-Path $Destination "alert_candidate_history.csv"
$alertHistoryLookup = @{}
if (Test-Path $alertHistoryPath) {
    foreach ($row in (Import-Csv $alertHistoryPath)) {
        $alertHistoryLookup["{0}|{1}" -f $row.event_key, $row.scored_at] = $row
    }
}

$decileCurvePath = Join-Path $repoRoot "backtest/out/investable_decile_score_sweep_0005_tplus2_open.csv"
$curvePoints = New-Object System.Collections.Generic.List[object]
if (Test-Path $decileCurvePath) {
    foreach ($row in (Import-Csv $decileCurvePath)) {
        $decileValue = 0.0
        $rawCutoffValue = 0.0
        if (
            [double]::TryParse([string]$row.decile_score_threshold, [System.Globalization.NumberStyles]::Float, $invariantCulture, [ref]$decileValue) -and
            [double]::TryParse([string]$row.raw_pred_mean4_cutoff, [System.Globalization.NumberStyles]::Float, $invariantCulture, [ref]$rawCutoffValue)
        ) {
            $curvePoints.Add([pscustomobject]@{
                decile_score = $decileValue
                raw_cutoff   = $rawCutoffValue
            }) | Out-Null
        }
    }
}
$curvePoints = @($curvePoints | Sort-Object raw_cutoff, decile_score)
$normalizedCurve = New-Object System.Collections.Generic.List[object]
$maxRawSeen = [double]::NegativeInfinity
foreach ($point in $curvePoints) {
    $rawValue = [double]$point.raw_cutoff
    if ($rawValue -lt $maxRawSeen) {
        $rawValue = $maxRawSeen
    } else {
        $maxRawSeen = $rawValue
    }

    if ($normalizedCurve.Count -gt 0 -and [double]$normalizedCurve[$normalizedCurve.Count - 1].raw_cutoff -eq $rawValue) {
        $normalizedCurve[$normalizedCurve.Count - 1] = [pscustomobject]@{
            decile_score = [double]$point.decile_score
            raw_cutoff   = $rawValue
        }
    } else {
        $normalizedCurve.Add([pscustomobject]@{
            decile_score = [double]$point.decile_score
            raw_cutoff   = $rawValue
        }) | Out-Null
    }
}

function Get-EstimatedDecileScore {
    param(
        [double]$RawPrediction,
        [object[]]$Curve
    )

    if (-not $Curve -or $Curve.Count -eq 0) {
        return $null
    }
    if ($Curve.Count -eq 1) {
        return [double]$Curve[0].decile_score
    }
    if ($RawPrediction -le [double]$Curve[0].raw_cutoff) {
        return [double]$Curve[0].decile_score
    }
    $lastIdx = $Curve.Count - 1
    if ($RawPrediction -ge [double]$Curve[$lastIdx].raw_cutoff) {
        return [double]$Curve[$lastIdx].decile_score
    }

    for ($i = 1; $i -lt $Curve.Count; $i++) {
        $x0 = [double]$Curve[$i - 1].raw_cutoff
        $x1 = [double]$Curve[$i].raw_cutoff
        if ($RawPrediction -le $x1) {
            $y0 = [double]$Curve[$i - 1].decile_score
            $y1 = [double]$Curve[$i].decile_score
            if ($x1 -eq $x0) {
                return $y1
            }
            return $y0 + (($RawPrediction - $x0) * ($y1 - $y0) / ($x1 - $x0))
        }
    }
    return [double]$Curve[$lastIdx].decile_score
}

$defaultDecileThreshold = 0.87
$defaultBaseAlloc = 0.25
$defaultBonusAlloc = 0.25

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
    $alertRow = $alertHistoryLookup[$lookupKey]

    $estimatedDecile = $null
    $estimatedDecileText = ""
    if ($null -ne $alertRow -and -not [string]::IsNullOrWhiteSpace([string]$alertRow.estimated_decile_score)) {
        $estimatedDecileText = [string]$alertRow.estimated_decile_score
        $parsedDecile = 0.0
        if ([double]::TryParse($estimatedDecileText, [System.Globalization.NumberStyles]::Float, $invariantCulture, [ref]$parsedDecile)) {
            $estimatedDecile = $parsedDecile
        }
    }
    if ($null -eq $estimatedDecile -and $normalizedCurve.Count -gt 0) {
        $estimatedDecile = Get-EstimatedDecileScore -RawPrediction $predValue -Curve $normalizedCurve
        if ($null -ne $estimatedDecile) {
            $estimatedDecileText = ([double]$estimatedDecile).ToString("G17", $invariantCulture)
        }
    }

    $decileThresholdText = if ($null -ne $alertRow -and -not [string]::IsNullOrWhiteSpace([string]$alertRow.decile_score_threshold)) {
        [string]$alertRow.decile_score_threshold
    } else {
        $defaultDecileThreshold.ToString("G17", $invariantCulture)
    }
    $decileThreshold = $defaultDecileThreshold
    [void][double]::TryParse($decileThresholdText, [System.Globalization.NumberStyles]::Float, $invariantCulture, [ref]$decileThreshold)

    $baseAllocText = if ($null -ne $alertRow -and -not [string]::IsNullOrWhiteSpace([string]$alertRow.advised_allocation_fraction)) {
        [string]$alertRow.advised_allocation_fraction
    } else {
        ""
    }
    $bonusAllocText = ""
    $decileStrengthText = ""
    $advisedAllocPctText = ""
    if ($null -ne $alertRow) {
        $decileStrengthText = [string]$alertRow.decile_strength
        $advisedAllocPctText = [string]$alertRow.advised_allocation_pct
    }

    if ([string]::IsNullOrWhiteSpace($baseAllocText) -and $null -ne $estimatedDecile) {
        $denom = [Math]::Max(1e-9, 1.0 - $decileThreshold)
        $decileStrength = [Math]::Min([Math]::Max((([double]$estimatedDecile) - $decileThreshold) / $denom, 0.0), 1.0)
        $advisedAllocFraction = [Math]::Min([Math]::Max($defaultBaseAlloc + ($defaultBonusAlloc * $decileStrength), $defaultBaseAlloc), $defaultBaseAlloc + $defaultBonusAlloc)
        $baseAllocText = $advisedAllocFraction.ToString("G17", $invariantCulture)
        $decileStrengthText = $decileStrength.ToString("G17", $invariantCulture)
        $advisedAllocPctText = ($advisedAllocFraction * 100.0).ToString("G17", $invariantCulture)
    }

    [pscustomobject]@{
        scored_at                   = $row.scored_at
        event_key                   = $row.event_key
        ticker                      = $row.ticker
        company_name                = $row.company_name
        owner_name                  = $row.owner_name
        title                       = $row.title
        trade_date                  = $row.trade_date
        buy_price                   = $row.buy_price
        score_1d                    = $row.pred_mean4
        score_3d                    = $score3Lookup[$lookupKey]
        pred_mean4                  = $row.pred_mean4
        estimated_decile_score      = $estimatedDecileText
        decile_strength             = $decileStrengthText
        advised_allocation_fraction = $baseAllocText
        advised_allocation_pct      = $advisedAllocPctText
        market_type                 = $row.market_type
        is_tradable                 = $row.is_tradable
        raw_alert_threshold         = if ($null -ne $alertRow -and -not [string]::IsNullOrWhiteSpace([string]$alertRow.raw_alert_threshold)) { [string]$alertRow.raw_alert_threshold } else { $AlertThreshold.ToString("G17", $invariantCulture) }
        decile_score_threshold      = $decileThresholdText
        threshold_source            = if ($null -ne $alertRow -and -not [string]::IsNullOrWhiteSpace([string]$alertRow.threshold_source)) { [string]$alertRow.threshold_source } else { "backtest/out/investable_decile_score_sweep_0005_tplus2_open.csv" }
        alert_score_column          = if ($null -ne $alertRow -and -not [string]::IsNullOrWhiteSpace([string]$alertRow.alert_score_column)) { [string]$alertRow.alert_score_column } else { "pred_mean4" }
        target_return_mode          = if ($null -ne $alertRow -and -not [string]::IsNullOrWhiteSpace([string]$alertRow.target_return_mode)) { [string]$alertRow.target_return_mode } else { "spy_adjusted_excess_return_pct" }
        benchmark_ticker            = if ($null -ne $alertRow -and -not [string]::IsNullOrWhiteSpace([string]$alertRow.benchmark_ticker)) { [string]$alertRow.benchmark_ticker } else { "SPY" }
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
