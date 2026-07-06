# Aggregate metrics.csv for each method in baseline_wikiarts20
$bw = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20"

foreach ($method in @("identity", "adain", "wct")) {
    $csv = "$bw\$method\metrics.csv"
    if (Test-Path $csv) {
        $rows = Import-Csv $csv
        $n = 0
        $csSum = 0.0
        $lpSum = 0.0
        foreach ($r in $rows) {
            try {
                $cs = [double]$r.clip_style
                $lp = [double]$r.content_lpips
                $csSum += $cs
                $lpSum += $lp
                $n++
            } catch { continue }
        }
        if ($n -gt 0) {
            $csAvg = $csSum / $n
            $lpAvg = $lpSum / $n
            Write-Output ("{0,-12} CLIP-S={1:F4}  LPIPS={2:F4}  n={3}" -f $method, $csAvg, $lpAvg, $n)
        } else {
            Write-Output ("{0,-12} NO VALID ROWS" -f $method)
        }
    } else {
        Write-Output ("{0,-12} NO CSV" -f $method)
    }
}

# Also check WD-VF eval
$ev = "I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts20_eval\metrics.csv"
if (Test-Path $ev) {
    $rows = Import-Csv $ev
    $n = 0
    $csSum = 0.0
    $lpSum = 0.0
    foreach ($r in $rows) {
        try {
            $cs = [double]$r.clip_style
            $lp = [double]$r.content_lpips
            $csSum += $cs
            $lpSum += $lp
            $n++
        } catch { continue }
    }
    if ($n -gt 0) {
        $csAvg = $csSum / $n
        $lpAvg = $lpSum / $n
        Write-Output ("{0,-12} CLIP-S={1:F4}  LPIPS={2:F4}  n={3}" -f "WD-VF", $csAvg, $lpAvg, $n)
    } else {
        Write-Output ("{0,-12} NO VALID ROWS" -f "WD-VF")
    }
} else {
    Write-Output ("{0,-12} NO CSV" -f "WD-VF")
}

# Check if pipeline still running
Write-Output ""
Write-Output "=== schtasks status ==="
schtasks /Query /TN "wikiarts20_all" /FO LIST 2>&1 | Select-String -Pattern "Status|Last"
