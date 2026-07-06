$logs = @(
    "I:\Github\Latent_Style\SchrodingerBridge\logs\wikiarts20_all.log",
    "I:\Github\Latent_Style\SchrodingerBridge\logs\baseline_wikiarts20.log"
)

foreach ($log in $logs) {
    Write-Output "=== $log ==="
    if (Test-Path $log) {
        Get-Content $log -Tail 40
    } else {
        Write-Output "  (not yet created)"
    }
    Write-Output ""
}

# Check image counts in baseline_wikiarts20
$bw = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20"
if (Test-Path $bw) {
    Write-Output "=== baseline_wikiarts20 image counts ==="
    Get-ChildItem $bw -Directory | ForEach-Object {
        $name = $_.Name
        $imgs = Join-Path $bw "$name\images"
        $cnt = if (Test-Path $imgs) { (Get-ChildItem $imgs -Filter "*.png" -ErrorAction SilentlyContinue).Count } else { 0 }
        $done = Test-Path (Join-Path $bw "$name\_DONE")
        $csv = Test-Path (Join-Path $bw "$name\metrics.csv")
        Write-Output ("  {0,-12} imgs={1,-6} done={2} csv={3}" -f $name, $cnt, $done, $csv)
    }
}

# Check WD-VF eval dir
$ev = "I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts20_eval"
if (Test-Path $ev) {
    Write-Output ""
    Write-Output "=== wikiarts20_eval ==="
    $csv = Test-Path (Join-Path $ev "metrics.csv")
    $imgs = Join-Path $ev "images"
    $imgCnt = if (Test-Path $imgs) { (Get-ChildItem $imgs -Filter "*.png" -ErrorAction SilentlyContinue).Count } else { 0 }
    Write-Output "  metrics.csv exists: $csv"
    Write-Output "  images: $imgCnt"
}

# schtasks status
Write-Output ""
Write-Output "=== schtasks wikiarts20_all status ==="
schtasks /Query /TN "wikiarts20_all" /FO LIST 2>&1 | Select-String -Pattern "Status|Last"
