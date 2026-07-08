$ErrorActionPreference = 'Continue'

$exp_root = "I:\Github\Latent_Style\SchrodingerBridge\exp"

Write-Host "============================================================"
Write-Host "PART A: Deep inspect unified_results.json"
Write-Host "============================================================"
$ur = "$exp_root\baseline_v2\eval\unified_results.json"
if (Test-Path $ur) {
    $j = Get-Content $ur -Raw | ConvertFrom-Json
    foreach ($method in $j.PSObject.Properties.Name) {
        Write-Host "`n--- $method ---"
        $m = $j.$method
        $keys = $m.PSObject.Properties.Name
        $time_keys = $keys | Where-Object { $_ -match "time|wall|infer|train|duration|elapsed|gen" }
        foreach ($k in $time_keys) {
            Write-Host ("  " + $k + ": " + $m.$k)
        }
        # Also show all keys for first few methods
        Write-Host ("  all_keys: " + ($keys -join ", "))
    }
}

Write-Host "`n============================================================"
Write-Host "PART B: Check metadata.json in baseline dirs"
Write-Host "============================================================"
$meta_dirs = @(
    "$exp_root\baseline_wikiarts20\identity",
    "$exp_root\baseline_wikiarts20\adain",
    "$exp_root\baseline_wikiarts20\wct",
    "$exp_root\baseline_wikiarts20\sdturbo",
    "$exp_root\baseline_wikiarts20\samst",
    "$exp_root\baseline_wikiarts20\samam",
    "$exp_root\baseline_wikiarts20\styleid"
)
foreach ($d in $meta_dirs) {
    $mj = Join-Path $d "metadata.json"
    if (Test-Path $mj) {
        Write-Host "`n--- $d\metadata.json ---"
        $j = Get-Content $mj -Raw | ConvertFrom-Json
        $keys = $j.PSObject.Properties.Name
        $time_keys = $keys | Where-Object { $_ -match "time|wall|infer|train|duration|elapsed|gen" }
        foreach ($k in $time_keys) {
            Write-Host ("  " + $k + ": " + $j.$k)
        }
        Write-Host ("  all_keys: " + ($keys -join ", "))
    }
}

Write-Host "`n============================================================"
Write-Host "PART C: Check baseline_v2/baseline_summary_table.csv"
Write-Host "============================================================"
$bcsv = "$exp_root\baseline_v2\baseline_summary_table.csv"
if (Test-Path $bcsv) {
    Get-Content $bcsv
}

Write-Host "`n============================================================"
Write-Host "PART D: Check baseline_v2/baseline_conclusions.md for timing"
Write-Host "============================================================"
$bmd = "$exp_root\baseline_v2\baseline_conclusions.md"
if (Test-Path $bmd) {
    $content = Get-Content $bmd -Raw
    # Find lines with time/timing/inference
    $content -split "`n" | Where-Object { $_ -match "time|infer|train|wall|second|minute|hour" } | ForEach-Object { Write-Host $_ }
}

Write-Host "`n============================================================"
Write-Host "PART E: Check sdturbo images dir for count and any timing"
Write-Host "============================================================"
$sdt_dir = "$exp_root\baseline_wikiarts20\sdturbo\images"
if (Test-Path $sdt_dir) {
    $cnt = (Get-ChildItem $sdt_dir -Filter *.png).Count
    Write-Host "  sdturbo R5 image count: $cnt"
}

# Also check D5 sdturbo images
$sdt_d5 = "$exp_root\baseline_v2\images\sdturbo"
if (Test-Path $sdt_d5) {
    $cnt2 = (Get-ChildItem $sdt_d5 -Filter *.png).Count
    Write-Host "  sdturbo D5 image count: $cnt2"
}

Write-Host "`n============================================================"
Write-Host "PART F: Check for seedream images in baseline_v2"
Write-Host "============================================================"
Get-ChildItem "$exp_root\baseline_v2" -Directory -Recurse -Depth 2 -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "seedream" } | ForEach-Object {
    $d = $_.FullName
    $cnt = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count
    $cnt2 = (Get-ChildItem $d -Filter *.jpg -ErrorAction SilentlyContinue).Count
    Write-Host ("  " + $d + " png=" + $cnt + " jpg=" + $cnt2)
    $sample = Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue | Select-Object -First 3
    $sample | ForEach-Object { Write-Host ("    " + $_.Name) }
}

Write-Host "`n============================================================"
Write-Host "PART G: Check for any timing in _scan_results.json"
Write-Host "============================================================"
$sr = "$exp_root\_scan_results.json"
if (Test-Path $sr) {
    $j = Get-Content $sr -Raw | ConvertFrom-Json
    # This might be large, just show keys
    Write-Host ("  top_keys: " + ($j.PSObject.Properties.Name -join ", "))
}
