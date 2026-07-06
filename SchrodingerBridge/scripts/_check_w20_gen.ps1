# Check W20 generation scripts and configs
$ErrorActionPreference = "Continue"

$REPO = "I:\Github\Latent_Style\SchrodingerBridge"

Write-Host "=== W20 gen scripts ==="
Get-ChildItem "$REPO\scripts" -Filter "*_gen_*w20*" -ErrorAction SilentlyContinue |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -Auto
Get-ChildItem "$REPO\scripts" -Filter "*_gen_*wiki20*" -ErrorAction SilentlyContinue |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -Auto
Get-ChildItem "$REPO\scripts" -Filter "*_run_*w20*" -ErrorAction SilentlyContinue |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -Auto

Write-Host ""
Write-Host "=== Check sdturbo w20 gen script ==="
$f = Get-ChildItem "$REPO\scripts" -Filter "*sdturbo*w20*" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($f) {
    Write-Host "Found: $($f.Name)"
    Get-Content $f.FullName -TotalCount 50
}

Write-Host ""
Write-Host "=== Check styleid w20 gen script ==="
$f2 = Get-ChildItem "$REPO\scripts" -Filter "*styleid*w20*" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($f2) {
    Write-Host "Found: $($f2.Name)"
    Get-Content $f2.FullName -TotalCount 50
}

Write-Host ""
Write-Host "=== Check samst w20 gen script ==="
$f3 = Get-ChildItem "$REPO\scripts" -Filter "*samst*w20*" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($f3) {
    Write-Host "Found: $($f3.Name)"
    Get-Content $f3.FullName -TotalCount 50
}

Write-Host ""
Write-Host "=== Check sample W20 sdturbo filename ==="
$dir = "$REPO\exp\baseline_wikiarts20\sdturbo\images"
if (Test-Path $dir) {
    Get-ChildItem $dir -File | Select-Object -First 3 | ForEach-Object { Write-Host $_.Name }
}

Write-Host ""
Write-Host "=== Check sample W20 adain filename ==="
$dir2 = "$REPO\exp\baseline_wikiarts20\adain\images"
if (Test-Path $dir2) {
    Get-ChildItem $dir2 -File | Select-Object -First 3 | ForEach-Object { Write-Host $_.Name }
}

Write-Host ""
Write-Host "=== adain target style distribution ==="
$adainDir = "$REPO\exp\baseline_wikiarts20\adain\images"
if (Test-Path $adainDir) {
    $files = Get-ChildItem $adainDir -File
    $styles = @{}
    foreach ($f in $files) {
        $name = $f.BaseName
        if ($name -match "_to_(.+)$") {
            $tgt = $matches[1]
            if (-not $styles.ContainsKey($tgt)) { $styles[$tgt] = 0 }
            $styles[$tgt]++
        }
    }
    $styles.GetEnumerator() | Sort-Object Name | ForEach-Object { Write-Host "  $($_.Name): $($_.Value)" }
}
