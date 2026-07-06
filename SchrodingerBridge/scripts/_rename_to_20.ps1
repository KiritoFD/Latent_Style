$ErrorActionPreference = "Continue"

Write-Output "=== Renaming wikiarts15 -> wikiarts20 ==="

# 1. Test dataset directory
$testSrc = "I:\datasets\wikiarts15_512_test"
$testDst = "I:\datasets\wikiarts20_512_test"
if (Test-Path $testSrc) {
    if (Test-Path $testDst) {
        Write-Output "  DEST exists, removing: $testDst"
        Remove-Item $testDst -Recurse -Force
    }
    Rename-Item $testSrc $testDst
    Write-Output "  RENAMED: $testSrc -> $testDst"
} else {
    Write-Output "  SRC not found: $testSrc"
    if (Test-Path $testDst) {
        Write-Output "  DEST already exists: $testDst"
    }
}

# 2. exp baseline_wikiarts15 -> baseline_wikiarts20
$bwSrc = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts15"
$bwDst = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20"
if (Test-Path $bwSrc) {
    if (Test-Path $bwDst) {
        Remove-Item $bwDst -Recurse -Force
    }
    Rename-Item $bwSrc $bwDst
    Write-Output "  RENAMED: $bwSrc -> $bwDst"
}

# 3. exp wikiarts15_eval -> wikiarts20_eval
$evSrc = "I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts15_eval"
$evDst = "I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts20_eval"
if (Test-Path $evSrc) {
    if (Test-Path $evDst) {
        Remove-Item $evDst -Recurse -Force
    }
    Rename-Item $evSrc $evDst
    Write-Output "  RENAMED: $evSrc -> $evDst"
}

# 4. Rename logs (just rename files matching *wikiarts15*)
$logDir = "I:\Github\Latent_Style\SchrodingerBridge\logs"
if (Test-Path $logDir) {
    Get-ChildItem $logDir -File | Where-Object { $_.Name -like "*wikiarts15*" } | ForEach-Object {
        $newName = $_.Name -replace "wikiarts15", "wikiarts20"
        $newPath = Join-Path $logDir $newName
        Rename-Item $_.FullName $newName
        Write-Output "  LOG RENAMED: $($_.Name) -> $newName"
    }
}

# 5. Rename script files in scripts\ dir (rename file names)
$scriptsDir = "I:\Github\Latent_Style\SchrodingerBridge\scripts"
if (Test-Path $scriptsDir) {
    Get-ChildItem $scriptsDir -File | Where-Object { $_.Name -like "*wikiarts15*" } | ForEach-Object {
        $newName = $_.Name -replace "wikiarts15", "wikiarts20"
        Rename-Item $_.FullName $newName
        Write-Output "  SCRIPT RENAMED: $($_.Name) -> $newName"
    }
}

# 6. Delete the failed schtasks job
Write-Output ""
Write-Output "=== Deleting old schtasks job ==="
schtasks /Delete /TN "wikiarts15_sdturbo_samam" /F 2>&1 | Out-String | Write-Output

Write-Output ""
Write-Output "=== Verifying renames ==="
Write-Output "TEST_DIR exists wikiarts20: $(Test-Path 'I:\datasets\wikiarts20_512_test')"
Write-Output "baseline_wikiarts20: $(Test-Path 'I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20')"
Write-Output "wikiarts20_eval: $(Test-Path 'I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts20_eval')"

Write-Output ""
Write-Output "=== Listing exp/baseline_wikiarts20 ==="
$bw = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20"
if (Test-Path $bw) {
    Get-ChildItem $bw | ForEach-Object {
        $name = $_.Name
        $sub = Join-Path $bw $name
        $done = Test-Path (Join-Path $sub "_DONE")
        $imgs = if (Test-Path (Join-Path $sub "images")) { (Get-ChildItem (Join-Path $sub "images") -Filter "*.png" -ErrorAction SilentlyContinue).Count } else { 0 }
        $csv = Test-Path (Join-Path $sub "metrics.csv")
        Write-Output ("  {0,-12} done={1} imgs={2} csv={3}" -f $name, $done, $imgs, $csv)
    }
}

Write-Output ""
Write-Output "=== Listing exp/wikiarts20_eval ==="
$ev = "I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts20_eval"
if (Test-Path $ev) {
    Get-ChildItem $ev | ForEach-Object { Write-Output ("  " + $_.Name) }
}
