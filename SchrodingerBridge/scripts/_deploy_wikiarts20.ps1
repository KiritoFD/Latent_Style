# Deploy wikiarts-20 scripts from /tmp to repo scripts dir, delete old wikiarts15 scripts, then create+run schtasks
$repo = "I:\Github\Latent_Style\SchrodingerBridge"
$scriptsDir = "$repo\scripts"
$tmpDir = "C:\Users\Administrator"

# Ensure scripts dir exists
if (-not (Test-Path $scriptsDir)) { New-Item -ItemType Directory -Force -Path $scriptsDir | Out-Null }

# Copy new wikiarts20 scripts from /tmp to repo scripts dir
$files = @(
    "gen_trainfree_wikiarts20.py",
    "_eval_baselines_wikiarts20.ps1",
    "_eval_wikiarts20.ps1",
    "_run_wikiarts20_all.ps1",
    "_create_wikiarts20_task.ps1"
)
foreach ($f in $files) {
    $src = "$tmpDir\$f"
    $dst = "$scriptsDir\$f"
    if (Test-Path $src) {
        Copy-Item $src $dst -Force
        Write-Output "  deployed: $f"
    } else {
        Write-Output "  MISSING: $src"
    }
}

# Delete any remaining wikiarts15 scripts (already renamed on remote, but double-check)
Get-ChildItem $scriptsDir -File | Where-Object { $_.Name -like "*wikiarts15*" } | ForEach-Object {
    Remove-Item $_.FullName -Force
    Write-Output "  deleted old: $($_.Name)"
}

Write-Output ""
Write-Output "=== Scripts in repo scripts dir (wikiarts-related) ==="
Get-ChildItem $scriptsDir -File | Where-Object { $_.Name -like "*wikiarts*" } | ForEach-Object {
    Write-Output ("  {0,-50} {1}" -f $_.Name, $_.LastWriteTime)
}

# Verify the WD-VF checkpoint still exists in renamed dir
$ckpt = "$repo\exp\wikiarts20_eval\epoch_0005.pt"
$config = "$repo\exp\wikiarts20_eval\config.json"
Write-Output ""
Write-Output "=== WD-VF checkpoint verification ==="
Write-Output "  checkpoint exists: $(Test-Path $ckpt)"
Write-Output "  config exists:     $(Test-Path $config)"
if (Test-Path $ckpt) {
    $sz = (Get-Item $ckpt).Length / 1MB
    Write-Output ("  checkpoint size:   {0:N2} MB" -f $sz)
}

# Verify test dir
$testDir = "I:\datasets\wikiarts20_512_test"
Write-Output ""
Write-Output "=== Test dir verification ==="
Write-Output "  test_dir exists: $(Test-Path $testDir)"
if (Test-Path $testDir) {
    $dirCount = (Get-ChildItem $testDir -Directory).Count
    Write-Output "  style dirs: $dirCount"
}

# Verify VGG/decoder weights for AdaIN/WCT
$modelsDir = "I:\Github\Latent_Style\Related_Works\repos\pytorch-AdaIN\models"
Write-Output ""
Write-Output "=== AdaIN/WCT models ==="
if (Test-Path $modelsDir) {
    Get-ChildItem $modelsDir -File | ForEach-Object { Write-Output ("  {0,-30} {1} bytes" -f $_.Name, $_.Length) }
} else {
    Write-Output "  MODELS DIR NOT FOUND: $modelsDir"
}

# Create and run schtasks job
Write-Output ""
Write-Output "=== Creating schtasks job: wikiarts20_all ==="
$createScript = "$scriptsDir\_create_wikiarts20_task.ps1"
& powershell -ExecutionPolicy Bypass -File $createScript
