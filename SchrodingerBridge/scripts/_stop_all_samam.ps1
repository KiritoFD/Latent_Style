# Stop SaMam W20 v2 and all related processes
$ErrorActionPreference = "Continue"

Write-Host "=== Stopping all python/powershell processes ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='powershell.exe'" |
    Where-Object {
        $_.CommandLine -like "*_gen_samam*" -or
        $_.CommandLine -like "*samam_w20*" -or
        $_.CommandLine -like "*_master_pipeline*" -or
        $_.CommandLine -like "*_eval_all_unified*"
    } |
    ForEach-Object {
        Write-Host "Killing PID $($_.ProcessId) ($($_.Name))"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

Write-Host ""
Write-Host "=== Disabling schtasks ==="
schtasks /End /TN "samam_w20_v2" 2>$null
schtasks /Change /TN "samam_w20_v2" /Disable 2>$null
schtasks /Delete /TN "samam_w20_v2" /F 2>$null
schtasks /End /TN "master_pipeline" 2>$null
schtasks /Delete /TN "master_pipeline" /F 2>$null

Start-Sleep -Seconds 3

Write-Host ""
Write-Host "=== VRAM after cleanup ==="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

Write-Host ""
Write-Host "=== Remaining processes ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='powershell.exe'" |
    Select-Object ProcessId, Name, @{N='Start';E={$_.CreationDate}} |
    Format-Table -Auto

Write-Host ""
Write-Host "=== samam_w20 images count ==="
$imgDir = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\samam\images"
if (Test-Path $imgDir) {
    $cnt = (Get-ChildItem $imgDir -File).Count
    Write-Host "samam images: $cnt"
} else {
    Write-Host "no samam images dir"
}

Write-Host ""
Write-Host "=== baseline_wikiarts20 subdirs ==="
$bw = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20"
if (Test-Path $bw) {
    Get-ChildItem $bw -Directory | ForEach-Object {
        $imgs = Join-Path $_.FullName "images"
        $cnt = 0
        if (Test-Path $imgs) {
            $cnt = (Get-ChildItem $imgs -File -ErrorAction SilentlyContinue | Measure-Object).Count
        }
        Write-Host ("{0}: {1} files" -f $_.Name, $cnt)
    }
}

Write-Host ""
Write-Host "=== WikiArt-20 test dir styles ==="
$testDir = "I:\datasets\wikiarts20_512_test"
if (Test-Path $testDir) {
    Get-ChildItem $testDir -Directory | Select-Object Name | Format-Table -Auto
}
