$imgDir = 'I:\Github\Latent_Style\exp_samam\training\samam_distinct5_512_scratch_7k_250eval_remote\curve_eval_30src\step_020000\images'
if (Test-Path $imgDir) {
    $cnt = (Get-ChildItem $imgDir).Count
    Write-Host "imgs_step20k: $cnt"
} else {
    Write-Host "step20k img dir missing"
}
Write-Host "--- processes ---"
Get-Process python,powershell -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,WS | Format-Table -AutoSize
Write-Host "--- wait log ---"
$waitLog = 'C:\Users\Administrator\_wait_and_run_samst.log'
if (Test-Path $waitLog) {
    Get-Content $waitLog
} else {
    Write-Host "no wait log"
}
Write-Host "--- samam log tail ---"
$samamLog = 'C:\Users\Administrator\_samam_curve_repro.log'
if (Test-Path $samamLog) {
    Get-Content $samamLog -Tail 10
}
Write-Host "--- samam output dir ---"
$outDir = 'I:\Github\Latent_Style\exp_samam\_dino_curve_repro'
if (Test-Path $outDir) {
    Get-ChildItem $outDir
} else {
    Write-Host "no output dir yet"
}
