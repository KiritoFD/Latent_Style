$imgDir = 'I:\Github\Latent_Style\exp_samam\training\samam_distinct5_512_scratch_7k_250eval_remote\curve_eval_30src\step_020000\images'
if (Test-Path $imgDir) {
    $cnt = (Get-ChildItem $imgDir).Count
    Write-Host "imgs_step20k: $cnt"
} else {
    Write-Host "step20k img dir missing"
}
$logPath = 'C:\Users\Administrator\_samam_curve_repro.log'
if (Test-Path $logPath) {
    Write-Host "--- last 20 log lines ---"
    Get-Content $logPath -Tail 20
} else {
    Write-Host "no samam log"
}
$outDir = 'I:\Github\Latent_Style\exp_samam\_dino_curve_repro'
if (Test-Path $outDir) {
    Write-Host "--- output dir ---"
    Get-ChildItem $outDir
}
