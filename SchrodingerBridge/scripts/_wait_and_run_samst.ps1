# Wait for SaMam python process to finish, then launch SaMST
$log = 'C:\Users\Administrator\_wait_and_run_samst.log'
"=== Wait script started: $(Get-Date) ===" | Out-File $log

# Wait for samam python to exit (max 60 min)
$waited = 0
while ($waited -lt 3600) {
    $py = Get-Process python -ErrorAction SilentlyContinue
    if (-not $py) {
        "[$(Get-Date)] No python process running. SaMam done." | Out-File $log -Append
        break
    }
    Start-Sleep -Seconds 30
    $waited += 30
    if ($waited % 300 -eq 0) {
        $imgDir = 'I:\Github\Latent_Style\exp_samam\training\samam_distinct5_512_scratch_7k_250eval_remote\curve_eval_30src\step_020000\images'
        $cnt = 0
        if (Test-Path $imgDir) { $cnt = (Get-ChildItem $imgDir).Count }
        "[$(Get-Date)] Waiting... imgs=$cnt, waited=${waited}s" | Out-File $log -Append
    }
}

# Show samam output
"[$(Get-Date)] === SaMam output dir ===" | Out-File $log -Append
$outDir = 'I:\Github\Latent_Style\exp_samam\_dino_curve_repro'
if (Test-Path $outDir) {
    Get-ChildItem $outDir | Out-File $log -Append
    $csv = Join-Path $outDir 'samam_curve_repro.csv'
    if (Test-Path $csv) {
        "[$(Get-Date)] === samam_curve_repro.csv ===" | Out-File $log -Append
        Get-Content $csv | Out-File $log -Append
    }
} else {
    "[$(Get-Date)] samam output dir missing" | Out-File $log -Append
}

# Launch SaMST (use cmd /c for reliable UTF-8 redirection)
"[$(Get-Date)] === Launching SaMST ===" | Out-File $log -Append
Set-Location 'C:\Users\Administrator'
$env:PYTHONPATH = 'C:\Users\Administrator\samst_repo'
cmd /c "python C:\Users\Administrator\_samst_curve_repro.py > C:\Users\Administrator\_samst_curve_repro.log 2>&1"
"[$(Get-Date)] === SaMST finished (exit=$LASTEXITCODE) ===" | Out-File $log -Append

# Show samst output
$samstCsv = 'I:\Github\Latent_Style\exp_samam\_dino_curve_repro\samst_curve_repro.csv'
if (Test-Path $samstCsv) {
    "[$(Get-Date)] === samst_curve_repro.csv ===" | Out-File $log -Append
    Get-Content $samstCsv | Out-File $log -Append
}
"=== Wait script done: $(Get-Date) ===" | Out-File $log -Append
