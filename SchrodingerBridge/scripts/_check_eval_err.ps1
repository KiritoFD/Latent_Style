# Check actual eval error from X05 (last completed but eval-failed experiment)
$EXP_ROOT = "I:\Github\Latent_Style\SchrodingerBridge\exp\abl512"

Write-Host "=== X05 full_eval directory contents ==="
$X05_EVAL = "$EXP_ROOT\X05_corrector_4\full_eval"
if (Test-Path $X05_EVAL) {
    Get-ChildItem $X05_EVAL -Recurse | Select-Object FullName, Length, LastWriteTime | Format-Table -AutoSize
} else {
    Write-Host "No full_eval dir"
}

Write-Host ""
Write-Host "=== X05 epoch_0005.log (if exists) ==="
$X05_LOG = "$EXP_ROOT\X05_corrector_4\logs"
if (Test-Path $X05_LOG) {
    Get-ChildItem $X05_LOG | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
    $LATEST = Get-ChildItem "$X05_LOG\*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($LATEST) {
        Write-Host "Latest log: $($LATEST.Name) - last 30 lines:"
        Get-Content $LATEST.FullName -Tail 30
    }
}

Write-Host ""
Write-Host "=== X05 stderr (from train log .err) - look for eval errors ==="
$ERR_LOG = "I:\Github\Latent_Style\SchrodingerBridge\logs\abl512_v3_X05_corrector_4_train.log.err"
if (Test-Path $ERR_LOG) {
    Write-Host "Looking for 'Error'/'Traceback'/'requests' in $ERR_LOG"
    $matches = Select-String -Path $ERR_LOG -Pattern "Error|Traceback|requests|ModuleNotFound|FileNotFound|ImportError" -SimpleMatch
    if ($matches) {
        $matches | Select-Object -Last 20 | ForEach-Object { "$($_.LineNumber): $($_.Line)" }
    } else {
        Write-Host "No error matches found, showing last 30 lines:"
        Get-Content $ERR_LOG -Tail 30
    }
}

Write-Host ""
Write-Host "=== Check requests module availability ==="
& "C:\Program Files\Python312\python.exe" -c "import requests; print('requests OK:', requests.__version__, 'at', requests.__file__)"
$exit1 = $LASTEXITCODE
Write-Host "Exit code (Administrator): $exit1"

Write-Host ""
Write-Host "=== Check requests as SYSTEM (using psexec-like check) ==="
# Test if SYSTEM can find it - check the user site-packages path
$USER_SITE = "C:\Users\Administrator\AppData\Roaming\Python\Python312\site-packages"
Write-Host "User site exists: $(Test-Path $USER_SITE)"
if (Test-Path $USER_SITE) {
    Write-Host "requests in user site: $(Test-Path "$USER_SITE\requests")"
}
$SYS_SITE = "C:\Program Files\Python312\Lib\site-packages"
Write-Host "requests in system site: $(Test-Path "$SYS_SITE\requests")"
