# Stop current training and fix requests module for SYSTEM account
$REPO = "I:\Github\Latent_Style\SchrodingerBridge"
$PYTHON = "C:\Program Files\Python312\python.exe"

Write-Host "=== Step 1: Stop current training ==="
# Kill any python running run.py
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Killing python PID $($_.Id) (StartTime: $($_.StartTime))"
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}
# Kill any powershell running batch script
Get-Process powershell -ErrorAction SilentlyContinue | ForEach-Object {
    $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine
    if ($cmd -match "run_abl512_v3|_watchdog_abl512") {
        Write-Host "Killing powershell PID $($_.Id): $cmd"
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "=== Step 2: Verify no python running ==="
$p = Get-Process python -ErrorAction SilentlyContinue
if ($p) {
    $p | Select-Object Id, StartTime | Format-Table
} else {
    Write-Host "No python process running"
}

Write-Host ""
Write-Host "=== Step 3: Install requests to system site-packages ==="
# Install requests (and deps) to system-wide site-packages, not --user
& $PYTHON -m pip install --upgrade requests 2>&1 | Tee-Object -Variable PIP_OUT
Write-Host "pip exit code: $LASTEXITCODE"

Write-Host ""
Write-Host "=== Step 4: Verify requests in system site-packages ==="
& $PYTHON -c "import requests, os; print('requests version:', requests.__version__); print('location:', os.path.dirname(os.path.dirname(requests.__file__)))"

Write-Host ""
Write-Host "=== Step 5: Also install other common deps that artfid_metric.py might need ==="
# Check what artfid_metric.py imports
$ARTFID = "$REPO\src\utils\artfid_metric.py"
if (Test-Path $ARTFID) {
    Write-Host "artfid_metric.py imports:"
    Select-String -Path $ARTFID -Pattern "^(import|from) " | ForEach-Object { "  $($_.Line.Trim())" }
}

Write-Host ""
Write-Host "=== Step 6: Check run_evaluation.py imports ==="
$RUN_EVAL = "$REPO\src\utils\run_evaluation.py"
if (Test-Path $RUN_EVAL) {
    Write-Host "run_evaluation.py imports (first 20):"
    Select-String -Path $RUN_EVAL -Pattern "^(import|from) " | Select-Object -First 20 | ForEach-Object { "  $($_.Line.Trim())" }
}

Write-Host ""
Write-Host "=== Step 7: Test import as if from src directory ==="
Push-Location "$REPO\src"
& $PYTHON -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'C:/Users/Administrator/AppData/Roaming/Python/Python312/site-packages')
try:
    from utils.artfid_metric import ArtFIDMetric
    print('artfid_metric import: OK')
except Exception as e:
    print(f'artfid_metric import FAILED: {type(e).__name__}: {e}')
"
$exit_code = $LASTEXITCODE
Write-Host "Test exit code: $exit_code"
Pop-Location
