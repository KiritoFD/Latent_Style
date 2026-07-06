# Install missing Python packages for full_eval
$PYTHON = "C:\Program Files\Python312\python.exe"
Write-Host "=== Installing missing packages ==="
& $PYTHON -m pip install requests
Write-Host ""
Write-Host "=== Verify ==="
& $PYTHON -c "import requests; print('requests version:', requests.__version__)"
