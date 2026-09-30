# Check Python environments and find correct one
Write-Host "=== Python locations ==="
$pyPaths = @(
    "C:\Program Files\Python312\python.exe",
    "I:\Github\Latent_Style\WEAVE\.conda\python.exe",
    "C:\Users\Administrator\anaconda3\python.exe",
    "C:\Users\Administrator\miniconda3\python.exe"
)
foreach ($p in $pyPaths) {
    if (Test-Path $p) {
        Write-Host ""
        Write-Host "FOUND: $p"
        & $p -c "import sys; print('  exec:', sys.executable); print('  version:', sys.version.split()[0])"
        & $p -c "import torch; print('  torch:', torch.__version__); print('  cuda available:', torch.cuda.is_available()); print('  cuda version:', torch.version.cuda); print('  device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1
    }
}

Write-Host ""
Write-Host "=== Conda envs ==="
$condaPaths = @(
    "C:\Users\Administrator\anaconda3\Scripts\conda.exe",
    "C:\Users\Administrator\miniconda3\Scripts\conda.exe"
)
foreach ($c in $condaPaths) {
    if (Test-Path $c) {
        Write-Host "Conda: $c"
        & $c env list 2>&1
    }
}

Write-Host ""
Write-Host "=== Looking for python with torch in PATH ==="
where.exe python 2>&1

Write-Host ""
Write-Host "=== Check WEAVE conda env ==="
if (Test-Path "I:\Github\Latent_Style\WEAVE\.conda") {
    Write-Host "WEAVE .conda exists"
    $condaPython = "I:\Github\Latent_Style\WEAVE\.conda\python.exe"
    if (Test-Path $condaPython) {
        & $condaPython -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1
    }
}

Write-Host ""
Write-Host "=== Look for any python.exe with torch ==="
Get-ChildItem -Path "C:\Users\Administrator" -Filter "python.exe" -Recurse -ErrorAction SilentlyContinue -Depth 4 | Select-Object -First 10 | ForEach-Object {
    Write-Host $_.FullName
}
