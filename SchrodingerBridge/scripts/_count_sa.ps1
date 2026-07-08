Write-Output "=== D5 ==="
if (Test-Path "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_stylealigned\distinct5\images") {
    (Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_stylealigned\distinct5\images" -File).Count
} else { Write-Output "not exists" }

Write-Output "=== P2A ==="
if (Test-Path "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_stylealigned\photo2art256\images") {
    (Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_stylealigned\photo2art256\images" -File).Count
} else { Write-Output "not exists" }

Write-Output "=== R5 ==="
if (Test-Path "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_stylealigned\random5\images") {
    (Get-ChildItem "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_stylealigned\random5\images" -File).Count
} else { Write-Output "not exists" }

Write-Output "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

Write-Output "=== SA Process ==="
gwmi Win32_Process -Filter "name='python.exe'" | Select-Object ProcessId,CommandLine | Format-List
