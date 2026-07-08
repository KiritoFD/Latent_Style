$ErrorActionPreference = 'Continue'

Write-Host "=== 1. StyleAligned D5 done? ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_stylealigned_distinct5\images (dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_stylealigned_distinct5\images\*.png 2>nul | find /C /V "") else (echo NOT FOUND)"
Write-Host "SA D5 images: $ssh_out"

Write-Host ""
Write-Host "=== 2. StyleAligned P2A done? ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_stylealigned\p2a_256\images (dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_stylealigned\p2a_256\images\*.png 2>nul | find /C /V "") else (echo NOT FOUND)"
Write-Host "SA P2A images: $ssh_out"

Write-Host ""
Write-Host "=== 3. StyleAligned R5 done? ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_stylealigned\r5_wikiart\images (dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_stylealigned\r5_wikiart\images\*.png 2>nul | find /C /V "") else (echo NOT FOUND)"
Write-Host "SA R5 images: $ssh_out"

Write-Host ""
Write-Host "=== 4. _run_stylealigned_remote.py on remote? ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist C:\Users\Administrator\_run_stylealigned_remote.py (echo EXISTS) else (echo NOT FOUND)"
Write-Host "SA remote script: $ssh_out"

Write-Host ""
Write-Host "=== 5. style_aligned module on remote? ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist C:\Users\Administrator\style_aligned\sa_handler_sd15.py (echo EXISTS) else (echo NOT FOUND)"
Write-Host "SA module: $ssh_out"

Write-Host ""
Write-Host "=== 6. Check SD1.5 model cache ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist C:\Users\Administrator\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5 (echo EXISTS) else (echo NOT FOUND)"
Write-Host "SD1.5 cache: $ssh_out"

Write-Host ""
Write-Host "=== 7. D5-512 s2wat (what is this?) ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "dir /B I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\s2wat 2>nul | head -3"
Write-Host "s2wat: $ssh_out"

Write-Host ""
Write-Host "=== 8. Search for ZSTAR/StyleShot code on remote ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist C:\Users\Administrator\ZSTAR (echo EXISTS) else (echo NOT FOUND)"
Write-Host "ZSTAR: $ssh_out"
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist C:\Users\Administrator\StyleShot (echo EXISTS) else (echo NOT FOUND)"
Write-Host "StyleShot: $ssh_out"

Write-Host ""
Write-Host "=== 9. GPU availability ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
Write-Host "GPU: $ssh_out"

Write-Host ""
Write-Host "=== 10. Check D5 SA eval results ==="
$ssh_out = ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62 "if exist I:\Github\Latent_Style\SchrodingerBridge\exp\_eval_stylealigned_d5.json (type I:\Github\Latent_Style\SchrodingerBridge\exp\_eval_stylealigned_d5.json) else (echo NOT FOUND)"
Write-Host "SA D5 eval: $ssh_out"
