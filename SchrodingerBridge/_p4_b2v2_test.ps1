# Test script: verify env vars and python execution
$env:P4_CKPT_PATH = "I:/Github/Latent_Style/SchrodingerBridge/exp/620_spectral_v2_weights/epoch_0001.pt"
$env:P4_CONFIG_PATH = "I:/Github/Latent_Style/SchrodingerBridge/configs/620_spectral_v2_weights.json"
$env:P4_BASELINE_CLIP = "0.6731"
$env:P4_BASELINE_LPIPS = "0.2781"
Set-Location I:/Github/Latent_Style/SchrodingerBridge
Write-Host "=== Test 1: env vars ==="
Write-Host "P4_CKPT_PATH = $env:P4_CKPT_PATH"
Write-Host "P4_CONFIG_PATH = $env:P4_CONFIG_PATH"
Write-Host "=== Test 2: python version ==="
& "C:/Program Files/Python312/python.exe" --version
Write-Host "=== Test 3: python env access ==="
$pyCode = @'
import os, sys
print("python ok, version:", sys.version[:30])
print("P4_CKPT_PATH =", os.environ.get("P4_CKPT_PATH", "MISSING"))
print("P4_CONFIG_PATH =", os.environ.get("P4_CONFIG_PATH", "MISSING"))
ckpt = os.environ.get("P4_CKPT_PATH", "")
import os.path
print("ckpt exists:", os.path.isfile(ckpt))
cfg = os.environ.get("P4_CONFIG_PATH", "")
print("cfg exists:", os.path.isfile(cfg))
'@
$pyCode | & "C:/Program Files/Python312/python.exe" -
Write-Host "=== Test 4: import ablation script deps ==="
$pyCode2 = @'
import sys
sys.path.insert(0, "I:/Github/Latent_Style/SchrodingerBridge/src")
try:
    import importlib
    run_module = importlib.import_module("run")
    config_schema = importlib.import_module("config_schema")
    print("imports OK")
except Exception as e:
    print("import error:", e)
'@
$pyCode2 | & "C:/Program Files/Python312/python.exe" -
Write-Host "=== Done ==="
