@echo off
set PYTHONIOENCODING=utf-8
cd /d I:\Github\Latent_Style\SchrodingerBridge
powershell -ExecutionPolicy Bypass -File "I:\Github\Latent_Style\SchrodingerBridge\scripts\_run_s1_single.ps1" -RunName "s4_amp20" -OverrideConfig "I:\Github\Latent_Style\SchrodingerBridge\configs\_s4_overrides\s4_amp20_wct_ll05.json" > "I:\Github\Latent_Style\SchrodingerBridge\logs\s4_amp20.log" 2>&1
