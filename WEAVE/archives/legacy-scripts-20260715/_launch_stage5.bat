@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
powershell -ExecutionPolicy Bypass -File scripts\run_sty_multipoint_eval.ps1 > C:\Users\Administrator\logs\sty_inject\stage5_main.out 2>&1
