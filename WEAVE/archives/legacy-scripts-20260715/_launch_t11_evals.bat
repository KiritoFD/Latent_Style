@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
set PYTHONPATH=.
set PYTHONIOENCODING=utf-8
powershell -NoProfile -ExecutionPolicy Bypass -File "I:\Github\Latent_Style\SchrodingerBridge\scripts\_run_t11_evals.ps1" > C:\Users\Administrator\logs\t11_evals_stdout.out 2>&1
