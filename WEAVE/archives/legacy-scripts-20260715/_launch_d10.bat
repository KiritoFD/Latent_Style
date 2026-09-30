@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
set PYTHONPATH=.
powershell -NoProfile -ExecutionPolicy Bypass -File "I:\Github\Latent_Style\SchrodingerBridge\scripts\_run_d10_batch.ps1" > "C:\Users\Administrator\logs\d10_batch_stdout.out" 2> "C:\Users\Administrator\logs\d10_batch_stderr.out"
