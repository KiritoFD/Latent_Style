@echo off
schtasks /create /tn latent256_train /tr "powershell -ExecutionPolicy Bypass -File scripts\run_latent256.ps1" /sc once /st 00:00 /f
schtasks /run /tn latent256_train
echo TASK_STARTED
