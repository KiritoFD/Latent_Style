@echo off
REM Launcher for 630 remote batch runner - starts in background
cd /d I:\Github\Latent_Style\SchrodingerBridge
start "630_batch" /min powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Administrator\_remote_batch_runner.ps1"
echo Batch runner started. PID check: tasklist /fi "WINDOWTITLE eq 630_batch"
