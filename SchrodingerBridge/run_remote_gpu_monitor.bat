@echo off
setlocal
cd /d "%~dp0"
if exist "tools\remote_gpu_monitor_native.hta" (
    echo Launching Remote GPU Monitor Native GUI...
    start "" mshta.exe "%~dp0tools\remote_gpu_monitor_native.hta"
) else (
    echo Launching Python GPU Monitor...
    python tools\gpu.py
)
