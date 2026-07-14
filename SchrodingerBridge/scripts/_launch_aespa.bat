@echo off
:: Launch AesPA-Net inference on all 3 datasets via schtasks
setlocal
set PYTHON=C:\Program Files\Python312\python.exe
set AESPA_ROOT=I:\AesPA-Net
set LOG=I:\AesPA-Net\aespa_run.log

echo === AesPA-Net inference start: %DATE% %TIME% === > "%LOG%"
cd /d "%AESPA_ROOT%"

powershell -ExecutionPolicy Bypass -File "%AESPA_ROOT%\run_aespa_all.ps1" >> "%LOG%" 2>&1

echo === AesPA-Net inference end: %DATE% %TIME% === >> "%LOG%"
endlocal
