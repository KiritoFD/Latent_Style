@echo off
REM StyleShot (AAAI2025) inference launcher for schtasks — survives SSH disconnect
set PY="C:\Program Files\Python312\python.exe"
set SCRIPT=I:\GitHub\Latent_Style\SchrodingerBridge\tools\_run_styleshot_remote.py
set LOG=I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_styleshot\styleshot.log
echo [%date% %time%] StyleShot LAUNCH %* >> %LOG%
%PY% %SCRIPT% %* >> %LOG% 2>&1
echo [%date% %time%] StyleShot DONE exit=%errorlevel% >> %LOG%
