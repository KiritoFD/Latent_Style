@echo off
REM StyleAligned P2A+R5 inference launcher for schtasks (survives SSH disconnect)
set PY="C:\Program Files\Python312\python.exe"
set SCRIPT=I:\GitHub\Latent_Style\SchrodingerBridge\tools\_run_stylealigned_remote.py
set LOG=I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_stylealigned\stylealigned.log
echo [%date% %time%] StyleAligned LAUNCH >> %LOG%
%PY% %SCRIPT% >> %LOG% 2>&1
echo [%date% %time%] StyleAligned DONE exit=%errorlevel% >> %LOG%
