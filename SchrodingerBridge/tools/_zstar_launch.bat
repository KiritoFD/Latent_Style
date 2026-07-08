@echo off
REM Z-STAR (CVPR2024) inference launcher for schtasks — survives SSH disconnect
set PY="C:\Program Files\Python312\python.exe"
set SCRIPT=I:\GitHub\Latent_Style\SchrodingerBridge\tools\_run_zstar_remote.py
set LOG=I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_zstar\zstar.log
echo [%date% %time%] Z-STAR LAUNCH %* >> %LOG%
%PY% %SCRIPT% %* >> %LOG% 2>&1
echo [%date% %time%] Z-STAR DONE exit=%errorlevel% >> %LOG%
