@echo off
REM Watchdog launcher (auto-chains baseline schtasks). Runs every 15 min via schtasks.
set PY="C:\Program Files\Python312\python.exe"
set SCRIPT=I:\GitHub\Latent_Style\SchrodingerBridge\tools\_watchdog.py
%PY% %SCRIPT% >> I:\GitHub\Latent_Style\SchrodingerBridge\exp\watchdog.log 2>&1
