@echo off
REM Phase 8F launcher: orthogonal ablation
set PYTHON=C:\Progra~1\Python312\python.exe
set ROOT=I:\Github\Latent_Style\SchrodingerBridge
set LOGDIR=%ROOT%\exp\628_ablation\orthogonal_logs
mkdir "%LOGDIR%" 2>nul
echo [%date% %time%] === Phase 8F START === > "%LOGDIR%\p8f_launcher.log"
cd /d "%ROOT%"
"%PYTHON%" "%ROOT%\_628_p8f_orthogonal_runner.py" >> "%LOGDIR%\p8f_launcher.log" 2>&1
echo [%date% %time%] === Phase 8F END === >> "%LOGDIR%\p8f_launcher.log"
