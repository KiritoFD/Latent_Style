@echo off
REM Phase 8B launcher: fine-grained num_steps sweep
set PYTHON=C:\Progra~1\Python312\python.exe
set ROOT=I:\Github\Latent_Style\SchrodingerBridge
set LOGDIR=%ROOT%\exp\628_ablation\p8b_steps_fine
mkdir "%LOGDIR%" 2>nul

echo [%date% %time%] === Phase 8B START === > "%LOGDIR%\p8b_launcher.log"
cd /d "%ROOT%"
"%PYTHON%" "%ROOT%\_628_p8b_steps_fine_sweep.py" >> "%LOGDIR%\p8b_launcher.log" 2>&1
echo [%date% %time%] === Phase 8B END rc=%errorlevel% >> "%LOGDIR%\p8b_launcher.log"
