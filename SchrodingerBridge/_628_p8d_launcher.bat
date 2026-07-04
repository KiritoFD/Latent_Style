@echo off
REM Phase 8D launcher: color_match deep exploration (train + eval + rescan)
set PYTHON=C:\Progra~1\Python312\python.exe
set ROOT=I:\Github\Latent_Style\SchrodingerBridge
set LOGDIR=%ROOT%\exp\628_ablation\destructive_logs
mkdir "%LOGDIR%" 2>nul

echo [%date% %time%] === Phase 8D START === > "%LOGDIR%\p8d_launcher.log"

echo [%date% %time%] Step 1: Training batch (D1-D12) >> "%LOGDIR%\p8d_launcher.log"
cd /d "%ROOT%"
"%PYTHON%" "%ROOT%\628_run_destructive_batch.py" >> "%LOGDIR%\p8d_launcher.log" 2>&1

echo [%date% %time%] Step 2: Evaluation batch (D1-D12) >> "%LOGDIR%\p8d_launcher.log"
"%PYTHON%" "%ROOT%\628_eval_all_batch.py" >> "%LOGDIR%\p8d_launcher.log" 2>&1

echo [%date% %time%] Step 3: Rescan all metrics >> "%LOGDIR%\p8d_launcher.log"
"%PYTHON%" "%ROOT%\_628_p8c_rescan_all_metrics.py" >> "%LOGDIR%\p8d_launcher.log" 2>&1

echo [%date% %time%] === Phase 8D END === >> "%LOGDIR%\p8d_launcher.log"
