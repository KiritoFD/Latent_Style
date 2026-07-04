@echo off
REM Phase 7 runner: D0_control training -> inference ablations #1-#10 -> #11-#12
REM Sequential to avoid GPU contention. Total ~2h.
set PYTHON=C:\Progra~1\Python312\python.exe
set ROOT=I:\Github\Latent_Style\SchrodingerBridge
set LOGDIR=%ROOT%\exp\628_ablation\destructive_logs
mkdir "%LOGDIR%" 2>nul

echo [%date% %time%] === Phase 7 runner START === > "%LOGDIR%\p7_runner.log"

REM === Stage 1: D0_control training (Phase 7A) ===
echo [%date% %time%] Stage 1: D0_control training >> "%LOGDIR%\p7_runner.log"
cd /d "%ROOT%"
"%PYTHON%" "%ROOT%\628_run_destructive_batch.py" >> "%LOGDIR%\p7_runner.log" 2>&1
echo [%date% %time%] Stage 1 exit code %errorlevel% >> "%LOGDIR%\p7_runner.log"

REM === Stage 2: Inference ablations #1-#10 (Phase 7C) ===
echo [%date% %time%] Stage 2: inference ablations #1-#10 >> "%LOGDIR%\p7_runner.log"
"%PYTHON%" "%ROOT%\_628_infer_ablations_p7.py" >> "%LOGDIR%\p7_runner.log" 2>&1
echo [%date% %time%] Stage 2 exit code %errorlevel% >> "%LOGDIR%\p7_runner.log"

REM === Stage 3: Inference ablations #11-#12 (num_steps + style_strength) ===
echo [%date% %time%] Stage 3: inference ablations #11-#12 >> "%LOGDIR%\p7_runner.log"
"%PYTHON%" "%ROOT%\_628_infer_steps_strength_p7.py" >> "%LOGDIR%\p7_runner.log" 2>&1
echo [%date% %time%] Stage 3 exit code %errorlevel% >> "%LOGDIR%\p7_runner.log"

echo [%date% %time%] === Phase 7 runner END === >> "%LOGDIR%\p7_runner.log"
