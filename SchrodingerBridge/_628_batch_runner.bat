@echo off
set PYTHON=C:\Progra~1\Python312\python.exe
set RUNNER=I:\Github\Latent_Style\SchrodingerBridge\628_run_destructive_batch.py
set LOGDIR=I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set STDOUT=%LOGDIR%\batch_runner_stdout.log
set STDERR=%LOGDIR%\batch_runner_stderr.log
cd /d I:\Github\Latent_Style\SchrodingerBridge
"%PYTHON%" "%RUNNER%" > "%STDOUT%" 2> "%STDERR%"