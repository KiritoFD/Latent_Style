@echo off
set PYTHON=C:\Progra~1\Python312\python.exe
set RUNNER=I:\Github\Latent_Style\SchrodingerBridge\628_run_destructive_batch.py
set STDOUT=I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\batch_runner_stdout.log
set STDERR=I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\batch_runner_stderr.log
cd /d I:\Github\Latent_Style\SchrodingerBridge
"%PYTHON%" "%RUNNER%" > "%STDOUT%" 2> "%STDERR%"