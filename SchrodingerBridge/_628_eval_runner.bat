@echo off
set PYTHON=C:\Progra~1\Python312\python.exe
set EVAL=I:\Github\Latent_Style\SchrodingerBridge\628_eval_x_batch.py
set STDOUT=I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\eval_runner_stdout.log
set STDERR=I:\Github\Latent_Style\SchrodingerBridge\exp\628_ablation\destructive_logs\eval_runner_stderr.log
cd /d I:\Github\Latent_Style\SchrodingerBridge
"%PYTHON%" "%EVAL%" > "%STDOUT%" 2> "%STDERR%"
