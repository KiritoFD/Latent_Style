@echo off
cd /d I:\GitHub\Latent_Style\SchrodingerBridge
"C:\Program Files\Python312\python.exe" -u tools\remote_master_baseline_v2.py > exp\baseline_v2\master_stdout.log 2> exp\baseline_v2\master_stderr.log
