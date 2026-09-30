@echo off
cd /d I:\GitHub\Latent_Style\SchrodingerBridge
"C:\Program Files\Python312\python.exe" -u tools\remote_sdturbo_fixed.py > exp\baseline_v2\sdturbo_stdout.log 2> exp\baseline_v2\sdturbo_stderr.log
