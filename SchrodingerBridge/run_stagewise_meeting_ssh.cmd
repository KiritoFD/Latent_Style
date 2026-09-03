@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
set LANCET_BATCH_SIZE=160
set LANCET_EVAL_BATCH_SIZE=8
set PYTHONPATH=I:\Github\Latent_Style\SchrodingerBridge\src
if not exist logs mkdir logs
"C:\Program Files\Python312\python.exe" tools\experiments\run_stagewise_meeting.py > logs\stagewise_meeting_remote.log 2>&1
