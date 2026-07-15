@echo off
cd /d C:\Users\Administrator\SchrodingerBridge\src
set PYTHONPATH=C:\Users\Administrator\SchrodingerBridge\src
python run.py > C:\Users\Administrator\logs\brk_a_15ep_train.log 2>&1
echo TRAIN_DONE >> C:\Users\Administrator\logs\brk_a_15ep_train.log
