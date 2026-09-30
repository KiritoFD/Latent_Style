@echo off
cd /d C:\Users\Administrator
set PYTHONIOENCODING=utf-8
python _monitor_and_chain_evals.py > C:\Users\Administrator\logs\_monitor_chain.out 2>&1
