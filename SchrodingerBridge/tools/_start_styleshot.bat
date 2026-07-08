@echo off
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python C:\Users\Administrator\_run_styleshot_remote.py > C:\Users\Administrator\logs\styleshot_inference.log 2>&1
