@echo off
start /B wsl bash -c "nohup bash /mnt/c/Users/Administrator/run_eval_adain_wct_256.sh > /mnt/c/Users/Administrator/eval_adain_wct_256.log 2>&1 &"
echo started
