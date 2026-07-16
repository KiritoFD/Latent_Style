@echo off
cd /d I:\Github\Latent_Style\WEAVE
echo === Re-running seed42 eval (skips epochs 1-3, evaluates epoch 4) === > C:\Users\Administrator\logs\expA_seed42_fix.log 2>&1
"C:\Program Files\Python312\python.exe" -u scripts\expA_per_epoch_eval.py --run_dir runs\submission\hf_oriented_internal_early_stop --seed 42 >> C:\Users\Administrator\logs\expA_seed42_fix.log 2>&1
echo EXPA_SEED42_FIX_DONE >> C:\Users\Administrator\logs\expA_seed42_fix.log
echo EXPA_SEED42_FIX_DONE >> C:\Users\Administrator\logs\chain_after_seed7.log
