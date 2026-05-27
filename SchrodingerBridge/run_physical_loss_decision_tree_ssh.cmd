@echo off
setlocal
cd /d I:\Github\Latent_Style\SchrodingerBridge
set PYTHONPATH=src
set LANCET_BATCH_SIZE=192
set LANCET_EVAL_BATCH_SIZE=20
python tools\experiments\run_physical_loss_decision_tree.py --max-total 64 --force-train
