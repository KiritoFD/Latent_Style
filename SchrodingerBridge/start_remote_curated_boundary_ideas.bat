@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
set LANCET_BATCH_SIZE=192
set LANCET_EVAL_BATCH_SIZE=20
set PYTHONPATH=I:\Github\Latent_Style\SchrodingerBridge\src
if not exist logs mkdir logs
python tools\experiments\run_curated_boundary_ideas.py --max-total 40 --force-train --force-eval > logs\curated_boundary_ideas_remote.log 2>&1
