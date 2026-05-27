@echo off
cd /d I:\Github\Latent_Style\SchrodingerBridge
set LANCET_BATCH_SIZE=192
set LANCET_EVAL_BATCH_SIZE=20
set PYTHONPATH=I:\Github\Latent_Style\SchrodingerBridge\src
python tools\experiments\run_color_guard_sweep.py --force-train --force-eval > exp\color_guard_sweep\color_guard_remote.log 2> exp\color_guard_sweep\color_guard_remote.err
