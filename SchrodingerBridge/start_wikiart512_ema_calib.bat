@echo off
setlocal
cd /d I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge
set PYTHONPATH=I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\src
if not exist exp\wikiart512_ema_direct_atom_residual_calib_b16 mkdir exp\wikiart512_ema_direct_atom_residual_calib_b16
py -3 src\run.py --config configs\wikiart512_ema_direct_atom_residual_calib.json > exp\wikiart512_ema_direct_atom_residual_calib_b16\train_console.log 2>&1
endlocal
