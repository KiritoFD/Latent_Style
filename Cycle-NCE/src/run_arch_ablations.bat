@echo off
setlocal

echo [Architecture Ablation] Running abl_macro_decoder...
uv run run.py --config config_abl_macro_decoder.json

echo [Architecture Ablation] Running abl_no_adagn...
uv run run.py --config config_abl_no_adagn.json

echo [Architecture Ablation] Running abl_naive_skip...
uv run run.py --config config_abl_naive_skip.json

echo [Architecture Ablation] Running abl_no_residual...
uv run run.py --config config_abl_no_residual.json

