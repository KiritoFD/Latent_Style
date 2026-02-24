$ErrorActionPreference = "Stop"

# Always run from Cycle-NCE/src
Set-Location $PSScriptRoot
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

Write-Host "=== Python / CUDA sanity check ==="
uv run --active python -c "import sys,torch; print('python=',sys.executable); print('torch=',torch.__version__); print('torch.cuda=',torch.version.cuda); print('cuda_available=',torch.cuda.is_available())"
Write-Host "=================================="

uv run --active python run.py --config ..\experiments\20260223-micro5style-bs384\runs\exp00-phase1_60e-baseline_bs384_lr2e4_bd64_sd256\config.json
uv run --active python run.py --config ..\experiments\20260223-micro5style-bs384\runs\exp01-phase1_60e-capacity_floor_bd32_lc32_sd128\config.json
uv run --active python run.py --config ..\experiments\20260223-micro5style-bs384\runs\exp02-phase1_60e-capacity_ceiling_bd128_lc128_sd256_bs256\config.json
uv run --active python run.py --config ..\experiments\20260223-micro5style-bs384\runs\exp03-phase1_60e-style_ctrl_sd512_bs288\config.json
uv run --active python run.py --config ..\experiments\20260223-micro5style-bs384\runs\exp04-phase1_60e-lr4e4_speedup\config.json
uv run --active python run.py --config ..\experiments\20260223-micro5style-bs384\runs\exp05-phase1_60e-lr8e4_stress\config.json
uv run --active python run.py --config ..\experiments\20260223-micro5style-bs384\runs\exp06-phase1_60e-l1_relax_wdl1_0p01\config.json
uv run --active python run.py --config ..\experiments\20260223-micro5style-bs384\runs\exp07-phase1_60e-swd_patch_1_3\config.json
uv run --active python run.py --config ..\experiments\20260223-micro5style-bs384\runs\exp08-phase2_150e-deep_cosine_lr5e4_min1e6\config.json
uv run --active python run.py --config ..\experiments\20260223-micro5style-bs384\runs\exp09-phase2_150e-flat_lr3e4\config.json
