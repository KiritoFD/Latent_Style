@echo off
REM ====================================================================
REM Remote baseline environment probe (read-only, no modifications)
REM Server: administrator@100.115.18.62:2222
REM ====================================================================

set "PY=C:\Program Files\Python312\python.exe"

echo === [1] REMOTE_HOME_BASELINE_CODE ===
echo --- Administrator home top-level dirs (samst/samam) ---
dir /b "C:\Users\Administrator" 2>nul | findstr /i "samst samam"
echo --- Administrator\SchrodingerBridge contents ---
dir /b "C:\Users\Administrator\SchrodingerBridge" 2>nul | findstr /i "samst samam tools src"
echo --- Administrator\samst dir (if exists) ---
if exist "C:\Users\Administrator\samst" (dir /b "C:\Users\Administrator\samst" 2>nul) else (echo NO_samst_DIR)
echo --- Administrator\samam dir (if exists) ---
if exist "C:\Users\Administrator\samam" (dir /b "C:\Users\Administrator\samam" 2>nul) else (echo NO_samam_DIR)
echo --- Administrator\tools (if exists) ---
if exist "C:\Users\Administrator\tools" (dir /b "C:\Users\Administrator\tools" 2>nul) else (echo NO_tools_DIR)
echo --- Administrator\Related_Works (if exists) ---
if exist "C:\Users\Administrator\Related_Works" (dir /b "C:\Users\Administrator\Related_Works" 2>nul) else (echo NO_Related_Works_DIR)

echo === [2] DATASET_256 ===
echo --- classview test (known existing) ---
if exist "I:\wikiart_distinct5_samam_512_classview\test" (dir /b "I:\wikiart_distinct5_samam_512_classview\test" 2>nul) else (echo MISSING_classview_test)
echo --- pixel256 train ---
if exist "I:\wikiart_distinct5_samam_512_pixel256\train" (echo pixel256_train_EXISTS & dir /b "I:\wikiart_distinct5_samam_512_pixel256\train" 2>nul) else (echo MISSING_pixel256_train)
echo --- latent256 train ---
if exist "I:\wikiart_distinct5_samam_512_latent256\train" (echo latent256_train_EXISTS & dir /b "I:\wikiart_distinct5_samam_512_latent256\train" 2>nul) else (echo MISSING_latent256_train)
echo --- I:\Github\Latent_Style\eval_cache\hf (clip cache) ---
if exist "I:\Github\Latent_Style\eval_cache\hf" (echo clip_cache_EXISTS & dir /b "I:\Github\Latent_Style\eval_cache\hf" 2>nul) else (echo MISSING_clip_cache)

echo === [3] BASELINE_EVAL_RESULTS ===
echo --- exp dir listing ---
if exist "C:\Users\Administrator\exp" (dir /b "C:\Users\Administrator\exp" 2>nul) else (echo NO_exp_DIR)
echo --- exp\samst* matches ---
dir /b "C:\Users\Administrator\exp\samst*" 2>nul
echo --- exp\samam* matches ---
dir /b "C:\Users\Administrator\exp\samam*" 2>nul
echo --- exp\baseline* matches ---
dir /b "C:\Users\Administrator\exp\baseline*" 2>nul
echo --- exp_baselines dir (top-level) ---
if exist "C:\Users\Administrator\exp_baselines" (dir /b "C:\Users\Administrator\exp_baselines" 2>nul) else (echo NO_exp_baselines_DIR)
echo --- baseline_reeval dir ---
if exist "C:\Users\Administrator\baseline_reeval" (dir /b "C:\Users\Administrator\baseline_reeval" 2>nul) else (echo NO_baseline_reeval_DIR)
echo --- tools\samam_distinct5_scratch (curve_metrics_hf.csv) ---
if exist "C:\Users\Administrator\tools\samam_distinct5_scratch\curve_metrics_hf.csv" (echo samam_curve_csv_EXISTS) else (echo MISSING_samam_curve_csv)
echo --- tools\samst runs (if any) ---
dir /b "C:\Users\Administrator\tools\samst*" 2>nul
dir /b "C:\Users\Administrator\tools\samam*" 2>nul

echo === [4] PYTHON_ENV ===
if not exist "%PY%" (
    echo PY_EXE_MISSING: %PY%
    set "PY=C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
)
echo PY_EXE=%PY%
"%PY%" --version 2>&1
echo --- key packages ---
"%PY%" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available())" 2>&1
"%PY%" -c "import torchvision; print('torchvision', torchvision.__version__)" 2>&1
"%PY%" -c "import diffusers; print('diffusers', diffusers.__version__)" 2>&1
"%PY%" -c "import transformers; print('transformers', transformers.__version__)" 2>&1
"%PY%" -c "import accelerate; print('accelerate', accelerate.__version__)" 2>&1
"%PY%" -c "import lpips; print('lpips', lpips.__version__)" 2>&1
"%PY%" -c "import mamba_ssm; print('mamba_ssm', mamba_ssm.__version__)" 2>&1
"%PY%" -c "import causal_conv1d; print('causal_conv1d', causal_conv1d.__version__)" 2>&1
"%PY%" -c "import einops; print('einops', einops.__version__)" 2>&1
"%PY%" -c "import numpy; print('numpy', numpy.__version__)" 2>&1

echo === [5] GPU_STATUS ===
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader 2>&1
echo --- GPU processes (if any) ---
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1

echo === DONE ===
