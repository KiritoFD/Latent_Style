@echo off
REM ====================================================================
REM Follow-up probe v2 - safer echo markers (no leading/trailing ---)
REM ====================================================================

set "PY=C:\Program Files\Python312\python.exe"

echo === [A] SCHRODINGERBRIDGE_REPO_CONTENTS ===
if exist "C:\Users\Administrator\SchrodingerBridge" (
    echo SB_DIR_EXISTS
    dir /b "C:\Users\Administrator\SchrodingerBridge" 2>nul
) else (
    echo SB_DIR_MISSING
)

echo === [B] ADMINISTRATOR_TOOLS_DIR ===
if exist "C:\Users\Administrator\tools" (
    echo tools_DIR_EXISTS
    dir /b "C:\Users\Administrator\tools" 2>nul
    echo SUBSECTION tools_experiments
    if exist "C:\Users\Administrator\tools\experiments" (dir /b "C:\Users\Administrator\tools\experiments" 2>nul) else (echo NO_experiments_SUBDIR)
) else (
    echo NO_tools_DIR
)

echo === [C] ARCHIVES ===
if exist "C:\Users\Administrator\samam_mamba_py312_artifacts.tgz" (echo samam_artifacts_tgz_EXISTS) else (echo MISSING_samam_artifacts_tgz)
if exist "C:\Users\Administrator\samam_train_src_sync.tar" (echo samam_train_src_tar_EXISTS) else (echo MISSING_samam_train_src_tar)

echo === [D] WSL_SIDE_SAMAM_CODE ===
wsl -- ls -la /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/ 2>&1 | findstr /i "samam samst"
echo SUBSECTION wsl_samam_distinct5_scratch
wsl -- test -d /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch && echo WSL_SAMAM_DIR_EXISTS || echo WSL_SAMAM_DIR_MISSING
echo SUBSECTION wsl_samam_curve_csv
wsl -- test -f /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/curve_metrics_hf.csv && echo WSL_SAMAM_CSV_EXISTS || echo WSL_SAMAM_CSV_MISSING
echo SUBSECTION wsl_find_samst_dirs
wsl -- find /mnt/i/Github/Latent_Style -maxdepth 4 -type d -iname "samst*" 2>/dev/null
echo SUBSECTION wsl_find_samam_dirs
wsl -- find /mnt/i/Github/Latent_Style -maxdepth 4 -type d -iname "samam*" 2>/dev/null

echo === [E] LPIPS_VERSION ===
"%PY%" -m pip show lpips 2>&1 | findstr /i "Version Name"

echo === [F] SAMAM_EVAL_SCRIPT_PRESENCE ===
if exist "C:\Users\Administrator\eval_samam_metrics_phase2.py" (echo win_eval_samam_metrics_EXISTS) else (echo MISSING_win_eval_samam_metrics)
if exist "C:\Users\Administrator\eval_samam_curve_gpu_batched.py" (echo win_eval_samam_curve_EXISTS) else (echo MISSING_win_eval_samam_curve)
if exist "C:\Users\Administrator\gen_samam_images_phase1.py" (echo win_gen_samam_images_EXISTS) else (echo MISSING_win_gen_samam_images)
if exist "C:\Users\Administrator\run_samst_distinct5_512.sh" (echo win_run_samst_sh_EXISTS) else (echo MISSING_win_run_samst_sh)

echo === [G] MAMBA_INSTALL_SCRIPT ===
if exist "C:\Users\Administrator\install_remote_samam_mamba.sh" (
    echo install_mamba_sh_EXISTS
    findstr /i "pip install mamba causal" "C:\Users\Administrator\install_remote_samam_mamba.sh" 2>nul
) else (
    echo MISSING_install_mamba_sh
)

echo === DONE_FOLLOWUP ===
