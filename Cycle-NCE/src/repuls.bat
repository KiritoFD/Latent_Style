@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/8] Running 00_l1_mean_filter...
uv run run.py --config config_00_l1_mean_filter.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/8] Running 01_mse_local_tear...
uv run run.py --config config_01_mse_local_tear.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/8] Running 02_micro_mesh...
uv run run.py --config config_02_micro_mesh.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/8] Running 03_zero_skip_isolation...
uv run run.py --config config_03_zero_skip_isolation.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/8] Running 04_subzero_one_hot...
uv run run.py --config config_04_subzero_one_hot.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/8] Running 05_highway_override...
uv run run.py --config config_05_highway_override.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [7/8] Running 06_structure_sacrifice...
uv run run.py --config config_06_structure_sacrifice.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [8/8] Running 07_edge_rebel...
uv run run.py --config config_07_edge_rebel.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo All training runs finished.
set "SRC_DIR=%cd%"
set "ROOT_DIR=%SRC_DIR%\.."
set "TARGET_DIR=%ROOT_DIR%\repuls"
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

for /d %%D in ("%ROOT_DIR%\repuls_*") do (
  echo Moving %%~nxD to %TARGET_DIR%...
  robocopy "%%~fD" "%TARGET_DIR%\%%~nxD" /MOVE /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
  if errorlevel 8 exit /b 8
  if exist "%%~fD" rmdir "%%~fD" /S /Q
)

echo Move finished. Exporting pre-distill CSV summary...
cd /d "%ROOT_DIR%"
python import_summary_history_to_csv.py -i repuls -o repuls_pre_distill.csv
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running batch distill + post-distill eval...
uv run src/batch_distill_full_eval.py --exp_dir repuls --recursive --distill_mode tokenizer --distill_epochs 200 --batch_size 32
if %errorlevel% neq 0 exit /b %errorlevel%

echo Distill/eval finished. Exporting post-distill CSV summary...
python import_summary_history_to_csv.py -i repuls -o repuls_post_distill.csv
if %errorlevel% neq 0 exit /b %errorlevel%
python import_summary_history_to_csv.py -i repuls -o repuls.csv
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running MA probe per experiment (base + tokenized)...
set "FINAL_EPOCH=0080"
set "DISTILL_TAG=distill_epochs200"
for /d %%D in ("%TARGET_DIR%\repuls_*") do (
  if exist "%%~fD\epoch_%FINAL_EPOCH%.pt" (
    echo Probing BASE %%~nxD with %%~fD\epoch_%FINAL_EPOCH%.pt...
    uv run src/probe_ma.py --checkpoint "%%~fD\epoch_%FINAL_EPOCH%.pt" --num-samples 8 --json-out "%%~fD\ma_probe_base_epoch_%FINAL_EPOCH%.json"
    if %errorlevel% neq 0 exit /b %errorlevel%
  ) else (
    echo WARNING: base checkpoint not found for %%~nxD at %%~fD\epoch_%FINAL_EPOCH%.pt
  )
  if exist "%%~fD\tokenizer_distill\epoch_%FINAL_EPOCH%_%DISTILL_TAG%\epoch_%FINAL_EPOCH%_tokenized.pt" (
    echo Probing TOKENIZED %%~nxD with tokenizer_distill\epoch_%FINAL_EPOCH%_%DISTILL_TAG%\epoch_%FINAL_EPOCH%_tokenized.pt...
    uv run src/probe_ma.py --checkpoint "%%~fD\tokenizer_distill\epoch_%FINAL_EPOCH%_%DISTILL_TAG%\epoch_%FINAL_EPOCH%_tokenized.pt" --num-samples 8 --json-out "%%~fD\ma_probe_tokenized_epoch_%FINAL_EPOCH%.json"
    if %errorlevel% neq 0 exit /b %errorlevel%
  ) else (
    echo WARNING: tokenized checkpoint not found for %%~nxD at %%~fD\tokenizer_distill\epoch_%FINAL_EPOCH%_%DISTILL_TAG%\epoch_%FINAL_EPOCH%_tokenized.pt
  )
  if exist "%%~fD\ma_probe*.json" (
    uv run src/probe_ma_sweep.py --input-glob "%%~fD\ma_probe*.json" --output-dir "%%~fD" --output-prefix ma_probe_view
    if %errorlevel% neq 0 exit /b %errorlevel%
  )
)

echo Building cross-experiment MA summary...
uv run src/probe_ma_sweep.py --input-glob "%TARGET_DIR%\repuls_*\ma_probe*.json" --output-dir "%TARGET_DIR%" --output-prefix ma_probe_all_pairs
if %errorlevel% neq 0 exit /b %errorlevel%

for /d %%D in ("%TARGET_DIR%\repuls_*") do (
  if exist "%TARGET_DIR%\ma_probe_all_pairs.html" copy /Y "%TARGET_DIR%\ma_probe_all_pairs.html" "%%~fD\ma_probe_all_pairs.html" >nul
  if exist "%TARGET_DIR%\ma_probe_all_pairs.csv" copy /Y "%TARGET_DIR%\ma_probe_all_pairs.csv" "%%~fD\ma_probe_all_pairs.csv" >nul
  if exist "%TARGET_DIR%\ma_probe_all_pairs.json" copy /Y "%TARGET_DIR%\ma_probe_all_pairs.json" "%%~fD\ma_probe_all_pairs.json" >nul
)

echo All done.
