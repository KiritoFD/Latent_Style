@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%

echo [1/7] Running 00_holy_grail...
uv run run.py --config config_00_holy_grail.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/7] Running 01_highway_cut...
uv run run.py --config config_01_highway_cut.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [3/7] Running 02_dirty_skip...
uv run run.py --config config_02_dirty_skip.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [4/7] Running 03_decoder_usurpation...
uv run run.py --config config_03_decoder_usurpation.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [5/7] Running 04_muddy_routing...
uv run run.py --config config_04_muddy_routing.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [6/7] Running 05_micro_dictatorship...
uv run run.py --config config_05_micro_dictatorship.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo [7/7] Running 06_hard_anchor...
uv run run.py --config config_06_hard_anchor.json
if %errorlevel% neq 0 exit /b %errorlevel%

echo All training runs finished.
set "SRC_DIR=%cd%"
set "ROOT_DIR=%SRC_DIR%\.."
set "TARGET_DIR=%ROOT_DIR%\46"
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

for /d %%D in ("%ROOT_DIR%\46_*") do (
  echo Moving %%~nxD to %TARGET_DIR%...
  robocopy "%%~fD" "%TARGET_DIR%\%%~nxD" /MOVE /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
  if errorlevel 8 exit /b 8
  if exist "%%~fD" rmdir "%%~fD" /S /Q
)

echo Move finished. Exporting pre-distill CSV summary...
cd /d "%ROOT_DIR%"
python import_summary_history_to_csv.py -i 46 -o 46_pre_distill.csv
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running batch distill + post-distill eval...
uv run src/batch_distill_full_eval.py --exp_dir 46 --recursive --distill_mode tokenizer --distill_epochs 200
if %errorlevel% neq 0 exit /b %errorlevel%

echo Distill/eval finished. Exporting post-distill CSV summary...
python import_summary_history_to_csv.py -i 46 -o 46_post_distill.csv
if %errorlevel% neq 0 exit /b %errorlevel%
python import_summary_history_to_csv.py -i 46 -o 46.csv
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running MA probe per experiment (base + tokenized)...
set "FINAL_EPOCH=0080"
set "DISTILL_TAG=distill_epochs200"
for /d %%D in ("%TARGET_DIR%\46_*") do (
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
uv run src/probe_ma_sweep.py --input-glob "%TARGET_DIR%\46_*\ma_probe*.json" --output-dir "%TARGET_DIR%" --output-prefix ma_probe_all_pairs
if %errorlevel% neq 0 exit /b %errorlevel%

for /d %%D in ("%TARGET_DIR%\46_*") do (
  if exist "%TARGET_DIR%\ma_probe_all_pairs.html" copy /Y "%TARGET_DIR%\ma_probe_all_pairs.html" "%%~fD\ma_probe_all_pairs.html" >nul
  if exist "%TARGET_DIR%\ma_probe_all_pairs.csv" copy /Y "%TARGET_DIR%\ma_probe_all_pairs.csv" "%%~fD\ma_probe_all_pairs.csv" >nul
  if exist "%TARGET_DIR%\ma_probe_all_pairs.json" copy /Y "%TARGET_DIR%\ma_probe_all_pairs.json" "%%~fD\ma_probe_all_pairs.json" >nul
)

echo All done.
