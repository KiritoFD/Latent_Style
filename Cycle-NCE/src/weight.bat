@echo off
setlocal
cd /d %~dp0
if %errorlevel% neq 0 exit /b %errorlevel%
echo ==========================================
echo Running Weight 2x2x2 Ablation (8 exps)
echo ==========================================
echo.
echo ------------------------------------------
echo Running Experiment 1: weight_exp1_latent_adain_swd30_tv01_id20_r16_e60
echo ------------------------------------------
uv run run.py --config config_weight_exp1_latent_adain_swd30_tv01_id20_r16_e60.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 2: weight_exp2_latent_adain_swd30_tv00_id40_r16_e60
echo ------------------------------------------
uv run run.py --config config_weight_exp2_latent_adain_swd30_tv00_id40_r16_e60.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 3: weight_exp3_latent_adain_swd60_tv01_id20_r16_e60
echo ------------------------------------------
uv run run.py --config config_weight_exp3_latent_adain_swd60_tv01_id20_r16_e60.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 4: weight_exp4_latent_adain_swd60_tv00_id40_r16_e60
echo ------------------------------------------
uv run run.py --config config_weight_exp4_latent_adain_swd60_tv00_id40_r16_e60.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 5: weight_exp5_pseudo_hist_swd30_tv01_id20_r16_e60
echo ------------------------------------------
uv run run.py --config config_weight_exp5_pseudo_hist_swd30_tv01_id20_r16_e60.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 6: weight_exp6_pseudo_hist_swd30_tv00_id40_r16_e60
echo ------------------------------------------
uv run run.py --config config_weight_exp6_pseudo_hist_swd30_tv00_id40_r16_e60.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 7: weight_exp7_pseudo_hist_swd60_tv01_id20_r16_e60
echo ------------------------------------------
uv run run.py --config config_weight_exp7_pseudo_hist_swd60_tv01_id20_r16_e60.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo ------------------------------------------
echo Running Experiment 8: weight_exp8_pseudo_hist_swd60_tv00_id40_r16_e60
echo ------------------------------------------
uv run run.py --config config_weight_exp8_pseudo_hist_swd60_tv00_id40_r16_e60.json
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
echo Weight ablation finished.
