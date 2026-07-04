@echo off
REM FC-SB Phase 4 Stage 0 inference ablation batch (D1-D11)
REM D0_baseline already run, this script runs D1-D11 sequentially
REM Each eval ~5-8 min, 11 groups ~55-88 min

cd /d I:\Github\Latent_Style\SchrodingerBridge

echo ============================================================
echo [P4 ABLATION] Starting D1-D11 sequential ablation
echo [P4 ABLATION] Start time: %TIME% %DATE%
echo ============================================================

echo.
echo === D1: + DWT (lowpass_mode=dwt_haar) ===
python -u _p4_infer_ablation.py D1_dwt dwt_haar 0 0 single 0 0.3 0.3

echo.
echo === D2: + U4 (style_extrap_alpha=0.1) ===
python -u _p4_infer_ablation.py D2_u4 avg_pool 0.1 0 single 0 0.3 0.3

echo.
echo === D3: + V3 (patch_adain_kernel=16) ===
python -u _p4_infer_ablation.py D3_v3 avg_pool 0 16 single 0 0.3 0.3

echo.
echo === D4: + V6 (patch_adain_kernel=32) ===
python -u _p4_infer_ablation.py D4_v6 avg_pool 0 32 single 0 0.3 0.3

echo.
echo === D5: + T (multiband_adain_mode=two_level) ===
python -u _p4_infer_ablation.py D5_t avg_pool 0 0 two_level 0 0.3 0.3

echo.
echo === D6: + tri_band_lock (tri_band_inference_lock=1) ===
python -u _p4_infer_ablation.py D6_triband avg_pool 0 0 single 1 0.3 0.3

echo.
echo === D7: U4 + V3 (alpha=0.1, k=16) ===
python -u _p4_infer_ablation.py D7_u4_v3 avg_pool 0.1 16 single 0 0.3 0.3

echo.
echo === D8: U4 + V6 (alpha=0.1, k=32) ===
python -u _p4_infer_ablation.py D8_u4_v6 avg_pool 0.1 32 single 0 0.3 0.3

echo.
echo === D9: U4 + V3 + DWT (alpha=0.1, k=16, dwt_haar) ===
python -u _p4_infer_ablation.py D9_u4_v3_dwt dwt_haar 0.1 16 single 0 0.3 0.3

echo.
echo === D10: U4 + V3 + DWT + T (alpha=0.1, k=16, dwt_haar, two_level) ===
python -u _p4_infer_ablation.py D10_u4_v3_dwt_t dwt_haar 0.1 16 two_level 0 0.3 0.3

echo.
echo === D11: U4 + V6 + DWT + T (alpha=0.1, k=32, dwt_haar, two_level) ===
python -u _p4_infer_ablation.py D11_u4_v6_dwt_t dwt_haar 0.1 32 two_level 0 0.3 0.3

echo.
echo ============================================================
echo [P4 ABLATION] All D1-D11 done. End time: %TIME% %DATE%
echo ============================================================

python -u _p4_summarize_ablations.py

echo [P4 ABLATION] Summary written to exp/p4_fusion_breakout/infer_ablation/_summary.md
