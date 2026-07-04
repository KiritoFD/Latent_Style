@echo off
REM FC-SB Phase 4 T4: Full fusion training (P0 highest priority)
REM T4 = E4-long + endpoint_style_hidden_dim=512 + D2(dwt_haar) + D3(spectral_w_ll=0.3, w_hh=1.5) + D4(style_extrap_alpha=0.1)
REM Expected: ~30 min for 10 epochs with batch=16, VRAM peak 9-11GB

cd /d I:\Github\Latent_Style\SchrodingerBridge

echo ============================================================
echo [P4 T4 TRAINING] Starting full fusion training
echo [P4 T4 TRAINING] Config: configs\p4_t4_full_fusion.json
echo [P4 T4 TRAINING] Start time: %TIME% %DATE%
echo ============================================================

python -u src\run.py --config configs\p4_t4_full_fusion.json 2>&1 | tee exp\p4_fusion_breakout\t4_train.log

echo ============================================================
echo [P4 T4 TRAINING] Done. End time: %TIME% %DATE%
echo ============================================================
