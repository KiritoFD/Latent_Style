#!/bin/bash
# Cleanup old v2 configs and verify X configs on remote
set -e
cd /mnt/i/Github/Latent_Style/SchrodingerBridge/configs
echo "=== Before cleanup ==="
ls abl512_*.json 2>/dev/null | head -5
echo "..."
# Remove old B-F v2 configs
rm -f abl512_B01_euler.json abl512_B02_rk4.json abl512_B03_euler_3ep.json
rm -f abl512_C01_no_spectral_ode.json abl512_C02_spectral_3levels.json abl512_C03_avgpool_lowpass.json abl512_C04_no_target_proj.json
rm -f abl512_D01_adain_00.json abl512_D02_adain_20.json abl512_D03_adain_every_step.json abl512_D04_lock_ll.json abl512_D05_no_extrap.json
rm -f abl512_E01_linear_path.json abl512_E02_no_coupling_struct.json abl512_E03_no_content_loss.json abl512_E04_no_style_loss.json abl512_E05_style_loss_32x.json
rm -f abl512_F01_steps_1.json abl512_F02_steps_32.json
echo "=== After cleanup ==="
echo "Total abl512_X configs:"
ls abl512_X*.json | wc -l
echo "=== First 5 ==="
ls abl512_X*.json | head -5
echo "=== Last 5 ==="
ls abl512_X*.json | tail -5
