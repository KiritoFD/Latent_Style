#!/bin/bash
# Copy scripts to correct remote locations and re-launch SaMam training.
SB_REMOTE=/mnt/i/Github/Latent_Style/SchrodingerBridge

# Copy updated scripts to remote repo
cp /mnt/c/Users/Administrator/gen_samst_latent_v2.py $SB_REMOTE/scripts/samst_latent/gen_samst_latent.py
cp /mnt/c/Users/Administrator/gen_samam_latent_v2.py $SB_REMOTE/scripts/samam_latent/gen_samam_latent.py
cp /mnt/c/Users/Administrator/compute_extra_metrics.py $SB_REMOTE/scripts/compute_extra_metrics.py

echo "Scripts copied to:"
ls -la $SB_REMOTE/scripts/samst_latent/gen_samst_latent.py
ls -la $SB_REMOTE/scripts/samam_latent/gen_samam_latent.py
ls -la $SB_REMOTE/scripts/compute_extra_metrics.py

# Verify diffusers version
/home/xy/venvs/samam312/bin/python -c "import diffusers; print('diffusers:', diffusers.__version__)"

# Re-launch SaMam training (in background)
echo "=== Re-launching SaMam-latent training ==="
bash /mnt/c/Users/Administrator/launch_samam_train.sh
