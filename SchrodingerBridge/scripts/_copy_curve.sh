#!/bin/bash
# Copy SaMam curve images from WSL-owned directory to a new Windows-accessible location
# Only copy the 15 sampled steps

SRC="/mnt/i/Github/Latent_Style/exp_samam/training/samam_distinct5_512_scratch_7k_250eval_remote/c"
DST="/mnt/i/Github/Latent_Style/exp_samam/curve_pngs"

STEPS="000250 000500 001000 002000 003000 004000 005000 006000 007000 008000 010000 012000 015000 017500 020000"

for step in $STEPS; do
    echo "Copying step_$step..."
    mkdir -p "$DST/step_$step/images"
    cp "$SRC/step_$step/images/"*.png "$DST/step_$step/images/" 2>&1
    echo "  Done: $(ls "$DST/step_$step/images/" | wc -l) files"
done

echo "All done."