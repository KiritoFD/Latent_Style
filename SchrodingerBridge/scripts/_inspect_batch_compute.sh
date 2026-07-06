#!/usr/bin/env bash
grep -n "methods-json\|methods_json\|argparse\|add_argument\|def main" /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/batch_compute_photo2art.py | head -30
echo "===tail of script (main function)==="
tail -80 /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/batch_compute_photo2art.py
