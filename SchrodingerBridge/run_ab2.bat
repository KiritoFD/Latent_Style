@echo off  
cd /d I:\Github\Latent_Style\SchrodingerBridge  
python run_phase1_diagnostic_probes.py --action launch --phase clean_base --resume S-add__K-1_C-0_W-20_Col-0/epoch_0007.pt --epochs 14 --keep-going --skip-eval  
exit 
