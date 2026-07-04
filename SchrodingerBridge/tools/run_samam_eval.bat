@echo off
"C:\Program Files\Python312\python.exe" "I:\GitHub\Latent_Style\SchrodingerBridge\tools\prepare_samam_eval.py"
echo ===PREP_DONE===
"C:\Program Files\Python312\python.exe" "I:\GitHub\Latent_Style\SchrodingerBridge\src\utils\run_evaluation.py" "I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\samam" --reuse_generated --save_generated_images --style_subdirs Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e --test_dir "I:\wikiart_distinct5_samam_512_classview\test" --eval_only_lpips_clip_style --clip_style_idt_baseline 0.6399
echo ==SAMAM_EVAL_DONE==
