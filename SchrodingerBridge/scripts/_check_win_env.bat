@echo off
echo === Check Python and deps ===
python --version 2>&1
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" 2>&1
python -c "import pyiqa; print('pyiqa OK')" 2>&1
python -c "import lpips; print('lpips OK')" 2>&1
python -c "import transformers; print('transformers:', transformers.__version__)" 2>&1
echo === Check CLIP cache ===
dir "C:\Users\Administrator\.cache\huggingface\hub" 2>&1 | findstr clip
dir "C:\Users\Administrator\.cache\torch\hub\pyiqa" 2>&1
echo === Check I drive ===
dir "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\samam\images" 2>&1 | findstr "File"
echo === Check existing eval results ===
dir "I:\Github\Latent_Style\SchrodingerBridge\exp\_eval_samam*.json" 2>&1
dir "I:\Github\Latent_Style\SchrodingerBridge\exp\_eval_sdturbo_w20*.json" 2>&1
