# Setup script for Baseline Reproduction Pipeline
Write-Host "=== Installing dependencies ===" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "`n=== Cloning baseline repositories ===" -ForegroundColor Cyan

# StyleID
cd baselines/styleid
git clone https://github.com/jiwoog/StyleID.git .
cd ../..

# StyleAligned
cd baselines/style_aligned
git clone https://github.com/google/style-aligned.git .
cd ../..

# S2WAT
cd baselines/s2wat
git clone https://github.com/NUST-Machine-Intelligence-Laboratory/S2WAT.git .
cd ../..

# CycleGAN-Turbo
cd baselines/cyclegan_turbo
git clone https://github.com/GaParmar/img2img-turbo.git .
cd ../..

# B-LoRA
cd baselines/blora
git clone https://github.com/yardenf/B-LoRA.git .
cd ../..

# CUT
cd baselines/cut
git clone https://github.com/taesungp/contrastive-unpaired-translation.git .
cd ../..

Write-Host "`n=== Downloading pre-trained weights (if available) ===" -ForegroundColor Cyan
# CycleGAN-Turbo pre-trained weights
Invoke-WebRequest -Uri "https://www.cs.cmu.edu/~img2img-turbo/models/photo2monet_pytorch.pt" -OutFile "checkpoints/photo2monet_pytorch.pt"
Invoke-WebRequest -Uri "https://www.cs.cmu.edu/~img2img-turbo/models/photo2vangogh_pytorch.pt" -OutFile "checkpoints/photo2vangogh_pytorch.pt"

Write-Host "`n=== Setup completed successfully ===" -ForegroundColor Green
Write-Host "Next steps: "
Write-Host "1. Place your 30 test content images in datasets/test_content/"
Write-Host "2. Place your test style images in datasets/test_style/<style_name>/"
Write-Host "3. Run main.py to execute the pipeline"
