@echo off
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import os; p='I:\\wikiart_distinct5_samam_512_classview\\train'; print('Train dir exists:', os.path.isdir(p)); [print(d, len(os.listdir(os.path.join(p,d)))) for d in os.listdir(p) if os.path.isdir(os.path.join(p,d))]"
