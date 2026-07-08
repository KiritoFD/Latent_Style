@echo off
python -c "import os; [print(d,':',os.path.exists(d)) for d in ['I:\\\\','I:\\\\Github','I:\\\\wikiart_distinct5_samam_512_classview','I:\\\\wikiart_distinct5_samam_512_latents_ema','I:\\\\wikiart_distinct5_samam_512_latents_ema\\\\train']]"
python -c "import os; p='I:\\\\wikiart_distinct5_samam_512_latents_ema\\\\train'; print('Latent train:', os.path.isdir(p)); dirs=[d for d in os.listdir(p) if os.path.isdir(os.path.join(p,d))] if os.path.isdir(p) else []; [print(d, len(os.listdir(os.path.join(p,d)))) for d in dirs]"
