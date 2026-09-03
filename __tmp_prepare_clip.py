from transformers import CLIPModel, CLIPProcessor
cache_dir = r'F:\GitHub\Latent_Style\Cycle-NCE\eval_cache\hf'
model_name = 'openai/clip-vit-base-patch32'
print('prepare_clip_start')
CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
print('prepare_clip_done')
