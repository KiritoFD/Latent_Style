import importlib
def tryimport(p):
    try:
        importlib.import_module(p); return "OK"
    except Exception as e:
        return "FAIL " + str(e)[:60]
for p in ["diffusers.models.controlnet", "diffusers.models.controlnets"]:
    print(p, tryimport(p))
for sym in ["ControlNetModel","StableDiffusionControlNetPipeline","MultiControlNetModel","PipelineImageInput","retrieve_timesteps"]:
    try:
        exec(f"from diffusers import {sym}")
        print("TOPOK", sym)
    except Exception as e:
        print("TOPFAIL", sym, str(e)[:60])
for p in ["diffusers.pipelines.controlnet.pipeline_controlnet", "diffusers.pipelines.controlnet", "diffusers.image_processor", "diffusers.utils"]:
    try:
        m = importlib.import_module(p)
        syms = [x for x in ("retrieve_timesteps","StableDiffusionControlNetPipeline","MultiControlNetModel","PipelineImageInput") if hasattr(m, x)]
        print("SUBOK", p, syms)
    except Exception as e:
        print("SUBFAIL", p, str(e)[:60])
