"""Pull Ours per-epoch DINO-S + CLIP-S + LPIPS from remote."""
import json, pathlib

base = pathlib.Path(r"I:\Github\Latent_Style\WEAVE\runs\submission\arch_hf_oriented_nohh_15ep\paper_eval")
print("epoch,dino_s,clip_s,lpips")
for d in sorted(base.glob("epoch_*")):
    ep = int(d.name.split("_")[1])
    dino_path = d / "dino_summary.json"
    summ_path = d / "summary.json"
    if not dino_path.exists() or not summ_path.exists():
        print(f"{ep},MISSING,MISSING,MISSING")
        continue
    dino = json.load(open(dino_path))
    summ = json.load(open(summ_path))
    st = summ["analysis"]["style_transfer_ability"]
    print(f"{ep},{dino['all_dino_s']:.6f},{st['clip_style']:.6f},{st['content_lpips']:.6f}")