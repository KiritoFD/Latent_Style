"""Extract Ours & repro per-epoch DINO-S + CLIP-S + LPIPS from remote submission runs."""
import json
import pathlib

# --- arch_hf_oriented_nohh_15ep ---
base1 = pathlib.Path(r"I:\Github\Latent_Style\WEAVE\runs\submission\arch_hf_oriented_nohh_15ep\paper_eval")
print("=== arch_hf_oriented_nohh_15ep ===")
print("epoch,dino_s,clip_s,lpips")
for d in sorted(base1.glob("epoch_*")):
    ep = int(d.name.split("_")[1])
    dino = json.load(open(d / "dino_summary.json"))
    summ = json.load(open(d / "summary.json"))
    st = summ["analysis"]["style_transfer_ability"]
    print(f"{ep},{dino['all_dino_s']:.6f},{st['clip_style']:.6f},{st['content_lpips']:.6f}")

# --- repro_brk_a_15ep ---
base2 = pathlib.Path(r"I:\Github\Latent_Style\WEAVE\runs\submission\repro_brk_a_15ep\paper_eval_adain20")
if base2.exists():
    print()
    print("=== repro_brk_a_15ep (adain20) ===")
    print("epoch,dino_s,clip_s,lpips")
    for d in sorted(base2.glob("epoch_*")):
        ep = int(d.name.split("_")[1])
        dino = json.load(open(d / "dino_summary.json"))
        summ = json.load(open(d / "summary.json"))
        st = summ["analysis"]["style_transfer_ability"]
        print(f"{ep},{dino['all_dino_s']:.6f},{st['clip_style']:.6f},{st['content_lpips']:.6f}")