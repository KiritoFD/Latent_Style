"""Diagnose T11 repro config and quick_eval. Run on remote."""
import os
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXP_DIR = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
CFG_DIR = r"I:\Github\Latent_Style\SchrodingerBridge\configs"

# 1. Check t11_repro_15ep config (resolved)
print("=" * 70)
print("=== t11_repro_15ep/config.json (key fields) ===")
print("=" * 70)
cfg_path = os.path.join(EXP_DIR, "t11_repro_15ep", "config.json")
if os.path.isfile(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = cfg.get("model", {})
    train = cfg.get("training", {})
    print(f"batch_size: {train.get('batch_size')}")
    print(f"num_epochs: {train.get('num_epochs')}")
    print(f"patience: {train.get('patience')}")
    print(f"full_eval_each_epoch: {train.get('full_eval_each_epoch')}")
    print(f"full_eval_max_src_samples: {train.get('full_eval_max_src_samples')}")
    print(f"full_eval_only_lpips_clip_style: {train.get('full_eval_only_lpips_clip_style')}")
    print(f"\n--- model key params ---")
    print(f"cross_attn_dwt_route: {model.get('cross_attn_dwt_route')}")
    print(f"dwt_route_train_prob: {model.get('dwt_route_train_prob')}")
    print(f"endpoint_adain_mode: {model.get('endpoint_adain_mode')}")
    print(f"endpoint_adain_only_last_step: {model.get('endpoint_adain_only_last_step')}")
    print(f"endpoint_adain_scale: {model.get('endpoint_adain_scale')}")
    print(f"endpoint_adain_scale_ll: {model.get('endpoint_adain_scale_ll')}")
    print(f"endpoint_adain_scale_lh: {model.get('endpoint_adain_scale_lh')}")
    print(f"endpoint_adain_scale_hl: {model.get('endpoint_adain_scale_hl')}")
    print(f"endpoint_adain_scale_hh: {model.get('endpoint_adain_scale_hh')}")
    print(f"style_extrap_alpha: {model.get('style_extrap_alpha')}")
    print(f"base_dim: {model.get('base_dim')}")
    print(f"num_res_blocks: {model.get('num_res_blocks')}")
else:
    print(f"[NOT FOUND: {cfg_path}]")

# 2. Compare with original T11 config
print("\n" + "=" * 70)
print("=== Original T11 config (630_local_t11_stochastic_dwt_p08.json) ===")
print("=" * 70)
orig_path = os.path.join(CFG_DIR, "630_local_t11_stochastic_dwt_p08.json")
if os.path.isfile(orig_path):
    with open(orig_path, "r", encoding="utf-8") as f:
        orig = json.load(f)
    model = orig.get("model", {})
    train = orig.get("training", {})
    print(f"batch_size: {train.get('batch_size')}")
    print(f"num_epochs: {train.get('num_epochs')}")
    print(f"patience: {train.get('patience')}")
    print(f"\n--- model key params ---")
    print(f"cross_attn_dwt_route: {model.get('cross_attn_dwt_route')}")
    print(f"dwt_route_train_prob: {model.get('dwt_route_train_prob')}")
    print(f"endpoint_adain_mode: {model.get('endpoint_adain_mode')}")
    print(f"endpoint_adain_only_last_step: {model.get('endpoint_adain_only_last_step')}")
    print(f"endpoint_adain_scale: {model.get('endpoint_adain_scale')}")
    print(f"endpoint_adain_scale_ll: {model.get('endpoint_adain_scale_ll')}")
    print(f"endpoint_adain_scale_lh: {model.get('endpoint_adain_scale_lh')}")
    print(f"endpoint_adain_scale_hl: {model.get('endpoint_adain_scale_hl')}")
    print(f"endpoint_adain_scale_hh: {model.get('endpoint_adain_scale_hh')}")
    print(f"style_extrap_alpha: {model.get('style_extrap_alpha')}")
else:
    print(f"[NOT FOUND: {orig_path}]")

# 3. Check quick_eval summary details
print("\n" + "=" * 70)
print("=== t11_repro_15ep quick_eval summary.json (full) ===")
print("=" * 70)
qe_path = os.path.join(EXP_DIR, "t11_repro_15ep", "quick_eval", "epoch_0005", "summary.json")
if os.path.isfile(qe_path):
    with open(qe_path, "r", encoding="utf-8") as f:
        s = json.load(f)
    # Print timings
    tim = s.get("timings_sec", {})
    print(f"wall_total: {tim.get('wall_total')}")
    print(f"lancet_generation: {tim.get('lancet_generation')}")
    print(f"lpips: {tim.get('lpips')}")
    print(f"clip: {tim.get('clip')}")
    print(f"generated_count: {s.get('generated_count')}")
    ana = s.get("analysis", {})
    tr = ana.get("style_transfer_ability", {})
    ap = ana.get("all_pairs_overview", {})
    print(f"\ntransfer: clip_style={tr.get('clip_style')}, content_lpips={tr.get('content_lpips')}")
    print(f"allpairs: clip_style={ap.get('clip_style')}, content_lpips={ap.get('content_lpips')}")
    # Check num_samples
    print(f"\ntransfer num_samples: {tr.get('num_samples', '?')}")
    print(f"allpairs num_samples: {ap.get('num_samples', '?')}")
else:
    print(f"[NOT FOUND: {qe_path}]")

# 4. Check repro_i config for comparison
print("\n" + "=" * 70)
print("=== 630_local_t11_repro_i/config.json (key fields) ===")
print("=" * 70)
ri_path = os.path.join(EXP_DIR, "630_local_t11_repro_i", "config.json")
if os.path.isfile(ri_path):
    with open(ri_path, "r", encoding="utf-8") as f:
        ri = json.load(f)
    model = ri.get("model", {})
    train = ri.get("training", {})
    print(f"batch_size: {train.get('batch_size')}")
    print(f"num_epochs: {train.get('num_epochs')}")
    print(f"full_eval_max_src_samples: {train.get('full_eval_max_src_samples')}")
    print(f"endpoint_adain_mode: {model.get('endpoint_adain_mode')}")
    print(f"endpoint_adain_scale_ll: {model.get('endpoint_adain_scale_ll')}")
    print(f"style_extrap_alpha: {model.get('style_extrap_alpha')}")
    print(f"dwt_route_train_prob: {model.get('dwt_route_train_prob')}")
else:
    print(f"[NOT FOUND: {ri_path}]")

# 5. Check repro_i full_eval summary
print("\n" + "=" * 70)
print("=== repro_i full_eval summary ===")
print("=" * 70)
ri_qe = os.path.join(EXP_DIR, "630_local_t11_repro_i", "full_eval", "epoch_0005", "summary.json")
if os.path.isfile(ri_qe):
    with open(ri_qe, "r", encoding="utf-8") as f:
        s = json.load(f)
    ana = s.get("analysis", {})
    tr = ana.get("style_transfer_ability", {})
    ap = ana.get("all_pairs_overview", {})
    print(f"transfer: clip_style={tr.get('clip_style')}, content_lpips={tr.get('content_lpips')}")
    print(f"allpairs: clip_style={ap.get('clip_style')}, content_lpips={ap.get('content_lpips')}")
    print(f"generated_count: {s.get('generated_count')}")
    tim = s.get("timings_sec", {})
    print(f"wall_total: {tim.get('wall_total')}")
else:
    print(f"[NOT FOUND: {ri_qe}]")

# 6. Check training log for t11_repro - epoch progression
print("\n" + "=" * 70)
print("=== t11_repro training log: epoch summaries ===")
print("=" * 70)
log_path = r"C:\Users\Administrator\logs\t11_repro_15ep_train_eval.out"
if os.path.isfile(log_path):
    with open(log_path, "rb") as f:
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    # Find epoch summary lines
    for line in text.splitlines():
        if "Epoch" in line and ("val" in line.lower() or "saved" in line.lower() or "eval" in line.lower() or "best" in line.lower()):
            print(line[:200])
        elif "epoch_000" in line.lower() or "checkpoint" in line.lower():
            print(line[:200])
else:
    print(f"[NOT FOUND: {log_path}]")
