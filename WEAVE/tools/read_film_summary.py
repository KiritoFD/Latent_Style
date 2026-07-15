import json, os, csv, glob, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, default="620_film_v2_5ep")
args = parser.parse_args()

base = f'/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/{args.exp}/full_eval'
for ep in [1, 2, 3, 4, 5]:
    p = os.path.join(base, f'epoch_{ep:04d}', 'summary.json')
    if os.path.exists(p):
        d = json.load(open(p))
        obs = d.get('runtime_observability', {}).get('all_pairs_overview', {})
        gamma = obs.get('model_film_gamma_abs', 'N/A')
        beta = obs.get('model_film_beta_abs', 'N/A')
        pre_gamma = obs.get('model_pre_film_gamma_abs', 'N/A')
        pre_beta = obs.get('model_pre_film_beta_abs', 'N/A')
        gate = obs.get('model_style_gate_value', 'N/A')
        vel = obs.get('model_velocity_abs', 'N/A')
        xent = obs.get('model_cross_attn_entropy', 'N/A')
        print(f"Epoch {ep}: film_g={gamma} film_b={beta} pre_g={pre_gamma} pre_b={pre_beta} gate={gate} vel={vel} xent={xent}")
        for k, v in d.items():
            if k == 'all_pairs_overview':
                cs = v.get('clip_style', 'N/A')
                csd = v.get('clip_s_delta_idt', 'N/A')
                clp = v.get('content_lpips', 'N/A')
                print(f"        clip_style={cs} clip_s_delta={csd} content_lpips={clp}")
                break
    else:
        print(f"Epoch {ep}: not found")

csv_files = glob.glob(f'/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/{args.exp}/logs/training_*.csv')
if csv_files:
    csv_path = sorted(csv_files)[-1]
    print(f"\n--- Training CSV: {os.path.basename(csv_path)} ---")
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        for row in rows[-5:]:
            ep = row.get('epoch', '?')
            loss = row.get('loss', '?')
            flow = row.get('flow_loss', row.get('loss_fm', '?'))
            vel = row.get('velocity_abs', '?')
            print(f"  epoch={ep} loss={loss} flow={flow} |v|={vel}")