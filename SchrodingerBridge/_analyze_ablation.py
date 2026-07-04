import csv
import json

def read_training_log(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

baseline = read_training_log(r'exp\620_spatial_bridge\620_film_smoke_baseline_local\logs\training_20260623_112415.csv')
no_norm = read_training_log(r'exp\620_spatial_bridge\620_film_smoke_no_norm_local\logs\training_20260623_112445.csv')

key_metrics = [
    'loss', 'flow', 'endpoint_alpha', 'endpoint_high_alpha', 'latent_alpha',
    'style_gate_value', 'endpoint_output_std', 'velocity_std', 'leak',
    'fiber_bundle_variance', 'otd', 'tgtshift', 'endpoint_low_alpha'
]

print('=' * 80)
print('关键指标对比 (Epoch 1, 10 steps)')
print('=' * 80)
print('{:<30} {:<20} {:<20} {:<15}'.format('指标', 'Baseline (GN)', 'No Norm', '差值'))
print('-' * 80)

b = baseline[0]
n = no_norm[0]

results = {}
for metric in key_metrics:
    b_val = b.get(metric, 'N/A')
    n_val = n.get(metric, 'N/A')
    try:
        b_f = float(b_val)
        n_f = float(n_val)
        diff = n_f - b_f
        print('{:<30} {:<20.6f} {:<20.6f} {:<+15.6f}'.format(metric, b_f, n_f, diff))
        results[metric] = {
            'baseline': b_f,
            'no_norm': n_f,
            'diff': diff
        }
    except:
        print('{:<30} {:<20} {:<20} {:<15}'.format(metric, b_val, n_val, 'N/A'))
        results[metric] = {
            'baseline': b_val,
            'no_norm': n_val,
            'diff': 'N/A'
        }

print()
print('=' * 80)
print('所有可用列名:')
print('=' * 80)
for i, col in enumerate(baseline[0].keys()):
    if i % 4 == 0:
        print()
    print('{:<40}'.format(col), end='')
print()

# 保存结果到 JSON
output = {
    'baseline_config': '620_film_smoke_baseline_local (use_norm=true)',
    'no_norm_config': '620_film_smoke_no_norm_local (use_norm=false)',
    'training_steps': 10,
    'num_epochs': 1,
    'model_params': {
        'baseline': 1750868,
        'no_norm': 1750612,
        'diff': -256
    },
    'metrics': results
}

with open('exp/620_spatial_bridge/endpoint_film_ablation_results.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print()
print('结果已保存到 exp/620_spatial_bridge/endpoint_film_ablation_results.json')
