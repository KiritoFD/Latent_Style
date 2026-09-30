import csv, sys
p = sys.argv[1] if len(sys.argv) > 1 else 'exp/evo_d5_baseline/full_eval/epoch_0005/metrics.csv'
r = list(csv.DictReader(open(p, encoding='utf-8')))
print('COLS:', list(r[0].keys()))
print('ROW0:')
for k, v in r[0].items():
    print(f'  {k}: {v[:80]}')
