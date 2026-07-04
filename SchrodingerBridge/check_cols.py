import pandas as pd
import glob

exp_name = 'task4_style_strength_w05_2ep'
csv_path = f'exp/{exp_name}/logs/training_*.csv'
files = glob.glob(csv_path)
if files:
    df = pd.read_csv(files[-1])
    print(f'总列数: {len(df.columns)}')
    print('\n所有列名:')
    for i, col in enumerate(df.columns):
        print(f'  {i:3d}: {col}')
    
    # 搜索包含 'style' 或 'strength' 或 'alpha' 的列
    print('\n包含 style/strength/alpha 的列:')
    for col in df.columns:
        if any(k in col.lower() for k in ['style', 'strength', 'alpha']):
            print(f'  {col}')
