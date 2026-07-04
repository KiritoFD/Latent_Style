import subprocess
commits = ['f5e652b36', '6f4b88c47', '6a80e1724']
for c in commits:
    path = f'SchrodingerBridge/aaai2027/paper_aaai2027.tex'
    out = subprocess.check_output(['git', 'show', f'{c}:{path}'], text=True)
    with open(f'aaai2027/_hist_{c}.tex', 'w', encoding='utf-8') as f:
        f.write(out)
    print(c, len(out))
