import ast
files = [
    "I:/Github/Latent_Style/SchrodingerBridge/src/spectral_bridge620.py",
    "I:/Github/Latent_Style/SchrodingerBridge/src/blocks620.py",
    "I:/Github/Latent_Style/SchrodingerBridge/src/config_schema.py",
]
for f in files:
    ast.parse(open(f, encoding='utf-8').read())
    print(f"OK: {f}")
print("ALL_SYNTAX_OK")
