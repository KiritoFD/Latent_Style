import random
from pathlib import Path

random.seed(20260705)
styles_min530 = [
    "Abstract_Expressionism",
    "Art_Nouveau_Modern",
    "Baroque",
    "Color_Field_Painting",
    "Cubism",
    "Early_Renaissance",
    "Expressionism",
    "Fauvism",
    "High_Renaissance",
    "Impressionism",
    "Mannerism_Late_Renaissance",
    "Minimalism",
    "Naive_Art_Primitivism",
    "Northern_Renaissance",
    "Pop_Art",
    "Post_Impressionism",
    "Realism",
    "Rococo",
    "Romanticism",
    "Symbolism",
    "Ukiyo_e",
]
selected = sorted(random.sample(styles_min530, 20))
print(",".join(selected))
Path("_tmp_random20_styles.txt").write_text(",".join(selected), encoding="utf-8")
print("Saved to _tmp_random20_styles.txt")
