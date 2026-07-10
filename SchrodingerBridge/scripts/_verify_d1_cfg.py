import json
c = json.load(open("configs/d1_gram_hf1_15ep.json"))
print("base=", c.get("_base"), "w_gram_hf=", c.get("bridge",{}).get("w_gram_hf"), "bs=", c.get("training",{}).get("batch_size"))
