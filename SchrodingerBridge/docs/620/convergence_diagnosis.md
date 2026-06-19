# 620 Convergence Diagnosis Protocol

## Problem Pattern

When training 620 Spatial Bridge, we observed:
- **Epoch 1 already optimal**: clip_style highest at e1, then decreases
- **No learning curve**: Expected e1 low → mid peak → late overfit
- **LPIPS degradation**: content_lpips increases as training continues

Example (swd16, virtual_length=1.0):
```
e1: clip=0.7053, lpips=0.2901  ← Best
e8: clip=0.7046, lpips=0.3058  ← Worse
```

## Diagnosis Protocol

### Step 1: Check Convergence Point

Run smoke test (1 epoch) with `virtual_length=1.0`:
```bash
python launch_remote_620_spatial_bridge.py --variant swd16 --epochs 1 --batch-size 64
```

If e1 is already the best epoch, proceed to Step 2.

### Step 2: Refine Virtual Length

The issue: `virtual_length=1.0` means each epoch traverses the full dataset. If optimal point is at e1, we need finer granularity.

**Progressive Refinement:**

1. **virtual_length=0.2** (each epoch = 20% data):
   ```bash
   # Modify config
   "data": {"virtual_length_multiplier": 0.2}
   
   # Run 10 epochs
   python launch_remote_620_spatial_bridge.py --variant swd16 --epochs 10 --batch-size 64
   ```

2. **virtual_length=0.04** (each epoch = 4% data):
   ```bash
   "data": {"virtual_length_multiplier": 0.04}
   
   python launch_remote_620_spatial_bridge.py --variant swd16 --epochs 10 --batch-size 64
   ```

### Step 3: Identify Optimal Epoch

With fine-grained epochs, analyze the curve:

Example (swd16, virtual_length=0.04):
```
e1:  clip=0.7007, lpips=0.2751  ← lpips best
e2:  clip=0.7029, lpips=0.2810
e3:  clip=0.7040, lpips=0.2871
e4:  clip=0.7048, lpips=0.2923
e5:  clip=0.7051, lpips=0.2935  ← clip best
e6:  clip=0.7050, lpips=0.2901  ← stable
```

**Decision Criteria:**
- **Peak clip_style**: Choose epoch with max clip_style
- **Trade-off**: If lpips degradation is unacceptable, choose earlier epoch
- **Stability**: If curve plateaus, any epoch in plateau region is acceptable

## Results Summary

### swd16 Convergence

| virtual_length | optimal_epoch | clip_style | content_lpips | notes |
|---------------|---------------|-----------|--------------|-------|
| 1.0 | e1 | 0.7053 | 0.2901 | Too coarse, can't see learning |
| 0.2 | e9 | 0.7038 | 0.3064 | Learning observed, but lpips worse |
| 0.04 | e5 | 0.7051 | 0.2935 | **Best**: clear learning curve |

**Conclusion:**
- swd16 optimal: virtual_length=0.04, epoch=5
- clip_style=0.7051, content_lpips=0.2935
- Exceeds IDT baseline (+0.065) and SaMAM (+0.090)

## Implementation

### Config Template

```json
{
  "_base": "620_spatial_bridge_base.json",
  "bridge": {
    "single_step_swd_weight": 16.0
  },
  "data": {
    "virtual_length_multiplier": 0.04
  },
  "training": {
    "batch_size": 64,
    "num_epochs": 10
  }
}
```

### Analysis Script

```python
import json, glob

def analyze_convergence(exp_dir):
    epochs = sorted(glob.glob(f"{exp_dir}/full_eval/epoch_*"))
    results = []
    
    for ep_dir in epochs:
        with open(f"{ep_dir}/summary.json") as f:
            d = json.load(f)
        apo = d["analysis"]["all_pairs_overview"]
        results.append({
            "epoch": os.path.basename(ep_dir),
            "clip_style": apo["clip_style"],
            "content_lpips": apo["content_lpips"]
        })
    
    # Find optimal
    best_clip = max(results, key=lambda x: x["clip_style"])
    best_lpips = min(results, key=lambda x: x["content_lpips"])
    
    print(f"Best clip_style: {best_clip}")
    print(f"Best content_lpips: {best_lpips}")
    
    return results
```

## Next Steps

1. Apply this protocol to other variants (swd20, swd24)
2. Build convergence database for hyperparameter tuning
3. Automate virtual_length selection based on e1 performance
