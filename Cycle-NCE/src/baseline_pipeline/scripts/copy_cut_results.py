"""
Copy existing CUT inference results from Related_Works to current pipeline
No need to retrain CUT
"""
import os
import shutil
from pathlib import Path

# Configuration
OLD_CUT_RESULTS = "../../../../Related_Works/runs/cut_5x5"  # Your existing 5x5 CUT results path
NEW_CUT_OUTPUT = "../results/cut"
STYLES = ["photo", "monet", "vangogh", "ukiyoe", "cezanne"]  # 5 styles for full 5x5 conversion, all equal for any direction

def copy_cut_results():
    print("Copying existing 5x5 CUT results...")
    
    source_images_dir = os.path.join(OLD_CUT_RESULTS, "infer_val_clean_5x5", "images")
    target_styles = [s for s in STYLES if s != "photo"]  # We only need art style generation results
    
    # Create output directories
    for style in target_styles:
        new_dir = os.path.join(NEW_CUT_OUTPUT, style)
        os.makedirs(new_dir, exist_ok=True)
    
    # Filter and copy photo -> art style results
    files = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    count = 0
    
    for f in files:
        # Format: photo_XXXX_to_{style}.jpg
        parts = f.split("_to_")
        if len(parts) != 2:
            continue
            
        src_style = parts[0].split("_")[0]
        tgt_style = parts[1].rsplit(".", 1)[0]
        
        # We only need photo -> art style conversion results
        if src_style == "photo" and tgt_style in target_styles:
            src = os.path.join(source_images_dir, f)
            dst = os.path.join(NEW_CUT_OUTPUT, tgt_style, f)
            shutil.copy2(src, dst)
            count += 1
    
    print(f"Successfully copied {count} CUT results for {len(target_styles)} styles")
    for style in target_styles:
        style_files = len(os.listdir(os.path.join(NEW_CUT_OUTPUT, style)))
        print(f"  - {style}: {style_files} images")
    
    print("All CUT results copied successfully!")

if __name__ == "__main__":
    copy_cut_results()
