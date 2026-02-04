import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import re
import json
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ==========================================
# 1. 配置区域
# ==========================================
MODEL_PATH = "./style_judge_resnet18_selective.pt"  # 你的分类器权重
IMG_DIR = "photo-vangogh/500classify"           # 待评测的图片目录
SAVE_REPORT = "./eval_report.json"

# 必须与你训练分类器时的 SELECTED_CLASSES 顺序完全一致！
CLASS_NAMES = ["vangogh", "photo"] 
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# 正则表达式：匹配 "风格A...to_风格B" 格式
# 例子: photo_2016-08-29 16_18_55_to_photo.jpg -> group(1)=photo, group(2)=photo
# 例子: photo_to_vangogh_epoch500_01.jpg -> group(1)=photo, group(2)=vangogh
FILENAME_PATTERN = r"^([a-zA-Z]+).*_to_([a-zA-Z]+)"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 加载模型
# ==========================================
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # 冻结骨干网络 (必须与训练时一致)
    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    
    # 🔥 关键：必须与 classify.py 中的架构完全一致！
    # classify.py 使用了 Sequential 头 + Dropout，这里也必须这样做
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, len(CLASS_NAMES))
    )
    
    # 加载权重
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# ==========================================
# 3. 图像预处理 (严格对齐训练环境)
# ==========================================
eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),      # 强制缩放对齐视野
    transforms.ToTensor(),              # 转为 [0, 1] RGB
    transforms.Normalize(               # ImageNet 标准归一化
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# 4. 主评估逻辑
# ==========================================
def run_evaluation():
    model = load_model()
    results = []
    
    y_true = []  # 目标风格 (Ground Truth for Style)
    y_pred = []  # 分类器预测风格
    
    print(f"开始读取目录: {IMG_DIR}")
    
    # 检查目录是否存在
    if not os.path.exists(IMG_DIR):
        print(f"❌ 错误: 目录不存在: {IMG_DIR}")
        print(f"当前工作目录: {os.getcwd()}")
        return
    
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"找到 {len(img_files)} 个图片文件")
    
    skipped_count = 0
    
    for filename in img_files:
        # A. 正则解析文件名
        match = re.search(FILENAME_PATTERN, filename)
        if not match:
            skipped_count += 1
            continue
            
        src_style = match.group(1) # 原始风格 (e.g., photo)
        tgt_style = match.group(2) # 目标风格 (e.g., vangogh)
        
        if tgt_style not in CLASS_TO_IDX:
            skipped_count += 1
            continue

        # B. 加载图片 (强制 RGB 防止 BGR 干扰)
        img_path = os.path.join(IMG_DIR, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            input_tensor = eval_transform(img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"跳过损坏图片 {filename}: {e}")
            continue

        # C. 预测
        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            conf = torch.softmax(output, dim=1)[0][pred_idx].item()

        # D. 记录
        target_idx = CLASS_TO_IDX[tgt_style]
        y_true.append(target_idx)
        y_pred.append(pred_idx)
        
        results.append({
            "filename": filename,
            "src": src_style,
            "tgt": tgt_style,
            "pred": CLASS_NAMES[pred_idx],
            "confidence": conf,
            "is_correct": (target_idx == pred_idx)
        })

    # ==========================================
    # 5. 生成指标报告
    # ==========================================
    if not y_true or not y_pred:
        print(f"❌ 错误: 未找到任何有效的图片进行评估")
        print(f"请检查:")
        print(f"  1. 目录是否存在且包含图片: {IMG_DIR}")
        print(f"  2. 文件名是否符合格式: 格式应为 'style_..._to_style.jpg'")
        print(f"     当前正则: {FILENAME_PATTERN}")
        print(f"  3. 目标风格是否在 CLASS_NAMES 中: {CLASS_NAMES}")
        print(f"  4. 已跳过 {skipped_count}/{len(img_files)} 个文件（不符合格式）")
        # 显示几个样本文件名供调试
        if img_files:
            print(f"\n样本文件名:")
            for i, fname in enumerate(img_files[:5]):
                print(f"    {fname}")
        return
    
    report = classification_report(
        y_true, y_pred, 
        target_names=CLASS_NAMES, 
        output_dict=True,
        zero_division=0
    )
    
    print(f"✓ 成功处理了 {len(results)} 个图片 (跳过了 {skipped_count} 个)")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # 统计每个 风格A -> 风格B 的迁移成功率
    transfer_matrix = {}
    for res in results:
        key = f"{res['src']}_to_{res['tgt']}"
        if key not in transfer_matrix:
            transfer_matrix[key] = {"count": 0, "correct": 0}
        transfer_matrix[key]["count"] += 1
        if res["is_correct"]:
            transfer_matrix[key]["correct"] += 1
            
    final_output = {
        "overall_report": report,
        "transfer_performance": {k: v["correct"]/v["count"] for k, v in transfer_matrix.items()},
        "confusion_matrix": cm.tolist()
    }

    with open(SAVE_REPORT, 'w') as f:
        json.dump(final_output, f, indent=4)

    print(f"\n评估完成！报告已保存至: {SAVE_REPORT}")
    print("\n--- 风格迁移成功率 (Classifier Accuracy) ---")
    for k, v in final_output["transfer_performance"].items():
        print(f"{k}: {v:.2%}")

if __name__ == "__main__":
    run_evaluation()