import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. 配置区域 (请在此处修改)
# ==========================================
DATASET_ROOT = "../style_data/train"  # <--- 数据集根目录
TEST_ROOT = "../style_data/test"      # <--- 测试集根目录 (可选)

# 🔥 关键修改：在这里指定你想要训练的文件夹名称
# 必须与你的文件夹名完全一致。顺序决定了 Label ID (0, 1, 2...)
SELECTED_CLASSES = ["vangogh", "photo"] 

SAVE_PATH = "./style_judge_resnet18_selective.pt"
BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-4
EVAL_ON_TEST = True  # 是否在测试集上评估

# ==========================================
# 2. 自定义数据集类 (只读取指定文件夹)
# ==========================================
class SelectiveWikiArtDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        """
        Args:
            root_dir (string): 数据集根目录
            classes (list): 需要包含的类别文件夹名称列表
            transform (callable, optional): 图像预处理
        """
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.samples = []
        
        # 建立 类别名 -> 索引 的映射 (例如: monet->0, vangogh->1)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        print(f"正在扫描指定类别: {classes} ...")
        
        for class_name in classes:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.exists(class_path):
                print(f"⚠️ 警告: 文件夹 '{class_name}' 不存在，跳过！")
                continue
                
            # 遍历该文件夹下的所有图片
            count = 0
            for root, _, fnames in os.walk(class_path):
                for fname in fnames:
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        path = os.path.join(root, fname)
                        # 存储 (图片路径, 标签索引)
                        self.samples.append((path, self.class_to_idx[class_name]))
                        count += 1
            print(f" -> 类别 '{class_name}': 找到 {count} 张图片")
            
        print(f"总计加载样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # 打开图片并转为 RGB (防止灰度图报错)
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"无法读取图片: {path}, Error: {e}")
            # 如果出错，随机返回一张别人的图防止崩坏（简单容错）
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        return image, label

# ==========================================
# 3. 图像预处理 (256 RandomResizedCrop)
# ==========================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.6, 1.0)), # 核心：不同分辨率归一化 + 强迫看局部
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 测试集不需要 augmentation，直接缩放和归一化
eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 4. 初始化数据与模型
# ==========================================
def evaluate_on_test(model, test_dataset, classes, device):
    """在测试集上评估模型"""
    if len(test_dataset) == 0:
        print("⚠️ 测试集为空，跳过测试评估")
        return
    
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    model.eval()
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    
    print("\n" + "="*50)
    print("📊 在测试集上进行评估...")
    print("="*50)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    test_acc = 100. * correct / total
    print(f"\n✓ 测试集准确率: {test_acc:.2f}%")
    
    # 生成详细报告
    report = classification_report(
        y_true, y_pred, 
        target_names=classes,
        digits=4
    )
    print("\n--- 分类报告 ---")
    print(report)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print("\n--- 混淆矩阵 ---")
    print(cm)
    print("="*50)

# ==========================================
# 5. 初始化数据与模型
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 使用自定义的数据集
    train_dataset = SelectiveWikiArtDataset(
        root_dir=DATASET_ROOT, 
        classes=SELECTED_CLASSES, 
        transform=train_transform
    )
    
    # 防止空数据集报错
    if len(train_dataset) == 0:
        print("错误: 未找到任何图片，请检查路径和文件夹名称。")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # ==========================================
    # 6. 构建模型 (改为 ResNet-18 + 冻结 + Dropout)
    # ==========================================
    print(f"正在加载 ResNet-18 预训练模型...")

    # 1. 加载 ResNet-18 (权重更小，更适合小数据集)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 2. 🔥 核心动作：冻结骨干网络 (Freeze Backbone)
    # 这一步锁死前置层，只让它们提取特征，不让它们更新参数
    for param in model.parameters():
        param.requires_grad = False

    # 3. 获取全连接层的输入维度
    # ResNet-50 是 2048，但 ResNet-18 只有 512，所以必须动态获取
    num_ftrs = model.fc.in_features 

    # 4. 🔥 核心动作：重写分类头 (加入 Dropout)
    # 这里的 requires_grad 默认为 True，所以优化器只会更新这几层
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # 50% 概率丢弃，强力防止过拟合
        nn.Linear(num_ftrs, 256), # 加一个中间层过渡一下（可选，对于几百张图推荐加）
        nn.ReLU(),
        nn.Dropout(0.2),  # 再加一层轻微 Dropout
        nn.Linear(256, len(SELECTED_CLASSES)) # 最终输出层
    )

    model = model.to(device)

    # ==========================================
    # 7. 训练循环
    # ==========================================
    criterion = nn.CrossEntropyLoss()
    # 注意：只传入 model.fc.parameters()，因为其他层冻结了，传了也没用
    # 学习率可以稍微给大一点点 (1e-3)，因为只训练最后几层
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    print("开始训练...")
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i}] Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%")

        epoch_acc = 100. * correct / total
        print(f"==> Epoch {epoch+1} 完成. 平均 Loss: {running_loss/len(train_loader):.4f}, 准确率: {epoch_acc:.2f}%")
        
        # 每个 Epoch 保存一次，防止意外中断
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"模型已保存至: {SAVE_PATH}")
    
    # ==========================================
    # 8. 测试集评估 (可选)
    # ==========================================
    if EVAL_ON_TEST and os.path.exists(TEST_ROOT):
        print("\n🔍 加载测试集...") 
        test_dataset = SelectiveWikiArtDataset(
            root_dir=TEST_ROOT,
            classes=SELECTED_CLASSES,
            transform=eval_transform
        )
        evaluate_on_test(model, test_dataset, SELECTED_CLASSES, device)
    else:
        if not EVAL_ON_TEST:
            print("\n⚠️ 跳过测试集评估 (EVAL_ON_TEST=False)")
        else:
            print(f"\n⚠️ 测试集路径不存在: {TEST_ROOT}")

if __name__ == '__main__':
    main()