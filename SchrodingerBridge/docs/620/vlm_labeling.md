# 620 风格数据集 VLM 文字打标规范与使用说明

为了在多模态架构设计中实现风格特征的高效融合（DINO 图像特征与 CLIP 文本特征联合注入），我们设计并启动了对 WikiArt 训练集的全量离线 VLM 文字打标工作。本文档用以说明已完成的工程实现、Prompt 设计逻辑以及后续在模型训练中的对接使用指南。

---

## 1. 我们做了什么 (已完成工作)

为了处理 5000 张规模的训练集图像标注，我们开发了专门的标注工具 [generate_style_captions_vlm.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/tools/generate_style_captions_vlm.py)，并针对生成模型训练的特殊需求进行了多项工程与 Prompt 优化：

### A. Prompt 设计优化 (Tags 纯净格式化)
为了确保提取出的文本直接用于扩散模型/Flow Matching 的 Cross-Attention 文本条件对齐，避免自然语言前缀（如 “这张图表现了...”）带来的文本噪声，我们设计了 **Few-shot Comma-separated Tags** 提示词：
*   **System Prompt**:
    > "You are an expert art critic and colorist. Your task is to extract ONLY the visual style of an image. You must respond with a clean, comma-separated list of keywords or short descriptive phrases. Do NOT output any conversational filler, greetings, or introductory sentences."
*   **User Prompt**:
    > "Analyze the image and describe its artistic style, medium, color palette, lighting, texture, and brushwork. CRITICAL REQUIREMENT: Do NOT mention any subjects, objects, characters, or scenes. Output ONLY the comma-separated list of style-related terms. Example output: 'oil painting, impressionism, thick impasto brushwork, warm golden hour lighting, pastel color palette'."

### B. 图像分辨率与 API 费用优化
针对数据集本身为 `512*512` 的情况，将默认的 `--max-edge` 从 `1024` 降低为 `512`。此举大幅减少了 Base64 Payload 尺寸，显著降低了 VLM API 的 Token 计费，并将平均 API 响应时间缩短了约 **60%**（降至单张约 4s）。

### C. 健壮性与自动重试机制
在大规模标注场景下，针对连接挂起、限流、安全审查阻断等问题，脚本内置了如下健壮方案：
1.  **分流超时设置**：将 `timeout` 分离为 `(15, timeout)`。15 秒内必须完成 TCP 握手建连，防止网络链路挂起，余下时间留给 VLM 做长推理。
2.  **安全审查短路**：当触发 MaaS 的安全或违规拦截时，API 会返回 200 或错误状态且含有 "抱歉/sorry/无法处理" 等字样。脚本进行**主动关键词短路检测**，一旦命中则返回固定占位符并**停止盲目重试**，保护 API 额度与整体进度。
3.  **多线程并发 + 断点续传**：使用多线程并发拉取标注，实时将增量结果写入 `.jsonl`。即便中断，重启时也会自动读取已处理过的图像进行断点续传。
4.  **串行降级兜底 (Fallback Pass)**：在多线程大图（512）标注结束后，若有残留失败样本，脚本会开启第二阶段串行扫描，强制以 `max_edge=512` 且更低 JPEG 质量 (`75`) 加上额外 `30s` 超时余量进行串行补发，确保标注完整率。

---

## 2. 本地数据集与文件位置

*   **打标数据源**（包含完整真实图像且 Windows 可读的目录）：
    [F:\wikiart_distinct5_samam_512_classview_real\train](file:///F:/wikiart_distinct5_samam_512_classview_real/train)
*   **MaaS API 密钥**：在 remote 端使用，或在本地命令行中使用 `--api-key` 手动传入。
*   **标注输出产物**：
    *   **最终 compiled JSON 映射表**：[train_style_captions.json](file:///F:/wikiart_distinct5_samam_512_classview_real/train_style_captions.json)
    *   **流式 JSONL 中间日志**：[train_style_captions.jsonl](file:///F:/wikiart_distinct5_samam_512_classview_real/train_style_captions.jsonl)
    *(注：本地 F 盘在 remote WSL 中自动映射为 `/mnt/f/` 路径，WSL 训练端无需复制即可直接读取。)*

---

## 3. 后续如何使用 (使用指南)

### A. 若需重启/增量运行脚本
如需在本地对未完成的任务进行重新扫描或追加标注，运行：
```bash
python SchrodingerBridge/tools/generate_style_captions_vlm.py \
  --dataset-dir F:\wikiart_distinct5_samam_512_classview_real\train \
  --output-json F:\wikiart_distinct5_samam_512_classview_real\train_style_captions.json \
  --output-jsonl F:\wikiart_distinct5_samam_512_classview_real\train_style_captions.jsonl \
  --workers 12 \
  --api-key <YOUR_XF_MAAS_API_KEY>
```

### B. 在 PyTorch Dataset 中加载
在 `AdaCUTLatentDataset`（或 620 新增的 Dataset）的 `__init__` 中读入生成好的 JSON。键为 `style_subdir/image_name`（例如 `Early_Renaissance/Early_Renaissance__andrea-del-castagno_david-with-the-head-of-goliath.jpg`）：

```python
import json
from pathlib import Path

class StyleMultimodalDataset(Dataset):
    def __init__(self, data_root, captions_json_path, ...):
        # ...
        with open(captions_json_path, "r", encoding="utf-8") as f:
            self.style_captions = json.load(f)
            
    def __getitem__(self, index):
        # ... 获取 content 潜变量 z_content, target 潜变量 z_target, dino 特征
        # 获取相对路径标识符 (由 subdir 与 stem 组成)
        rel_path = f"{style_subdir}/{image_name}"
        caption = self.style_captions.get(rel_path, "style, artwork") # 缺省占位词
        
        return {
            "z_content": z_content,
            "z_target": z_target,
            "style_caption": caption,
            "style_dino_features": style_dino_features,
            # ...
        }
```

### C. 训练中的 Modality Dropout 与 Null Token
如架构方案所设计，训练时我们需要在 forward 时对 Text / Image 条件进行联合的 Dropout（图文双全、图文全丢、仅有文字、仅有图片），以支持推理时的任意组合以及 **Classifier-Free Guidance (CFG)**：

```python
class OmniModalStyleTransfer(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        # 冻结的 CLIP 和 DINO
        self.clip_text = FrozenCLIPTextEncoder()
        self.dino_image = FrozenDINOv2Encoder()
        
        # 可学习映射层
        self.proj_text = nn.Linear(768, d_model)
        self.proj_image = nn.Linear(1024, d_model)
        
        # 可学习空白占位符
        self.null_text = nn.Parameter(torch.randn(1, 77, d_model) * 0.02)
        self.null_image = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)

    def forward(self, z_t, t, captions, style_images):
        batch_size = z_t.shape[0]
        
        # 投影特征
        feat_text = self.proj_text(self.clip_text(captions))   # [B, 77, d_model]
        feat_image = self.proj_image(self.dino_image(style_images)) # [B, 256, d_model]
        
        # Modality Dropout
        rand_probs = torch.rand(batch_size, device=z_t.device)
        for i in range(batch_size):
            p = rand_probs[i]
            if p < 0.1:     # 10% 概率：图文全丢 (无条件)
                feat_text[i] = self.null_text[0]
                feat_image[i] = self.null_image[0]
            elif p < 0.3:   # 20% 概率：仅有文字 (Image Dropout)
                feat_image[i] = self.null_image[0]
            elif p < 0.5:   # 20% 概率：仅有图片 (Text Dropout)
                feat_text[i] = self.null_text[0]
            else:           # 50% 概率：图文双全
                pass
                
        # 拼接成为 DiT 的 Cross-Attention 输入序列 (长度 77 + 256 = 333)
        context = torch.cat([feat_text, feat_image], dim=1) 
        
        # 传入 Backbone DiT 进行速度场预测
        v_pred = self.backbone(z_t, t, context)
        return v_pred
```
