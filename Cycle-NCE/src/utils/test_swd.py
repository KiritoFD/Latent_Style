import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

class ChannelLifter(nn.Module):
    """
    绾补鐨勯€氶亾鍗囩淮鍣?(1x1 Conv MLP)銆?
    涓嶇湅閭诲煙锛屽彧璐熻矗鎶?4 缁寸殑鍘嬬缉鐗瑰緛 '瑙ｅ帇' 鍒伴珮缁存祦褰€?
    """
    def __init__(self, dim_in=4, dim_mid=64, dim_out=256):
        super().__init__()
        self.net = nn.Sequential(
            # 绗竴灞傦細瑙ｅ帇 + 褰掍竴鍖?(鍏抽敭!)
            nn.Conv2d(dim_in, dim_mid, 1),
            nn.GroupNorm(4, dim_mid), # GroupNorm 鏁堟灉閫氬父姣?BatchNorm 濂?
            nn.SiLU(inplace=True),
            
            # 绗簩灞傦細鏄犲皠鍒伴珮缁?
            nn.Conv2d(dim_mid, dim_out, 1)
        )
        
        # 姝ｄ氦鍒濆鍖栵細璁╃壒寰佸敖鍙兘姝ｄ氦锛屼簰涓嶇浉鍏?
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
                
        # 鍐荤粨
        self.eval()
        for p in self.parameters(): p.requires_grad = False

    def forward(self, x):
        return self.net(x * 0.13025)

def compute_gram(feats):
    """
    璁＄畻 Gram 鐭╅樀锛氭崟鎹夐€氶亾闂寸殑鐩稿叧鎬?(绾圭悊/椋庢牸鎸囩汗)
    Input: [B, C, H, W]
    Output: [B, C*C] (Flattened Gram)
    """
    b, c, h, w = feats.shape
    f = feats.view(b, c, -1)
    # [B, C, HW] @ [B, HW, C] -> [B, C, C]
    # 闄や互 HW 褰掍竴鍖栵紝淇濊瘉瀵瑰昂瀵镐笉鏁忔劅
    gram = torch.bmm(f, f.transpose(1, 2)) / (h * w)
    
    # 鍙栦笂涓夎鐭╅樀 (鍘婚噸)
    # 灞曞钩
    return gram.view(b, -1)

def compute_swd_on_features(feats, num_projections=64, device='cuda'):
    """
    鍦ㄧ壒寰佺┖闂磋绠?SWD (浣滀负瀵规瘮)
    """
    b, c, h, w = feats.shape
    f = feats.view(b, c, -1).permute(0, 2, 1) # [B, HW, C]
    
    # 闅忔満鎶曞奖
    proj_mat = torch.randn(c, num_projections, device=device)
    proj_mat = proj_mat / torch.norm(proj_mat, dim=0, keepdim=True)
    
    projected = torch.matmul(f, proj_mat)
    sorted_proj, _ = torch.sort(projected, dim=1)
    
    # 涓嬮噰鏍?
    indices = torch.linspace(0, h*w-1, 64, device=device).long()
    return sorted_proj[:, indices, :].reshape(b, -1)

def run_lifted_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path("../../../sdxl-256")
    styles = ["photo", "Hayao", "vangogh", "monet"]
    
    print(f"馃殌 Running Lifted Analysis (1x1 MLP + Norm) on {device}...")
    
    # 1. 鍑嗗鏁版嵁
    all_lats, labels = [], []
    for s in styles:
        files = list((data_root / s).glob("*.pt"))[:300]
        if not files: continue
        all_lats.append(torch.stack([torch.load(f, map_location=device, weights_only=True) for f in files]))
        labels.extend([s] * len(files))
    
    if not all_lats: return
    X = torch.cat(all_lats)
    
    # 2. 瀹氫箟 Lifted Space
    # 灏濊瘯鍗囩淮鍒?256锛岀湅 Gram 鏄惁鏈夋晥
    lifter = ChannelLifter(dim_in=4, dim_mid=64, dim_out=256).to(device)
    
    # 3. 鎻愬彇鐗瑰緛 & 璁＄畻涓ょ鎸囨爣
    print("Extracting features (4 -> 256 dim)...")
    
    feat_gram_list = []
    feat_swd_list = []
    
    batch_size = 64
    for i in range(0, len(X), batch_size):
        with torch.no_grad():
            batch_x = X[i:i+batch_size]
            # 鍗囩淮
            lifted_x = lifter(batch_x)
            
            # 鏂规硶 A: Gram Matrix
            gram = compute_gram(lifted_x)
            feat_gram_list.append(gram.cpu().numpy())
            
            # 鏂规硶 B: SWD (鍚岀淮搴﹀姣?
            swd = compute_swd_on_features(lifted_x, device=device)
            feat_swd_list.append(swd.cpu().numpy())
            
    feat_gram = np.concatenate(feat_gram_list)
    feat_swd = np.concatenate(feat_swd_list)
    
    # 4. 璇勪及瀵规瘮
    print("\n" + "="*50)
    print(f"{'Method':<25} | {'Silhouette':<10} | {'Verdict'}")
    print("-" * 50)
    
    # Compute both scores before winner comparison.
    scaler = StandardScaler()
    gram_norm = scaler.fit_transform(feat_gram)
    swd_norm = scaler.fit_transform(feat_swd)
    score_gram = silhouette_score(gram_norm, labels)
    score_swd = silhouette_score(swd_norm, labels)
    print(f"{'Projected Gram Matrix':<25} | {score_gram: .4f}    | {'WINNER?' if score_gram > score_swd else ''}")
    print(f"{'Projected SWD':<25} | {score_swd: .4f}    | {'WINNER?' if score_swd > score_gram else ''}")
    print("="*50)

    # 5. 鍙鍖栨渶浣崇粨鏋?
    best_feats = gram_norm if score_gram > score_swd else swd_norm
    best_name = "Projected Gram" if score_gram > score_swd else "Projected SWD"
    
    pca = PCA(n_components=2).fit_transform(best_feats)
    plt.figure(figsize=(10, 8))
    for s in styles:
        mask = np.array(labels) == s
        plt.scatter(pca[mask, 0], pca[mask, 1], label=s, s=15, alpha=0.6)
    plt.title(f"Best Metric: {best_name}\n(1x1 Conv -> Norm -> 256ch)\nScore: {max(score_gram, score_swd):.4f}")
    plt.legend()
    plt.savefig("lifted_metric_result.png")
    print(f"Saved visualization for {best_name}")

if __name__ == "__main__":
    run_lifted_comparison()
