# Infra.md — 远程实验执行基础设施高效指南

> 本指南记录远程 RTX 3060 (Windows) 实验执行的高效模式，避免重复踩坑。
> 远程: `ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62`
> 代码目录: `I:\Github\Latent_Style\SchrodingerBridge`
> Python: `"C:\Program Files\Python312\python.exe"` (路径含空格，bat 中必须用双引号)

---

## 一、SSH 执行的核心陷阱与规避

### 1.1 引号嵌套地狱（最大痛点）
**问题**: 通过 SSH 执行 PowerShell 命令时，路径反斜杠 + 引号嵌套导致解析失败。
**错误示例**:
```bash
# 这样会失败 — PowerShell 内层引号与 SSH 外层引号冲突
ssh admin@host "powershell -Command \"& 'C:\Program Files\Python312\python.exe' script.py\""
```

**正确模式: 永远用文件传输，不在 SSH 内联复杂命令**
```bash
# 1. 本地写 .bat 或 .ps1 文件
# 2. scp 传输到远程
scp -P 2222 script.bat admin@host:"I:/path/script.bat"
# 3. SSH 执行文件
ssh admin@host "powershell -ExecutionPolicy Bypass -File I:\path\script.ps1"
```

### 1.2 后台启动模式
**问题**: `Start-Process` + SSH 引号嵌套不稳定，wmic 在新 Windows 版本可能被弃用。
**推荐: 前台 non-blocking 执行 + CheckCommandStatus 轮询**
```bash
# 本地 RunCommand: blocking=false, command_type=long_running_process
ssh admin@host "powershell -ExecutionPolicy Bypass -File I:\path\script.ps1"
# 然后用 CheckCommandStatus 轮询输出，无需 sleep
```

### 1.3 Python 路径空格
**问题**: `C:\Program Files\Python312\python.exe` 通过 SSH 执行时 `Program` 后空格导致命令截断。
**修复**: bat 文件中用双引号包裹: `"C:\Program Files\Python312\python.exe"`

---

## 二、高效轮询模式（替代固定 sleep）

**痛点**: 固定 `Start-Sleep 300` 浪费时间 — 任务可能 120s 就完成了。

**模式: exponential backoff 轮询**
```
1. 启动 non-blocking 命令 (blocking=false, wait_ms_before_async=5000-10000)
2. CheckCommandStatus 检查输出 (wait_ms_before_check=0)
3. 如果还在运行，等待递增时间后再次检查 (30s → 60s → 120s → 240s)
4. 看到 "=== ALL COMPLETE ===" 立即处理结果，不继续 sleep
```

**时间预算参考（远程 RTX 3060）**:
| 任务 | 耗时 | 说明 |
|------|------|------|
| 10ep 训练 (bf16, batch=96) | ~3 分钟 | D5 数据集 |
| 10ep 训练 (fp32, batch=96) | ~6 分钟 | 无 AMP |
| CLIP-S + LPIPS 评估 (750图) | ~4 分钟 | batch=2, 含生成 |
| DINO 评估 (750图) | ~45 秒 | max_refs=30 |
| VAE decode | ~1.6 秒 | 750 latents |
| uint8_cpu_copy | ~38 秒 | **硬瓶颈**, 750 images GPU→CPU |

**总评估时间**: 生成(55s) + uint8_copy(38s) + CLIP(4s) + LPIPS(1.6s) + DINO(45s) ≈ **2.5 分钟**
- 如果复用已生成图像 (`--reuse_generated`): 跳过生成和 uint8_copy，仅 ~50 秒

---

## 三、评估配置协议（关键！）

### 3.1 主表评估必须用 `--config_override`，不是 `--config`
```bash
# 正确: --config_override 只覆盖特定字段
--config_override configs/eval_adain_20.json

# 错误: --config 替换整个配置，会丢失 checkpoint 内嵌配置
--config configs/eval_adain_20.json
```

### 3.2 `eval_adain_20.json` 内容
```json
{
  "model": {
    "endpoint_adain_scale": 2.0
  }
}
```
- `config_override` 通过 `merge_config_dicts` (deep_merge) 合并到 checkpoint config 上
- 只需包含要覆盖的字段，其余继承 checkpoint 内嵌配置

### 3.3 主表配置链
```
src/default_config.json (合并基线, endpoint_adain_scale=1.0 训练默认)
  → configs/exp_brk_a_ll03_10ep.json (brk_a 训练配置)
  → checkpoint: exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt
  → 评估时 --config_override configs/eval_adain_20.json (adain=2.0)
```

**主表指标** (adain=2.0): DINO-S=0.4859, CLIP-S=0.7075, LPIPS=0.2583, DINO-C=0.8287
**训练默认** (adain=1.0): DINO-S=0.4832, CLIP-S=0.7180, LPIPS=0.2938, DINO-C=0.8000

### 3.4 复用已生成图像
```bash
--reuse_generated  # 跳过生成步骤，直接评估 output_dir/images 中的图像
```
- 用于后处理实验（WCT、后处理 ablation）
- 节省 ~93 秒（生成 + uint8_copy）

---

## 四、标准实验 Pipeline 模板

### 4.1 训练 + 评估一体化脚本
```powershell
# scripts/_pipeline_template.ps1
$ErrorActionPreference = "Continue"
$Root = "I:\Github\Latent_Style\SchrodingerBridge"
$Py = "C:\Program Files\Python312\python.exe"
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

$TestDir = "I:\datasets\wikiart_distinct5_samam_512_classview\test"
$HfCache = "C:\Users\Administrator\.cache\huggingface\hub"

function Run-One([string]$Tag, [string]$Config, [string]$ConfigOverride = "$Root\configs\eval_adain_20.json") {
    $CkptDir = "$Root\exp\$Tag"
    $LogDir = "$CkptDir\logs"
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

    # 训练
    Write-Output "=== [$Tag] TRAIN START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    & $Py -u "$Root\src\run.py" --config "$Root\configs\$Config" 2>&1 > "$LogDir\train.log"
    if ($LASTEXITCODE -ne 0) { Write-Output "=== [$Tag] TRAIN FAILED ==="; return }

    # CLIP/LPIPS 评估 (adain=2.0)
    $Ckpt = "$CkptDir\epoch_0010.pt"
    Write-Output "=== [$Tag] EVAL START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    & $Py -u "$Root\src\utils\run_evaluation.py" `
        --config_override $ConfigOverride `
        --checkpoint $Ckpt --output $CkptDir `
        --save_generated_images --batch_size 2 `
        --ref_feature_batch_size 2 --clip_hf_cache_dir $HfCache 2>&1 > "$LogDir\eval.log"
    if ($LASTEXITCODE -ne 0) { Write-Output "=== [$Tag] EVAL FAILED ==="; return }

    # DINO 评估
    Write-Output "=== [$Tag] DINO START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    & $Py -u "$Root\_compute_dino.py" `
        --images_dir "$CkptDir\images" --test_dir $TestDir --dataset wikiart `
        --output "$CkptDir\dino.json" --hf_cache $HfCache --max_refs 30 2>&1 > "$LogDir\dino.log"
    Write-Output "=== [$Tag] ALL COMPLETE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
}
```

### 4.2 只评估（复用 checkpoint）
```powershell
# 不需要训练，只跑评估
$Ckpt = "$Root\exp\dino_s_break\brk_a_ll03_10ep\epoch_0010.pt"
& $Py -u "$Root\src\utils\run_evaluation.py" `
    --config_override "$Root\configs\eval_adain_20.json" `
    --checkpoint $Ckpt --output $EvalDir ...
```

---

## 五、结果提取快捷方法

### 5.1 从 summary.json 提取 CLIP-S / LPIPS
关键路径: `analysis.all_pairs_overview`
```json
{
  "analysis": {
    "all_pairs_overview": {
      "clip_style": 0.7075,   // ← CLIP-S
      "content_lpips": 0.2583  // ← LPIPS
    }
  }
}
```

### 5.2 从 dino.json 提取 DINO-C / DINO-S
```json
{
  "dino_content": 0.8287,  // ← DINO-C
  "dino_style": 0.4859     // ← DINO-S
}
```

### 5.3 SCP 下载结果
```bash
scp -P 2222 admin@host:"I:/path/summary.json" local.json
scp -P 2222 admin@host:"I:/path/dino.json" local_dino.json
```

---

## 六、文件同步清单

实验前必须同步的文件:
```
src/default_config.json          → I:/Github/.../src/
src/run.py, src/model.py, src/flow.py → I:/Github/.../src/
configs/*.json                   → I:/Github/.../configs/
scripts/_pipeline_*.ps1          → I:/Github/.../scripts/
_compute_dino.py                 → I:/Github/.../  (根目录)
```

**批量 SCP**:
```bash
scp -P 2222 file1 file2 file3 admin@host:"I:/path/"
```

---

## 七、显存控制

| 阶段 | 显存上限 | 配置 |
|------|---------|------|
| 训练 | 11.2 GB | batch_size=96, bf16 AMP |
| 评估 | 7 GB | batch_size=2, ref_feature_batch_size=2 |
| DINO | 3 GB | batch_size=8 |

**OOM 处理**: 降低 batch_size，num_workers=0, pin_memory=False, persistent_workers=False

---

## 八、常见错误与修复

| 错误 | 原因 | 修复 |
|------|------|------|
| `unrecognized arguments: --full_eval_batch_size` | run_evaluation.py 不支持此参数 | 移除，用 `--batch_size` |
| `'C:\Program' 不是内部命令` | Python 路径空格未引用 | bat 中用 `"C:\Program Files\..."` |
| DINO-S 偏低 0.003 | 用了 `--config` 而非 `--config_override` | 改用 `--config_override` |
| 评估结果全 0 | `--config` 替换了整个配置 | 用 `--config_override` 只覆盖字段 |
| PowerShell Start-Process 失败 | SSH 引号嵌套 | 用 .ps1 文件 + `powershell -File` |
| DataLoader OOM | num_workers>0 | num_workers=0, pin_memory=False |
