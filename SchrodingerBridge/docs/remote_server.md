# Remote Server Guide

## Connection

Server: `100.115.18.62:2222`, user `administrator`, SSH key auth.

```bash
ssh administrator@100.115.18.62 "command"
```

## Long-Running Tasks

**Use `schtasks`, NOT `Start-Process`** — `Start-Process` via SSH ties the process to the SSH session, which dies when disconnected.

```bash
# Create and run a scheduled task
ssh administrator@100.115.18.62 "schtasks /create /tn TaskName /tr \"path\to\script.bat\" /sc once /st 00:00 /f && schtasks /run /tn TaskName"

# Check task status
ssh administrator@100.115.18.62 "schtasks /query /tn TaskName"

# Stop a task
ssh administrator@100.115.18.62 "schtasks /end /tn TaskName"
```

## GameViewer (UU远程)

Service: `GameViewerService`. Processes: `GameViewerService.exe`, `GameViewerServer.exe`, `GameViewerHealthd.exe`.

Restart:
```bash
ssh administrator@100.115.18.62 "sc stop GameViewerService & timeout /t 5 /nobreak >nul & sc start GameViewerService"
```

Or run: `I:\Github\Latent_Style\exp\highres\restart_gameviewer.bat`

## SaMST Pipeline

Location: `I:\Github\Latent_Style\exp\highres\`

Scripts:
- `samst_full_pipeline.py` — train 5 styles → inference → CLIP+LPIPS eval
- `run_samst.py` — single style training
- `run_samst_inference_all.py` — inference all checkpoints
- `samst_clip_dino_eval.py` — CLIP+LPIPS evaluation

Launch via schtasks:
```bash
ssh administrator@100.115.18.62 "schtasks /create /tn SaMSTPipeline /tr \"I:\Github\Latent_Style\exp\highres\samst_pipeline_wrapper.bat\" /sc once /st 00:00 /f && schtasks /run /tn SaMSTPipeline"
```

Status: `I:\Github\Latent_Style\exp\highres\samst_pipeline.log`

## Directory Structure (Remote)

```
I:\Github\Latent_Style\
├── SchrodingerBridge\          # SB codebase
├── Related_Works\repos\
│   ├── S2WAT-main\             # S2WAT code + VGG weights
│   └── SaMST-main\             # SaMST code + train_dataset
├── exp\highres\                # Training scripts + results
│   ├── samst\{style}\checkpoints\  # SaMST checkpoints
│   ├── samst\{style}\outputs\      # Inference outputs
│   └── samst\clip_lpips_eval.csv   # Eval results
└── SchrodingerBridge\scale\datasets\  # 1024px datasets
    ├── wikiart_1024_matched\    # 5-style matched images (5040)
    ├── wikiart_1024_27test\     # 27-style test set
    └── wikiart_1024_27support\  # 27-style support set
```

## GPU

12GB VRAM. Key limits:
- SaMST 1024 bs=1: ~7.3 GB
- SaMST 1024 bs=4: ~12 GB (near OOM)
- S2WAT 256 bs=1: ~4.3 GB
