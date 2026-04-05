# Ablate43 高速推理链路（独立目录）

这个目录是独立的推理/导出链路，默认绑定：
- checkpoint: `Cycle-NCE/Ablate43/Ablate43_S01_Baseline_Gold/epoch_0060.pt`
- model 定义: `Cycle-NCE/Ablate43/Ablate43_S01_Baseline_Gold/model.py`
- VAE: `stabilityai/sd-vae-ft-mse`

包含：
- `runtime.py`: 本地 CLI，支持 `image -> image` 与 `video -> video`
- `server.py`: HTTP 接口（FastAPI）
- `export_onnx.py`: 导出含 VAE 的全链 ONNX
- `build_trt_engine.py`: 基于 ONNX 用 `trtexec` 构建 TensorRT engine

## 1) 安装依赖

```powershell
pip install -r fast_infer_ablate43/requirements.txt
```

## 2) 图片推理

```powershell
python fast_infer_ablate43/runtime.py image `
  --input path/to/in.png `
  --output path/to/out.png `
  --style-id 2
```

可选参数：
- `--step-size 1.0`
- `--style-strength 1.0`
- `--no-compile` 关闭 `torch.compile`
- `--keep-exact-resolution` 不做 8 对齐（默认会内部对齐后再缩回原分辨率）

## 3) 视频推理（直接可用）

```powershell
python fast_infer_ablate43/runtime.py video `
  --input path/to/in.mp4 `
  --output path/to/out.mp4 `
  --style-id 3 `
  --batch-size 8
```

说明：
- 默认按批次推理视频帧，`batch-size` 越大吞吐越高，但显存占用更大。
- 输出视频默认 `mp4v` 编码，便于直接落盘。

## 4) HTTP 接口（图片/视频）

```powershell
uvicorn fast_infer_ablate43.server:app --host 0.0.0.0 --port 18080
```

- 健康检查：`GET /health`
- 图片推理：`POST /infer/image?style_id=1`
  - form-data: `image=<file>`
  - 返回：`image/png`
- 视频推理：`POST /infer/video?style_id=1&batch_size=8`
  - form-data: `video=<file>`
  - 返回：`video/mp4`

## 5) 导出 ONNX（含 VAE 前后处理）

```powershell
python fast_infer_ablate43/export_onnx.py `
  --output fast_infer_ablate43/ablate43_full_pipeline.onnx `
  --height 512 --width 512 --batch 1 --dynamic
```

输入定义：
- `image`: `NCHW`, float32, 范围 `[-1, 1]`
- `style_id`: int64, shape `[N]`
输出定义：
- `output`: `NCHW`, float32, 范围 `[0, 1]`

## 6) 编译 TensorRT engine（GPU 字节码）

先确保系统可执行 `trtexec`。

```powershell
python fast_infer_ablate43/build_trt_engine.py `
  --onnx fast_infer_ablate43/ablate43_full_pipeline.onnx `
  --engine fast_infer_ablate43/ablate43_full_pipeline_fp16.engine `
  --fp16
```

## 风格 ID 对照（来自 checkpoint config）

- `0: photo`
- `1: Hayao`
- `2: monet`
- `3: vangogh`
- `4: cezanne`
