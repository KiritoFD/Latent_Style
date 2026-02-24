import argparse
import json
from pathlib import Path

import cv2
import MNN
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_path(base_dir: Path, raw: str) -> Path:
    p = Path(str(raw).strip())
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_images(test_dir: Path, style_subdirs: list[str]) -> list[Path]:
    paths: list[Path] = []
    if style_subdirs:
        for sub in style_subdirs:
            d = test_dir / sub
            if not d.exists():
                continue
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES:
                    paths.append(p)
    else:
        for p in sorted(test_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES:
                paths.append(p)
    return paths


def _infer_one(interpreter, session, input_tensor, style_tensor, output_tensor, img_path: Path, style_id_value: int) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_data = (img.astype(np.float32) / 127.5) - 1.0
    input_data = input_data.transpose((2, 0, 1))[np.newaxis, :]
    tmp_input = MNN.Tensor(
        input_data.shape,
        MNN.Halide_Type_Float,
        input_data,
        MNN.Tensor_DimensionType_Caffe,
    )
    input_tensor.copyFrom(tmp_input)

    # MNN.Halide_Type_Int expects int32 host data.
    style_id = np.ascontiguousarray(np.array([style_id_value], dtype=np.int32))
    tmp_style = MNN.Tensor(
        style_id.shape,
        MNN.Halide_Type_Int,
        style_id,
        MNN.Tensor_DimensionType_Caffe,
    )
    style_tensor.copyFrom(tmp_style)

    interpreter.runSession(session)
    host_output = MNN.Tensor(
        output_tensor.getShape(),
        MNN.Halide_Type_Float,
        np.zeros(output_tensor.getShape(), dtype=np.float32),
        MNN.Tensor_DimensionType_Caffe,
    )
    output_tensor.copyToHostTensor(host_output)

    out_data = host_output.getNumpyData()[0].transpose((1, 2, 0))
    out_img = out_data * 127.5 + 127.5
    out_img = np.clip(out_img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)


def verify_mnn() -> None:
    parser = argparse.ArgumentParser(description="Run MNN model on config.training.test_image_dir and save outputs.")
    parser.add_argument("--config", type=str, default=str((_ROOT / "config.json").resolve()))
    parser.add_argument("--mnn", type=str, default=str((Path(__file__).resolve().parent / "200-mobile-core.mnn").resolve()))
    parser.add_argument("--out", type=str, default=str((_ROOT / "mnn_overfit50_outputs").resolve()))
    parser.add_argument("--style_id", type=int, default=1)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = _load_config(config_path)

    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    test_image_dir_raw = str(train_cfg.get("test_image_dir", "")).strip()
    if not test_image_dir_raw:
        raise ValueError("config.training.test_image_dir is empty.")
    test_dir = _resolve_path(_ROOT, test_image_dir_raw)
    if not test_dir.exists():
        raise FileNotFoundError(f"Resolved test_image_dir not found: {test_dir}")

    style_subdirs = [str(x) for x in data_cfg.get("style_subdirs", [])]
    images = _collect_images(test_dir, style_subdirs)
    if not images:
        raise RuntimeError(f"No images found under: {test_dir}")

    mnn_path = Path(args.mnn).resolve()
    if not mnn_path.exists():
        raise FileNotFoundError(f"MNN model not found: {mnn_path}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    interpreter = MNN.Interpreter(str(mnn_path))
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session, "input_rgb_fp32")
    style_tensor = interpreter.getSessionInput(session, "style_id")
    output_tensor = interpreter.getSessionOutput(session, "output_rgb_fp32")

    saved = 0
    for idx, img_path in enumerate(images):
        rel = img_path.relative_to(test_dir)
        out_name = f"{idx:04d}_{rel.as_posix().replace('/', '__')}_to_style{int(args.style_id)}.png"
        out_path = out_dir / out_name
        out_img = _infer_one(
            interpreter=interpreter,
            session=session,
            input_tensor=input_tensor,
            style_tensor=style_tensor,
            output_tensor=output_tensor,
            img_path=img_path,
            style_id_value=int(args.style_id),
        )
        cv2.imwrite(str(out_path), out_img)
        saved += 1

    print(f"Config: {config_path}")
    print(f"Input (overfit50): {test_dir}")
    print(f"MNN model: {mnn_path}")
    print(f"Saved images: {saved}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    verify_mnn()
