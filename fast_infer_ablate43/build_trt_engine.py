from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX with trtexec")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--min-shape", type=str, default="image:1x3x256x256,style_id:1")
    parser.add_argument("--opt-shape", type=str, default="image:1x3x512x512,style_id:1")
    parser.add_argument("--max-shape", type=str, default="image:4x3x1024x1024,style_id:4")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workspace-mb", type=int, default=4096)
    args = parser.parse_args()

    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError("trtexec not found. Please install TensorRT and add trtexec to PATH.")

    args.engine.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        trtexec,
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        f"--memPoolSize=workspace:{max(16, int(args.workspace_mb))}",
        f"--minShapes={args.min_shape}",
        f"--optShapes={args.opt_shape}",
        f"--maxShapes={args.max_shape}",
        "--builderOptimizationLevel=5",
        "--best",
    ]
    if args.fp16:
        cmd.append("--fp16")

    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[ok] TensorRT engine built: {args.engine.resolve()}")


if __name__ == "__main__":
    main()
