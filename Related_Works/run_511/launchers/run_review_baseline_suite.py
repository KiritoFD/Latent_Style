from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
RUN511_ROOT = THIS_DIR.parent
WORKSPACE_ROOT = RUN511_ROOT.parent.parent
REPOS_ROOT = RUN511_ROOT / "repos"
DEFAULT_OUTPUT_ROOT = RUN511_ROOT / "outputs" / "review_baseline_suite"
IMG_SIZE = 256
PYTHON_EXE = Path(os.environ.get("UV_PYTHON") or sys.executable)

METHODS = {
    "stytr2": {
        "launcher": THIS_DIR / "run_stytr2_750.py",
        "supports_preflight": False,
    },
    "cast": {
        "launcher": THIS_DIR / "run_cast_750.py",
        "supports_preflight": True,
    },
    "aesfa": {
        "launcher": THIS_DIR / "run_aesfa_750.py",
        "supports_preflight": False,
    },
    "aespa": {
        "launcher": THIS_DIR / "run_aespa_750.py",
        "supports_preflight": True,
    },
}


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _run(cmd: list[str], *, cwd: Path, log_path: Path, dry_run: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = "$ " + " ".join(str(x) for x in cmd)
    print(line, flush=True)
    if dry_run:
        with log_path.open("a", encoding="utf-8", errors="replace") as f:
            f.write(line + "\n")
        return 0
    with log_path.open("a", encoding="utf-8", errors="replace") as f:
        f.write(f"\n[{_timestamp()}]\n{line}\n")
        f.flush()
        return subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode


def _load_summary(run_root: Path) -> dict[str, Any]:
    path = run_root / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_stage_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary.get("runs", []) if isinstance(summary, dict) else []
    out: dict[str, Any] = {
        "train_status": "",
        "train_elapsed_sec": None,
        "infer_status": "",
        "infer_elapsed_sec": None,
        "generated_images": None,
    }
    for row in rows:
        stage = str(row.get("stage", ""))
        if stage == "train":
            out["train_status"] = row.get("status", "")
            out["train_elapsed_sec"] = row.get("elapsed_sec")
        elif stage == "infer":
            out["infer_status"] = row.get("status", "")
            out["infer_elapsed_sec"] = row.get("elapsed_sec")
            out["generated_images"] = row.get("images")
    return out


@contextmanager
def _push_sys_path(paths: list[Path]):
    old = list(sys.path)
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = old


@contextmanager
def _push_cwd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _safe_profile(model, inputs: tuple[Any, ...]) -> dict[str, Any]:
    try:
        params = int(sum(int(p.numel()) for p in model.parameters()))
    except Exception:
        params = None
    result = {"params": params, "flops": None, "macs": None, "backend": None}
    try:
        from thop import profile  # type: ignore

        macs, params = profile(model, inputs=inputs, verbose=False)
        result["params"] = int(params)
        result["macs"] = float(macs)
        result["flops"] = float(macs) * 2.0
        result["backend"] = "thop"
        return result
    except Exception:
        pass
    try:
        import torch

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.inference_mode():
            with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                _ = model(*inputs)
        total_flops = float(sum(float(getattr(evt, "flops", 0.0) or 0.0) for evt in prof.key_averages()))
        if total_flops > 0:
            result["flops"] = total_flops
            result["macs"] = total_flops / 2.0
            result["backend"] = "torch_profiler"
    except Exception:
        pass
    return result


def _purge_modules(prefixes: list[str]) -> None:
    for name in list(sys.modules.keys()):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            sys.modules.pop(name, None)


def _benchmark_forward(model, inputs: tuple[Any, ...], warmup: int, iters: int) -> dict[str, Any]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            _ = model(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(max(1, iters)):
            _ = model(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    batch_size = int(inputs[0].shape[0]) if inputs and hasattr(inputs[0], "shape") else 1
    images = max(1, iters) * batch_size
    return {
        "device": device,
        "batch_size": batch_size,
        "elapsed_sec": float(elapsed),
        "avg_sec_per_iter": float(elapsed / max(1, iters)),
        "avg_sec_per_img": float(elapsed / images),
        "throughput_img_per_sec": float(images / max(elapsed, 1e-8)),
        "peak_vram_mb": float(torch.cuda.max_memory_allocated() / (1024.0**2)) if device == "cuda" else 0.0,
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024.0**2)) if device == "cuda" else 0.0,
    }


def _styletr_profile(run_root: Path, warmup: int, iters: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    repo = REPOS_ROOT / "StyTR-2"
    _purge_modules(["models", "util", "function"])
    with _push_sys_path([repo]), _push_cwd(repo):
        import models.transformer as transformer
        import models.StyTR as StyTR

        vgg = StyTR.vgg
        vgg.load_state_dict(torch.load(repo / "experiments" / "vgg_normalised.pth", map_location="cpu"))
        vgg = nn.Sequential(*list(vgg.children())[:44])
        decoder = StyTR.decoder
        trans = transformer.Transformer()
        embedding = StyTR.PatchEmbed()

        max_iter = None
        summary = _load_summary(run_root)
        for row in summary.get("runs", []):
            if row.get("stage") == "train":
                max_iter = int(row.get("max_iter", 0) or 0)
        if not max_iter:
            max_iter = 1000
        ckpt_dir = run_root / "checkpoints" / "stytr2"
        decoder.load_state_dict(torch.load(ckpt_dir / f"decoder_iter_{max_iter}.pth", map_location="cpu"))
        trans.load_state_dict(torch.load(ckpt_dir / f"transformer_iter_{max_iter}.pth", map_location="cpu"))
        embedding.load_state_dict(torch.load(ckpt_dir / f"embedding_iter_{max_iter}.pth", map_location="cpu"))

        class Wrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = StyTR.StyTrans(vgg, decoder, embedding, trans, SimpleNamespace(hidden_dim=512))

            def forward(self, content, style):
                return self.net(content, style)

        model = Wrapper().eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        content = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        style = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        prof = _safe_profile(model, (content, style))
        bench = _benchmark_forward(model, (content, style), warmup, iters)
        return {**prof, **bench}


def _cast_profile(run_root: Path, warmup: int, iters: int) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    repo = REPOS_ROOT / "cast"
    _purge_modules(["models", "util", "data", "options"])
    with _push_sys_path([repo]), _push_cwd(repo):
        from options.test_options import TestOptions
        from models import create_model

        cmd = (
            f"--dataroot {str(run_root / 'dummy')} --name run511_cast --model cast "
            f"--checkpoints_dir {str(run_root / 'checkpoints')} --gpu_ids {'0' if torch.cuda.is_available() else '-1'} "
            "--results_dir ./results --num_test 1 --eval"
        )
        opt = TestOptions(cmd_line=cmd).parse()
        ckpt_dir = run_root / "checkpoints" / "run511_cast"
        epochs: list[int] = []
        for path in ckpt_dir.glob("*_net_AE.pth"):
            prefix = path.name.split("_net_AE.pth", 1)[0]
            if prefix.isdigit():
                epochs.append(int(prefix))
        if (ckpt_dir / "latest_net_AE.pth").exists():
            opt.epoch = "latest"
        elif epochs:
            opt.epoch = str(max(epochs))
        model_obj = create_model(opt)
        model_obj.setup(opt)
        model_obj.eval()

        class Wrapper(nn.Module):
            def __init__(self, m) -> None:
                super().__init__()
                self.netAE = m.netAE
                self.netDec_B = m.netDec_B

            def forward(self, content, style):
                feat = self.netAE(content, style)
                return self.netDec_B(feat)

        wrapper = Wrapper(model_obj).eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wrapper = wrapper.to(device)
        content = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        style = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        prof = _safe_profile(wrapper, (content, style))
        bench = _benchmark_forward(wrapper, (content, style), warmup, iters)
        return {**prof, **bench}


def _aesfa_profile(run_root: Path, warmup: int, iters: int) -> dict[str, Any]:
    import torch

    repo = REPOS_ROOT / "AesFA"
    _purge_modules(["Config", "model", "blocks", "networks", "vgg19", "DataSplit"])
    with _push_sys_path([repo]), _push_cwd(repo):
        from Config import Config
        from model import AesFA_test
        from blocks import test_model_load

        config = Config()
        config.gpu = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.device = device
        model = AesFA_test(config)
        ckpt = run_root / "checkpoints" / "aesfa" / "main.pth"
        model = test_model_load(str(ckpt), model)
        model = model.to(device).eval()
        content = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        style = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

        class Wrapper(torch.nn.Module):
            def __init__(self, m) -> None:
                super().__init__()
                self.m = m

            def forward(self, content, style):
                out, _ = self.m(content, style, False)
                return out

        wrapper = Wrapper(model).eval()
        prof = _safe_profile(wrapper, (content, style))
        bench = _benchmark_forward(wrapper, (content, style), warmup, iters)
        return {**prof, **bench}


def _aespa_profile(run_root: Path, warmup: int, iters: int) -> dict[str, Any]:
    import torch

    repo = REPOS_ROOT / "AesPA-Net"
    _purge_modules(["baseline", "aespanet_models", "utils", "contextual_utils", "hist_loss", "style_decorator", "data"])
    with _push_sys_path([repo]), _push_cwd(repo):
        from aespanet_models import Baseline_net
        from baseline import Baseline

        baseline = Baseline(
            SimpleNamespace(
                imsize=512,
                batch_size=1,
                cencrop=False,
                cropsize=256,
                num_workers=0,
                content_dir="",
                style_dir="",
                lr=1e-3,
                train_result_dir=str(run_root / "checkpoints" / "aespa"),
                comment="run511",
                max_iter=1,
                check_iter=1,
            )
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = baseline.network
        model.decoder.load_state_dict(torch.load(baseline.result_st_dir + "/dec_model_.pth", map_location="cpu")["state_dict"])
        model.transformer.load_state_dict(torch.load(baseline.result_st_dir + "/transformer_model_.pth", map_location="cpu")["state_dict"])
        model = model.to(device).eval()

        content = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        style = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        gray_content = content.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        adaptive_alpha = torch.full((1, 1), 0.5, device=device)

        class Wrapper(torch.nn.Module):
            def __init__(self, m) -> None:
                super().__init__()
                self.m = m

            def forward(self, content, style, adaptive_alpha, gray_content):
                out, *_ = self.m(content, style, adaptive_alpha, gray_content, style)
                return out

        wrapper = Wrapper(model).eval()
        prof = _safe_profile(wrapper, (content, style, adaptive_alpha, gray_content))
        bench = _benchmark_forward(wrapper, (content, style, adaptive_alpha, gray_content), warmup, iters)
        return {**prof, **bench}


PROFILE_BUILDERS = {
    "stytr2": _styletr_profile,
    "cast": _cast_profile,
    "aesfa": _aesfa_profile,
    "aespa": _aespa_profile,
}


def _preflight_method(method: str) -> dict[str, Any]:
    cfg = METHODS[method]
    launcher = cfg["launcher"]
    if cfg["supports_preflight"]:
        proc = subprocess.run(
            [str(PYTHON_EXE), str(launcher), "--mode", "preflight"],
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
        )
        text = (proc.stdout or proc.stderr or "").strip()
        try:
            payload = json.loads(text)
            payload["returncode"] = proc.returncode
            return payload
        except Exception:
            return {"status": "blocked" if proc.returncode else "ok", "error": text, "returncode": proc.returncode}
    return {"status": "ok", "returncode": 0}


def _profile_method(method: str, run_root: Path, warmup: int, iters: int) -> dict[str, Any]:
    try:
        data = PROFILE_BUILDERS[method](run_root, warmup, iters)
        if data.get("flops") is not None and data.get("throughput_img_per_sec") is not None:
            data["effective_tflops"] = float(data["flops"]) * float(data["throughput_img_per_sec"]) / 1e12
        else:
            data["effective_tflops"] = None
        return {"profile_status": "ok", **data}
    except Exception as exc:
        return {"profile_status": "blocked", "profile_error": f"{type(exc).__name__}: {exc}"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequential review baseline suite: reproduce launchers, record timings, and estimate Params/FLOPs/VRAM."
    )
    parser.add_argument("--methods", nargs="*", default=list(METHODS.keys()), choices=list(METHODS.keys()))
    parser.add_argument("--profile", default="7g")
    parser.add_argument("--mode", choices=["all", "smoke", "measure_only"], default="all")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--profile_warmup", type=int, default=10)
    parser.add_argument("--profile_iters", type=int, default=30)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    manifest = {
        "timestamp": _timestamp(),
        "methods": args.methods,
        "profile": args.profile,
        "mode": args.mode,
        "skip_existing": bool(args.skip_existing),
        "dry_run": bool(args.dry_run),
    }
    _write_json(args.output_root / "manifest.json", manifest)

    for method in args.methods:
        row: dict[str, Any] = {"method": method}
        run_root = args.output_root / method
        preflight = _preflight_method(method)
        row["preflight_status"] = preflight.get("status", "")
        row["preflight_error"] = preflight.get("error", "")

        if row["preflight_status"] == "blocked":
            profile = _profile_method(method, run_root, args.profile_warmup, args.profile_iters) if args.mode == "measure_only" else {"profile_status": "blocked"}
            row.update(profile)
            all_rows.append(row)
            continue

        launcher = METHODS[method]["launcher"]
        summary_exists = (run_root / "summary.json").exists()
        if args.mode != "measure_only" and not (args.skip_existing and summary_exists):
            cmd = [
                str(PYTHON_EXE),
                str(launcher),
                "--mode",
                "all" if args.mode == "all" else "smoke",
                "--profile",
                args.profile,
                "--run_root",
                str(run_root),
            ]
            t0 = time.perf_counter()
            rc = _run(cmd, cwd=WORKSPACE_ROOT, log_path=run_root / "driver.log", dry_run=args.dry_run)
            row["launcher_rc"] = rc
            row["launcher_elapsed_sec"] = round(time.perf_counter() - t0, 3)
        else:
            row["launcher_rc"] = 0
            row["launcher_elapsed_sec"] = 0.0
            row["launcher_note"] = "skipped_existing" if summary_exists else "measure_only"

        all_rows.append(row)

    for row in all_rows:
        method = str(row["method"])
        run_root = args.output_root / method
        summary = _load_summary(run_root)
        row.update(_parse_stage_metrics(summary))
        if row.get("preflight_status") != "blocked":
            profile = _profile_method(method, run_root, args.profile_warmup, args.profile_iters)
            row.update(profile)
        if row.get("generated_images") and row.get("infer_elapsed_sec"):
            try:
                row["end_to_end_img_per_sec"] = float(row["generated_images"]) / float(row["infer_elapsed_sec"])
            except Exception:
                row["end_to_end_img_per_sec"] = None
        _write_json(run_root / "review_record.json", row)

    fieldnames = sorted({key for row in all_rows for key in row.keys()})
    _write_csv(args.output_root / "summary.csv", all_rows, fieldnames)
    _write_json(args.output_root / "summary.json", {"rows": all_rows, "timestamp": _timestamp()})
    print(f"Baseline review suite finished. Summary: {args.output_root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
