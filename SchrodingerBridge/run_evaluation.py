from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    workspace = root.parent
    local_clip_dir = workspace / "Cycle-NCE" / "eval_cache" / "manual_clip" / "openai-clip-vit-base-patch32"
    default_test_dir = workspace / "style_data" / "overfit50"
    default_cache_dir = workspace / "Cycle-NCE" / "eval_cache"
    default_clip_hf_cache_dir = default_cache_dir / "hf"

    argv = list(sys.argv[1:])
    has_clip_model_name = any(arg == "--clip_model_name" or arg.startswith("--clip_model_name=") for arg in argv)
    has_clip_backend = any(arg == "--clip_backend" or arg.startswith("--clip_backend=") for arg in argv)
    has_checkpoint_flag = any(arg == "--checkpoint" or arg.startswith("--checkpoint=") for arg in argv)
    has_output_flag = any(arg == "--output" or arg.startswith("--output=") for arg in argv)
    has_test_dir = any(arg == "--test_dir" or arg.startswith("--test_dir=") for arg in argv)
    has_cache_dir = any(arg == "--cache_dir" or arg.startswith("--cache_dir=") for arg in argv)
    has_clip_hf_cache_dir = any(arg == "--clip_hf_cache_dir" or arg.startswith("--clip_hf_cache_dir=") for arg in argv)

    positional = [arg for arg in argv if arg and not arg.startswith("-")]
    if positional and not has_checkpoint_flag:
        first = Path(positional[0])
        if first.suffix.lower() == ".pt":
            ckpt_path = str(first)
            default_output = str(root / "artifacts" / "full_eval" / first.stem)
            argv = ["--checkpoint", ckpt_path, "--output", default_output, *argv[1:]]
            has_output_flag = True

    if has_checkpoint_flag and not has_output_flag:
        try:
            idx = argv.index("--checkpoint")
            ckpt_path = Path(argv[idx + 1])
            argv.extend(["--output", str(root / "artifacts" / "full_eval" / ckpt_path.stem)])
        except Exception:
            pass

    if not has_clip_backend:
        argv.extend(["--clip_backend", "hf"])
    if not has_clip_model_name and local_clip_dir.exists():
        argv.extend(["--clip_model_name", str(local_clip_dir)])
    if not has_test_dir and default_test_dir.exists():
        argv.extend(["--test_dir", str(default_test_dir)])
    if not has_cache_dir and default_cache_dir.exists():
        argv.extend(["--cache_dir", str(default_cache_dir)])
    if not has_clip_hf_cache_dir and default_clip_hf_cache_dir.exists():
        argv.extend(["--clip_hf_cache_dir", str(default_clip_hf_cache_dir)])

    sys.argv = [sys.argv[0], *argv]
    runpy.run_module("schrodinger_bridge.utils.run_evaluation", run_name="__main__")


if __name__ == "__main__":
    main()
