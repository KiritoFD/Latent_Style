import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, TextIO

import yaml


# 固定的扫描列表，用于判别器谱归一化消融实验。
SWEEP_DISCRIMINATOR_SN = [True, False]


def now_str() -> str:
    """返回当前本地时间字符串。"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def format_log_line(message: str) -> str:
    """为日志消息添加统一时间戳前缀。"""
    return f"[{now_str()}] {message}"


def sweep_log(message: str, flush: bool = True) -> None:
    """输出带时间戳的 sweep 日志。"""
    print(format_log_line(message), flush=flush)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="CycleGAN 判别器谱归一化消融实验扫描脚本")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "example.yaml"),
        help="基础配置文件路径，默认读取 configs/example.yaml。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅生成实验目录和配置快照，不实际启动训练进程。",
    )
    return parser.parse_args()


def get_repo_root() -> str:
    """返回脚本所在仓库根目录。"""
    return os.path.dirname(os.path.abspath(__file__))


def get_base_config_path(repo_root: str) -> str:
    """返回固定的基础配置文件路径。"""
    return os.path.join(repo_root, "configs", "example.yaml")


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """读取 YAML 配置，并确保顶层结构为字典。"""
    with open(config_path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj) or {}

    if not isinstance(config, dict):
        raise ValueError(f"配置文件顶层必须是字典: {config_path}")
    return config


def resolve_exp_root(config_path: str, config: Dict[str, Any]) -> str:
    """从配置文件读取 exp_root，并解析为绝对路径。"""
    exp_root = config.get("exp_root")
    if not isinstance(exp_root, str) or not exp_root.strip():
        raise ValueError("配置项 exp_root 缺失或为空，请在 YAML 中提供实验输出根目录")

    config_dir = os.path.dirname(os.path.abspath(config_path)) or os.getcwd()
    if os.path.isabs(exp_root):
        return exp_root
    return os.path.abspath(os.path.join(config_dir, exp_root))


def resolve_config_path_value(config_path: str, path_value: str) -> str:
    """将配置里的路径解析为绝对路径，避免快照迁移后相对路径失效。"""
    if os.path.isabs(path_value):
        return path_value
    config_dir = os.path.dirname(os.path.abspath(config_path)) or os.getcwd()
    return os.path.abspath(os.path.join(config_dir, path_value))


def resolve_model_name_or_path(config_path: str, value: str) -> str:
    """仅在值看起来像本地路径时做展开；Hugging Face repo id 保持原样。"""
    normalized = os.path.expanduser(value.strip())
    if os.path.isabs(normalized):
        return os.path.abspath(normalized)
    if normalized.startswith("."):
        config_dir = os.path.dirname(os.path.abspath(config_path)) or os.getcwd()
        return os.path.abspath(os.path.join(config_dir, normalized))

    candidate = resolve_config_path_value(config_path, normalized)
    if os.path.exists(candidate):
        return candidate
    return value


def dump_yaml_config(config_path: str, config: Dict[str, Any]) -> None:
    """将配置快照写入目标路径。"""
    with open(config_path, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(config, file_obj, allow_unicode=True, sort_keys=False)


def ensure_mapping(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    """确保某个配置段存在且为字典。"""
    value = parent.get(key)
    if value is None:
        parent[key] = {}
        value = parent[key]
    if not isinstance(value, dict):
        raise ValueError(f"配置项 {key} 必须是字典")
    return value


def make_experiment_config(
    base_config: Dict[str, Any],
    base_config_path: str,
    exp_root: str,
    exp_name: str,
    use_discriminator_sn: bool,
) -> Dict[str, Any]:
    """基于基础配置生成单次实验的配置快照。"""
    exp_dir = os.path.join(exp_root, exp_name)

    data_cfg = ensure_mapping(base_config, "data")
    shared_cfg = ensure_mapping(base_config, "shared")
    model_cfg = ensure_mapping(base_config, "model")
    train_cfg = ensure_mapping(base_config, "train")
    logging_cfg = ensure_mapping(base_config, "logging")
    vis_cfg = ensure_mapping(base_config, "visualization")

    base_config["exp_root"] = exp_root
    if isinstance(data_cfg.get("a_dir"), str) and data_cfg["a_dir"]:
        data_cfg["a_dir"] = resolve_config_path_value(base_config_path, data_cfg["a_dir"])
    if isinstance(data_cfg.get("b_dir"), str) and data_cfg["b_dir"]:
        data_cfg["b_dir"] = resolve_config_path_value(base_config_path, data_cfg["b_dir"])
    if isinstance(shared_cfg.get("vae_model_name_or_path"), str) and shared_cfg["vae_model_name_or_path"]:
        shared_cfg["vae_model_name_or_path"] = resolve_model_name_or_path(
            base_config_path, shared_cfg["vae_model_name_or_path"]
        )
    model_cfg["use_discriminator_sn"] = bool(use_discriminator_sn)
    train_cfg["checkpoint_dir"] = os.path.join(exp_dir, "model")
    logging_cfg["log_dir"] = os.path.join(exp_dir, "logs")
    vis_cfg["out_dir"] = os.path.join(exp_dir, "vis")
    return base_config


def stream_reader(stream: TextIO, label: str, log_fp: TextIO, lock: threading.Lock) -> None:
    """实时读取子进程输出，并同时写入终端与日志文件。"""
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            tagged_line = format_log_line(f"[{label}] {line.rstrip()}") + "\n"
            with lock:
                print(tagged_line, end="", flush=True)
                log_fp.write(tagged_line)
                log_fp.flush()
    finally:
        stream.close()


def run_experiment(repo_root: str, exp_dir: str, config_path: str, log_file: str) -> int:
    """执行单次训练实验，并实时双工转发 stdout/stderr。"""
    train_script = os.path.join(repo_root, "train_latent_cyclegan.py")
    command = [sys.executable, train_script, "--config", config_path]

    with open(log_file, "a", encoding="utf-8") as log_fp:
        log_fp.write(format_log_line("=" * 100) + "\n")
        log_fp.write(format_log_line(f"Experiment directory: {exp_dir}") + "\n")
        log_fp.write(format_log_line(f"Command: {' '.join(command)}") + "\n")
        log_fp.flush()

        process = subprocess.Popen(
            command,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        io_lock = threading.Lock()
        threads = [
            threading.Thread(target=stream_reader, args=(process.stdout, "stdout", log_fp, io_lock), daemon=True),
            threading.Thread(target=stream_reader, args=(process.stderr, "stderr", log_fp, io_lock), daemon=True),
        ]

        for thread in threads:
            thread.start()

        return_code = process.wait()

        for thread in threads:
            thread.join()

        return return_code


def write_summary_csv(summary_path: str, results: List[Dict[str, Any]]) -> None:
    """将当前 sweep 汇总结果写入 CSV，便于后续统计与追踪。"""
    fieldnames = [
        "experiment_name",
        "use_discriminator_sn",
        "status",
        "return_code",
        "duration_sec",
        "exp_dir",
        "config_path",
        "log_file",
        "message",
    ]
    with open(summary_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in fieldnames})


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """以终端表格形式打印当前 sweep 汇总。"""
    if not results:
        return

    headers = ["experiment_name", "status", "return_code", "duration_sec", "exp_dir"]
    rows: List[List[str]] = []
    for result in results:
        rows.append(
            [
                str(result.get("experiment_name", "")),
                str(result.get("status", "")),
                str(result.get("return_code", "")),
                f"{float(result.get('duration_sec', 0.0)):.2f}",
                str(result.get("exp_dir", "")),
            ]
        )

    col_widths: List[int] = []
    for idx, header in enumerate(headers):
        width = len(header)
        for row in rows:
            width = max(width, len(row[idx]))
        col_widths.append(width)

    def _format_row(values: List[str]) -> str:
        return " | ".join(value.ljust(col_widths[idx]) for idx, value in enumerate(values))

    separator = "-+-".join("-" * width for width in col_widths)
    sweep_log("Current sweep summary:")
    print(_format_row(headers))
    print(separator)
    for row in rows:
        print(_format_row(row))


def main() -> None:
    """按预定义 SN 开关列表循环执行消融实验。"""
    args = parse_args()
    repo_root = get_repo_root()
    base_config_path = os.path.abspath(args.config if args.config else get_base_config_path(repo_root))

    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"基础配置文件不存在: {base_config_path}")

    base_config = load_yaml_config(base_config_path)
    exp_root = resolve_exp_root(base_config_path, base_config)
    os.makedirs(exp_root, exist_ok=True)

    summary_path = os.path.join(exp_root, "sweep_summary.csv")
    results: List[Dict[str, Any]] = []

    sweep_log(f"Using Python executable: {sys.executable}")
    sweep_log(f"Base config: {base_config_path}")
    sweep_log(f"Experiment root: {exp_root}")
    sweep_log(f"Dry-run mode: {args.dry_run}")

    for use_discriminator_sn in SWEEP_DISCRIMINATOR_SN:
        exp_name = "discriminator_sn_on" if use_discriminator_sn else "discriminator_sn_off"
        exp_dir = os.path.join(exp_root, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        print()
        sweep_log("=" * 100)
        sweep_log(
            f"Starting sweep run for {exp_name} (use_discriminator_sn={use_discriminator_sn})"
        )
        sweep_log(f"Experiment directory: {os.path.abspath(exp_dir)}")
        sweep_log("=" * 100)

        start_time = time.time()
        snapshot_path = os.path.join(exp_dir, "config.yaml")
        log_file = os.path.join(exp_dir, "stdout_stderr.log")

        try:
            config = load_yaml_config(base_config_path)
            run_config = make_experiment_config(
                config,
                base_config_path=base_config_path,
                exp_root=exp_root,
                exp_name=exp_name,
                use_discriminator_sn=use_discriminator_sn,
            )
            dump_yaml_config(snapshot_path, run_config)

            if args.dry_run:
                return_code = 0
                status = "DRY-RUN"
                message = "Config snapshot created. Training skipped by --dry-run."
                sweep_log(f"[DRY-RUN] Generated config only: {snapshot_path}")
            else:
                return_code = run_experiment(
                    repo_root=repo_root,
                    exp_dir=os.path.abspath(exp_dir),
                    config_path=snapshot_path,
                    log_file=log_file,
                )
                if return_code != 0:
                    status = "FAILED"
                    message = f"Sweep run failed with return code {return_code}."
                    sweep_log(
                        f"\033[91m[ERROR] Sweep run failed for {exp_name}, return code={return_code}\033[0m",
                    )
                else:
                    status = "OK"
                    message = "Sweep run finished successfully."
                    sweep_log(f"[OK] Sweep run finished for {exp_name}")

            duration_sec = time.time() - start_time
            results.append(
                {
                    "experiment_name": exp_name,
                    "use_discriminator_sn": use_discriminator_sn,
                    "status": status,
                    "return_code": return_code,
                    "duration_sec": round(duration_sec, 2),
                    "exp_dir": os.path.abspath(exp_dir),
                    "config_path": snapshot_path,
                    "log_file": log_file,
                    "message": message,
                }
            )
            write_summary_csv(summary_path, results)
            print_summary_table(results)
        except Exception as exc:
            sweep_log(f"\033[91m[ERROR] Sweep run crashed for {exp_name}: {exc}\033[0m")
            duration_sec = time.time() - start_time
            results.append(
                {
                    "experiment_name": exp_name,
                    "use_discriminator_sn": use_discriminator_sn,
                    "status": "CRASHED",
                    "return_code": -1,
                    "duration_sec": round(duration_sec, 2),
                    "exp_dir": os.path.abspath(exp_dir),
                    "config_path": snapshot_path,
                    "log_file": log_file,
                    "message": str(exc),
                }
            )
            write_summary_csv(summary_path, results)
            print_summary_table(results)
            continue

    print()
    sweep_log("=" * 100)
    sweep_log(f"Sweep finished. Summary saved to: {summary_path}")
    print_summary_table(results)
    sweep_log("=" * 100)


if __name__ == "__main__":
    main()
