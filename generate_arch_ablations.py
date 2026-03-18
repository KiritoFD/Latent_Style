from __future__ import annotations

import copy
import json
import os
from pathlib import Path


def _find_target_dir(repo_root: Path) -> Path:
    candidates = [
        repo_root / 'Cycle-NCE' / 'src',
        repo_root / 'Cycle-NCE' / 'src-decoder',
    ]
    for d in candidates:
        if (d / 'run.py').exists() and (d / 'config_decoder-D-sweetspot.json').exists():
            return d
    raise FileNotFoundError(
        'Cannot find training directory containing both run.py and config_decoder-D-sweetspot.json'
    )


def _load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def _generate_run_script_lines(is_windows: bool, ablations: list[str]) -> list[str]:
    lines: list[str] = []
    if is_windows:
        lines.extend([
            '@echo off\n',
            'setlocal\n\n',
        ])
    else:
        lines.extend([
            '#!/usr/bin/env bash\n',
            'set -euo pipefail\n\n',
        ])

    for name in ablations:
        lines.append(f'echo [Architecture Ablation] Running {name}...\n')

        lines.append(f'uv run run.py --config config_{name}.json\n')

        lines.append('\n')

    return lines


def create_arch_ablations() -> None:
    repo_root = Path.cwd()
    target_dir = _find_target_dir(repo_root)

    base_cfg_path = target_dir / 'config_decoder-D-sweetspot.json'
    base = _load_json(base_cfg_path)

    base.setdefault('training', {})
    base['training']['num_epochs'] = 80
    base['training']['save_interval'] = 40
    base['training']['full_eval_interval'] = 40

    ablations: list[tuple[str, dict]] = [
        # 1) Capacity distortion at extreme scale.
        ('abl_macro_decoder', {'model': {'num_decoder_blocks': 12, 'base_dim': 160, 'ada_mix_rank': 1}}),
        # 2) Kill AdaGN style modulation completely (zero-output branch).
        ('abl_no_adagn', {'model': {'ablation_no_adagn': True, 'ablation_no_adagn_zero_out': True}}),
        # 3) Aggressive skip leakage amplification.
        ('abl_naive_skip', {'model': {'ablation_naive_skip': True, 'ablation_naive_skip_gain': 20.0}}),
        # 4) Remove residual anchor and amplify absolute output.
        ('abl_no_residual', {'model': {'ablation_no_residual': True, 'ablation_no_residual_gain': 20.0}}),
    ]

    for name, json_overrides in ablations:
        cfg = copy.deepcopy(base)
        for section, params in json_overrides.items():
            cfg.setdefault(section, {})
            cfg[section].update(params)

        cfg.setdefault('checkpoint', {})['save_dir'] = f'../{name}'
        _write_json(target_dir / f'config_{name}.json', cfg)

    is_win = os.name == 'nt'
    run_script_name = 'run_arch_ablations.bat' if is_win else 'run_arch_ablations.sh'
    run_script_path = target_dir / run_script_name

    script_lines = _generate_run_script_lines(
        is_windows=is_win,
        ablations=[name for name, _ in ablations],
    )
    run_script_path.write_text(''.join(script_lines), encoding='utf-8')

    if not is_win:
        run_script_path.chmod(0o755)

    print(f'Base config      : {base_cfg_path}')
    print(f'Target directory : {target_dir}')
    print(f'Generated configs:')
    for name, _ in ablations:
        print(f'  - {target_dir / f"config_{name}.json"}')
    print(f'Run script       : {run_script_path}')
    print('Next step        : cd into target directory and run the script above.')


if __name__ == '__main__':
    create_arch_ablations()
