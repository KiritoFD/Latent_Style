import argparse
import json
import random
import time
from pathlib import Path


def _repo_root() -> Path:
    # .../Latent_Style/Thermal/src/utils -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def _resolve_path(path_str: str, base: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _to_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def _sample_without_replacement(rng: random.Random, items: list[Path], count: int) -> list[Path]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    return rng.sample(items, count)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build fixed SWD evaluation manifest from latent files')
    parser.add_argument('--latent_root', type=str, default='style_data/latents/test')
    parser.add_argument('--styles', type=str, default='photo,monet,vangogh,cezanne')
    parser.add_argument('--style_pool_size', type=int, default=100)
    parser.add_argument('--control_style', type=str, default='photo')
    parser.add_argument('--control_count', type=int, default=100)
    parser.add_argument('--identity_count', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, required=True, help='Output manifest.json path')
    args = parser.parse_args()

    repo_root = _repo_root()
    latent_root = _resolve_path(args.latent_root, repo_root)
    output_path = _resolve_path(args.output, Path.cwd())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = [s.strip() for s in args.styles.split(',') if s.strip()]
    if not styles:
        raise ValueError('No styles provided')
    if args.control_style not in styles:
        raise ValueError(f'control_style={args.control_style} is not in styles={styles}')

    rng = random.Random(args.seed)

    style_pool_abs: dict[str, list[Path]] = {}
    style_pool_rel: dict[str, list[str]] = {}

    for style in styles:
        style_dir = latent_root / style
        if not style_dir.exists():
            raise FileNotFoundError(f'Style latent directory not found: {style_dir}')
        files = sorted(style_dir.glob('*.pt'))
        if not files:
            raise RuntimeError(f'No latent .pt files found for style={style} in {style_dir}')

        if args.style_pool_size > 0:
            chosen = _sample_without_replacement(rng, files, min(args.style_pool_size, len(files)))
        else:
            chosen = files

        style_pool_abs[style] = chosen
        style_pool_rel[style] = [_to_repo_relative(p, repo_root) for p in chosen]

    control_candidates = style_pool_abs[args.control_style]
    control_selected = _sample_without_replacement(
        rng,
        control_candidates,
        min(args.control_count, len(control_candidates)),
    )

    control_subset = [
        {
            'source_style': args.control_style,
            'latent_path': _to_repo_relative(path, repo_root),
        }
        for path in control_selected
    ]

    num_styles = len(styles)
    base = args.identity_count // num_styles
    remainder = args.identity_count % num_styles

    identity_subset = []
    for idx, style in enumerate(styles):
        target_count = base + (1 if idx < remainder else 0)
        source = style_pool_abs[style]
        if target_count <= len(source):
            selected = _sample_without_replacement(rng, source, target_count)
        else:
            selected = list(source)
            while len(selected) < target_count:
                selected.append(rng.choice(source))

        for path in selected:
            identity_subset.append(
                {
                    'source_style': style,
                    'target_style': style,
                    'latent_path': _to_repo_relative(path, repo_root),
                }
            )

    style_to_id = {style: idx for idx, style in enumerate(styles)}

    manifest = {
        'version': 1,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seed': args.seed,
        'repo_root': repo_root.resolve().as_posix(),
        'latent_root': latent_root.resolve().as_posix(),
        'styles': styles,
        'style_to_id': style_to_id,
        'style_pool': style_pool_rel,
        'control_subset': control_subset,
        'identity_subset': identity_subset,
        'counts': {
            'style_pool': {style: len(items) for style, items in style_pool_rel.items()},
            'control_subset': len(control_subset),
            'identity_subset': len(identity_subset),
        },
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f'Wrote SWD eval manifest: {output_path}')
    print(json.dumps(manifest['counts'], indent=2))


if __name__ == '__main__':
    main()
