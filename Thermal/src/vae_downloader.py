"""
VAE Model Downloader Utility

Provides convenient functions to download and manage VAE models from Hugging Face
with progress tracking and caching support.

Usage:
    python vae_downloader.py --model sd15 --device cuda
    python vae_downloader.py --list
    python vae_downloader.py --model stabilityai/sdxl-vae --cache-dir ~/.cache/custom
"""

import argparse
import logging
import os
from pathlib import Path
import torch
from typing import Optional, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Available VAE models
VAE_MODELS: Dict[str, str] = {
    'sd15': 'stabilityai/sd-vae-ft-mse',
    'sd15-ema': 'stabilityai/sd-vae-ft-ema',
    'sdxl': 'stabilityai/sdxl-vae',
    'mse': 'stabilityai/sd-vae-ft-mse',
    'ema': 'stabilityai/sd-vae-ft-ema',
}

# Optional ModelScope support
try:
    from modelscope.hub import snapshot_download as ms_snapshot_download  # type: ignore
    MODELSCOPE_AVAILABLE = True
except Exception:
    try:
        # Some installations expose a module rather than a direct function; keep a reference to it
        import modelscope.hub as ms_hub  # type: ignore
        # try to grab the function; otherwise keep the module object for introspection
        ms_snapshot_download = getattr(ms_hub, 'snapshot_download', ms_hub)
        MODELSCOPE_AVAILABLE = True
    except Exception:
        ms_snapshot_download = None
        MODELSCOPE_AVAILABLE = False


def get_cache_dir(custom_cache: Optional[str] = None) -> str:
    """
    Get cache directory for HuggingFace models.
    
    Args:
        custom_cache: Custom cache directory path
    
    Returns:
        Cache directory path
    """
    if custom_cache:
        return custom_cache
    
    cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
    return cache_dir


def download_vae_model(
    model_id: str,
    device: str = 'cuda',
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    prefer_modelscope: bool = False,
) -> bool:
    """
    Download VAE model from Hugging Face with optional ModelScope fallback.
    
    Behaviour:
    - If not force_download and model cached locally, return success immediately
    - Otherwise try Hugging Face first (unless prefer_modelscope=True), then try ModelScope if available

    Args:
        model_id: Model identifier or preset name
        device: Target device ('cuda' or 'cpu')
        cache_dir: Custom cache directory
        force_download: Force re-download even if cached
        prefer_modelscope: Try ModelScope before Hugging Face
    
    Returns:
        True if download successful, False otherwise
    """
    from diffusers import AutoencoderKL
    
    # Resolve model ID
    if model_id in VAE_MODELS:
        resolved_id = VAE_MODELS[model_id]
        logger.info(f"Using preset: {model_id} -> {resolved_id}")
    else:
        resolved_id = model_id
    
    # Setup cache
    cache_dir = get_cache_dir(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    logger.info(f"Fetching VAE: {resolved_id}")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Target device: {device}")

    def _is_cached():
        # Try to load model from local cache only
        try:
            AutoencoderKL.from_pretrained(resolved_id, cache_dir=cache_dir, local_files_only=True)
            return True
        except Exception:
            return False

    if not force_download and _is_cached():
        logger.info(f"✓ Model is already cached: {resolved_id}")
        return True

    # Try sources in preferred order
    sources = ['modelscope', 'huggingface'] if prefer_modelscope else ['huggingface', 'modelscope']

    for src in sources:
        if src == 'huggingface':
            try:
                vae = AutoencoderKL.from_pretrained(
                    resolved_id,
                    torch_dtype=torch.float16,
                    cache_dir=cache_dir,
                    force_download=force_download,
                )
                logger.info(f"✓ Successfully downloaded from HuggingFace: {resolved_id}")
                logger.info(f"  Model config: {vae.config}")
                return True
            except Exception as e:
                logger.warning(f"HuggingFace download failed for {resolved_id}: {e}")
                continue

        if src == 'modelscope' and MODELSCOPE_AVAILABLE:
            try:
                # Download into a subfolder to avoid clobbering HF cache layout
                dest = os.path.join(cache_dir, 'modelscope', resolved_id.replace('/', '_'))
                os.makedirs(dest, exist_ok=True)
                logger.info(f"Attempting ModelScope snapshot download for: {resolved_id}")

                ret = _call_modelscope_snapshot(resolved_id, dest)

                # Resolve actual model root: prefer returned path if valid, otherwise search dest
                download_root = None
                if isinstance(ret, str) and os.path.exists(ret):
                    download_root = ret
                    logger.debug(f"ModelScope returned path: {ret}")
                else:
                    found = _find_hf_repo_root(dest)
                    if found:
                        download_root = found
                        logger.debug(f"ModelScope placed files under nested path: {found}")
                    else:
                        nested_guess = os.path.join(dest, *resolved_id.split('/'))
                        if os.path.exists(nested_guess):
                            download_root = nested_guess
                            logger.debug(f"Using nested guess path: {nested_guess}")

                if download_root is None:
                    raise RuntimeError(f"No HF-style model files found under {dest} after ModelScope download")

                # Attempt to load from resolved path
                vae = AutoencoderKL.from_pretrained(download_root, torch_dtype=torch.float16).to(device)
                logger.info(f"✓ Successfully downloaded from ModelScope: {resolved_id} (root={download_root})")
                return True
            except Exception as e:
                logger.warning(f"ModelScope download failed for {resolved_id}: {e}")
                continue

    logger.error(f"✗ All download attempts failed for: {resolved_id}")
    return False


def list_available_models() -> None:
    """Print list of available VAE models."""
    logger.info("Available VAE Model Presets:")
    logger.info("-" * 50)
    for name, hf_id in VAE_MODELS.items():
        logger.info(f"  {name:15} -> {hf_id}")
    logger.info("-" * 50)
    logger.info("You can also use any Hugging Face model ID directly.")


def verify_vae_download(
    model_id: str,
    cache_dir: Optional[str] = None
) -> bool:
    """
    Verify if VAE model is cached locally.
    
    Args:
        model_id: Model identifier or preset name
        cache_dir: Custom cache directory
    
    Returns:
        True if model is cached, False otherwise
    """
    from diffusers import AutoencoderKL

    # Resolve model ID
    if model_id in VAE_MODELS:
        resolved_id = VAE_MODELS[model_id]
    else:
        resolved_id = model_id

    # Check cache
    cache_dir = get_cache_dir(cache_dir)

    try:
        # Try to load model from cache only
        AutoencoderKL.from_pretrained(resolved_id, cache_dir=cache_dir, local_files_only=True)
        logger.info(f"✓ Model is cached: {resolved_id}")
        return True
    except Exception:
        logger.info(f"✗ Model not in cache: {resolved_id}")
        return False


def clear_cache(cache_dir: Optional[str] = None) -> None:
    """
    Clear HuggingFace cache directory.
    
    Args:
        cache_dir: Custom cache directory
    """
    cache_dir = get_cache_dir(cache_dir)
    
    if not os.path.exists(cache_dir):
        logger.info("Cache directory does not exist.")
        return
    
    logger.warning(f"Clearing cache: {cache_dir}")
    import shutil
    try:
        shutil.rmtree(cache_dir)
        logger.info("✓ Cache cleared successfully")
    except Exception as e:
        logger.error(f"✗ Failed to clear cache: {e}")


def _call_modelscope_snapshot(repo_id: str, dest: str):
    """
    Normalize calling ModelScope snapshot_download across different versions/shapes.

    Tries several common invocation patterns and falls back to looking up a nested
    'snapshot_download' (or 'download') attribute if the imported object is a module.
    Raises an exception if no callable target is found or all attempts fail.
    """
    if not MODELSCOPE_AVAILABLE or ms_snapshot_download is None:
        raise RuntimeError("ModelScope snapshot downloader not available")

    logger.debug(f"ModelScope snapshot object: type={type(ms_snapshot_download)}, repr={repr(ms_snapshot_download)}")

    # If the imported object is directly callable, try several common call signatures
    if callable(ms_snapshot_download):
        last_exc = None
        for attempt in (
            lambda: ms_snapshot_download(repo_id, cache_dir=dest),
            lambda: ms_snapshot_download(repo_id, dest),
            lambda: ms_snapshot_download(repo_id=repo_id, cache_dir=dest),
            lambda: ms_snapshot_download(repo_id=repo_id, cache_dir=dest, progress=False),
        ):
            try:
                return attempt()
            except TypeError as e:
                last_exc = e
                continue
        # If none matched, raise the last TypeError to preserve context
        raise last_exc or RuntimeError("Callable ms_snapshot_download failed without TypeError")
    else:
        # The imported object is likely a module; look for nested functions
        func = getattr(ms_snapshot_download, 'snapshot_download', None) or getattr(ms_snapshot_download, 'download', None)
        if callable(func):
            last_exc = None
            for attempt in (
                lambda: func(repo_id, cache_dir=dest),
                lambda: func(repo_id, dest),
                lambda: func(repo_id=repo_id, cache_dir=dest),
            ):
                try:
                    return attempt()
                except TypeError as e:
                    last_exc = e
                    continue
            raise last_exc or RuntimeError("Nested snapshot_download found but failed to call")
        # Try importing hub module directly as a last resort
        try:
            import modelscope.hub as ms_hub  # type: ignore
            func = getattr(ms_hub, 'snapshot_download', None)
            if callable(func):
                return func(repo_id, cache_dir=dest)
        except Exception:
            pass

        raise RuntimeError("No callable snapshot_download available in ModelScope")


def _find_hf_repo_root(dest: str) -> Optional[str]:
    """
    Search `dest` recursively for a directory that looks like a Hugging Face repo root.
    """
    if not os.path.exists(dest):
        return None

    for root, dirs, files in os.walk(dest):
        if 'config.json' in files or 'model_index.json' in files or 'pytorch_model.bin' in files:
            logger.debug(f"Found HF-style repo root: {root}")
            return root
    return None


def main():
    parser = argparse.ArgumentParser(
        description='VAE Model Downloader Utility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vae_downloader.py --model sd15
  python vae_downloader.py --model sdxl --device cuda
  python vae_downloader.py --list
  python vae_downloader.py --verify sd15
  python vae_downloader.py --model stabilityai/sd-vae-ft-mse
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='VAE model to download (preset name or HF model ID)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Target device (default: cuda)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Custom cache directory'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available model presets'
    )
    parser.add_argument(
        '--verify',
        type=str,
        help='Verify if a model is cached'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download (ignore cache)'
    )
    parser.add_argument(
        '--prefer-huggingface-first',
        action='store_true',
        help='Try HuggingFace before ModelScope when downloading (overrides default ModelScope-first behavior)'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear HuggingFace cache'
    )
    
    args = parser.parse_args()
    
    # List models
    if args.list:
        list_available_models()
        return
    
    # Verify cached model
    if args.verify:
        verify_vae_download(args.verify, args.cache_dir)
        return
    
    # Clear cache
    if args.clear_cache:
        clear_cache(args.cache_dir)
        return
    
    # Download model
    if args.model:
        # Default: prefer ModelScope first unless user explicitly requests HuggingFace-first
        prefer_modelscope = not args.prefer_huggingface_first
        success = download_vae_model(
            args.model,
            device=args.device,
            cache_dir=args.cache_dir,
            force_download=args.force,
            prefer_modelscope=prefer_modelscope,
        )
        exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
