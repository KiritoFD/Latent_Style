from __future__ import annotations

import hashlib
import json
import logging
import re
import struct
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared-memory helpers for the pairing cache
# ---------------------------------------------------------------------------
try:
    import multiprocessing.shared_memory as _shm_mod
    _SHM_AVAILABLE = True
except ImportError:  # Python < 3.8 fallback (shouldn't happen)
    _SHM_AVAILABLE = False


def _shm_name_for_path(prefix: str, path: Path) -> str:
    """Return a stable, short POSIX SHM name derived from *path* and its mtime.

    The mtime is incorporated so the cache auto-invalidates when the source
    file changes.  Names are kept to <= 30 chars (prefix + 16-hex digest)
    to stay well inside the Linux POSIX limit of 255 chars.
    """
    try:
        mtime_ns = int(path.stat().st_mtime_ns)
    except OSError:
        mtime_ns = 0
    raw = f"{path}:{mtime_ns}".encode()
    digest = hashlib.md5(raw).hexdigest()[:16]
    safe_prefix = re.sub(r"[^a-zA-Z0-9]", "", prefix)[:6]
    return f"ls_{safe_prefix}_{digest}"


def _shm_read_json(shm_name: str) -> object | None:
    """Attach to an existing named SHM block and decode its JSON payload.

    Layout: 8-byte little-endian uint64 data-length, then UTF-8 JSON bytes.
    Returns *None* on any error (block not found, corrupt data, etc.).
    """
    if not _SHM_AVAILABLE:
        return None
    shm = None
    try:
        shm = _shm_mod.SharedMemory(name=shm_name, create=False)
        if shm.size < 8:
            return None
        data_len = struct.unpack_from("<Q", shm.buf, 0)[0]
        if data_len == 0 or data_len > shm.size - 8:
            return None
        raw = bytes(shm.buf[8 : 8 + data_len])
        return json.loads(raw)
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.debug("SHM read failed for %s: %s", shm_name, exc)
        return None
    finally:
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass


def _shm_write_json(shm_name: str, obj: object) -> bool:
    """Serialise *obj* as JSON and store it in a named SHM block.

    Creates or reuses a block.  Silently returns *False* on any failure so
    callers can treat SHM as a best-effort write-through cache.
    """
    if not _SHM_AVAILABLE:
        return False
    shm = None
    try:
        raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        total = 8 + len(raw)
        # Try to reuse an existing block of sufficient size.
        try:
            shm = _shm_mod.SharedMemory(name=shm_name, create=False)
            if shm.size < total:
                shm.close()
                shm.unlink()
                shm = None
        except FileNotFoundError:
            shm = None
        except Exception:
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass
            shm = None
        if shm is None:
            shm = _shm_mod.SharedMemory(name=shm_name, create=True, size=max(total, 4096))
        struct.pack_into("<Q", shm.buf, 0, len(raw))
        shm.buf[8 : 8 + len(raw)] = raw
        return True
    except Exception as exc:
        logger.debug("SHM write failed for %s: %s", shm_name, exc)
        return False
    finally:
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass


def _normalize_base_stem_text(stem: str) -> str:
    text = str(stem or "").strip()
    return text[:-5] if text.endswith("_flip") else text


def _stem_aliases(style: str, stem: str) -> tuple[str, ...]:
    style_name = str(style or "").strip()
    base_stem = _normalize_base_stem_text(stem)
    if not style_name or not base_stem:
        return tuple()
    aliases = [base_stem]
    prefix = f"{style_name}__"
    if base_stem.startswith(prefix):
        aliases.append(base_stem[len(prefix):])
    else:
        aliases.append(prefix + base_stem)
    return tuple(dict.fromkeys(x for x in aliases if x))


def _load_latent_file(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".pt":
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            obj = obj.get("latent", obj)
        latent = torch.as_tensor(obj).float()
    elif path.suffix.lower() == ".npy":
        latent = torch.from_numpy(np.load(path)).float()
    else:
        raise ValueError(f"Unsupported latent format: {path}")

    if latent.ndim == 4 and latent.shape[0] == 1:
        latent = latent.squeeze(0)
    if latent.ndim != 3:
        raise ValueError(f"Expected latent shape [C,H,W], got {tuple(latent.shape)} from {path}")
    return latent


class AdaCUTLatentDataset(Dataset):
    """
    Unpaired latent dataset with uniform target-style sampling:
    - content style sampled from all styles
    - target style sampled uniformly from all styles (including self)
      so identity probability is naturally 1 / num_styles.
    """

    def __init__(
        self,
        data_root: str,
        style_subdirs: Sequence[str],
        allow_hflip: bool = True,
        identity_ratio: float | None = None,
        batch_size_hint: int = 0,
        balance_target_styles_per_batch: bool = False,
        preload_to_gpu: bool = False,
        preload_max_vram_gb: float = 0.0,
        preload_reserve_ratio: float = 0.35,
        virtual_length_multiplier: float = 1.0,
        content_style_sampling_weights: Sequence[float] | None = None,
        target_style_sampling_weights: Sequence[float] | None = None,
        pairing_cache_path: str = "",
        pairing_cache_topk: int = 4,
        pairing_cache_active_topk: int = 0,
        pairing_cache_sample_mode: str = "uniform_topk",
        pairing_cache_rank_schedule: str = "fixed",
        pairing_cache_min_topk: int = 1,
        pairing_cache_curriculum_epochs: int = 0,
        pairing_cache_rank_power: float = 1.0,
        pairing_cache_explore_prob: float = 0.0,
        pairing_cache_explore_topk: int = 0,
        pairing_cache_dual_target_mix: float = 0.0,
        pairing_cache_dual_target_topk: int = 0,
        pairing_cache_aux_target_topk: int = 0,
        pairing_cache_cross_only: bool = True,
        latent_cache_mode: str = "off",
        latent_cache_dir: str = "",
        style_caption_path: str = "",
        device: str = "cpu",
    ) -> None:
        self.data_root = Path(data_root)
        self.style_subdirs = list(style_subdirs)
        self.allow_hflip = bool(allow_hflip)
        self.identity_ratio = None if identity_ratio is None else float(max(0.0, min(1.0, identity_ratio)))
        self.batch_size_hint = max(0, int(batch_size_hint))
        self.balance_target_styles_per_batch = bool(balance_target_styles_per_batch)
        requested_preload_to_gpu = bool(preload_to_gpu)
        self.preload_max_vram_gb = max(0.0, float(preload_max_vram_gb))
        self.preload_reserve_ratio = max(0.0, min(0.95, float(preload_reserve_ratio)))
        self.preload_to_gpu = False
        self.device = device
        self.epoch = 0
        self.style_item_stems: Dict[int, list[str]] = {}
        self.style_base_to_indices: Dict[int, dict[str, list[int]]] = {}
        self.offline_pairing_map: dict[tuple[str, str, str], list[str]] = {}
        self.style_caption_path = str(style_caption_path or "").strip()
        self.style_captions: dict[str, str] = {}
        if self.style_caption_path and self.style_caption_path.endswith(".jsonl"):
            self._load_style_captions(self.style_caption_path)
        
        # Cache for pre-computed indices to remove CPU overhead in __getitem__
        self._cache_content_style_ids = None
        self._cache_content_rands = None
        self._cache_target_style_ids = None
        self._cache_target_rands = None
        self._cache_pairing_rands = None
        self._cache_pairing_explore_rands = None
        self._cache_pairing_dual_rands = None
        self._cache_pairing_aux_rands = None
        self._cache_flip_content = None
        self._cache_flip_target = None

        if not self.style_subdirs:
            raise ValueError("style_subdirs cannot be empty")
        if len(self.style_subdirs) < 2:
            raise ValueError("At least two style domains are required for cross-domain sampling")

        self.pairing_cache_path = str(pairing_cache_path or "").strip()
        self.pairing_cache_topk = max(1, int(pairing_cache_topk))
        self.pairing_cache_active_topk = max(0, int(pairing_cache_active_topk))
        self.pairing_cache_sample_mode = str(pairing_cache_sample_mode).strip().lower() or "uniform_topk"
        self.pairing_cache_rank_schedule = str(pairing_cache_rank_schedule).strip().lower() or "fixed"
        if self.pairing_cache_rank_schedule not in {"fixed", "easy_to_hard", "hard_to_easy"}:
            logger.warning("Unknown pairing_cache_rank_schedule=%s; using fixed.", self.pairing_cache_rank_schedule)
            self.pairing_cache_rank_schedule = "fixed"
        self.pairing_cache_min_topk = max(1, int(pairing_cache_min_topk))
        self.pairing_cache_curriculum_epochs = max(0, int(pairing_cache_curriculum_epochs))
        self.pairing_cache_rank_power = max(1e-3, float(pairing_cache_rank_power))
        self.pairing_cache_explore_prob = max(0.0, min(1.0, float(pairing_cache_explore_prob)))
        self.pairing_cache_explore_topk = max(0, int(pairing_cache_explore_topk))
        self.pairing_cache_dual_target_mix = max(0.0, min(1.0, float(pairing_cache_dual_target_mix)))
        self.pairing_cache_dual_target_topk = max(0, int(pairing_cache_dual_target_topk))
        self.pairing_cache_aux_target_topk = max(0, int(pairing_cache_aux_target_topk))
        self.pairing_cache_cross_only = bool(pairing_cache_cross_only)
        self.latent_cache_mode = str(latent_cache_mode or "off").strip().lower()
        if self.latent_cache_mode not in {"off", "manifest", "packed", "refresh"}:
            logger.warning("Unknown latent_cache_mode=%s; using off.", self.latent_cache_mode)
            self.latent_cache_mode = "off"
        self.latent_cache_dir = Path(latent_cache_dir) if str(latent_cache_dir or "").strip() else (self.data_root / ".latent_cache")
        if not self.latent_cache_dir.is_absolute():
            self.latent_cache_dir = (self.data_root / self.latent_cache_dir).resolve()
        self._latent_manifest: dict[str, list[Path]] | None = None
        if self.latent_cache_mode != "off":
            self._latent_manifest = self._load_or_build_manifest(force_refresh=self.latent_cache_mode == "refresh")

        self.style_tensors: Dict[int, torch.Tensor] = {}
        logger.info("Loading latent dataset from %s", self.data_root)
        for style_id, subdir in enumerate(self.style_subdirs):
            files = self._resolve_style_files(subdir)
            if not files:
                raise RuntimeError(f"No latent files found for style={subdir} under {self.data_root}")
            stack = self._load_style_stack(style_id, subdir, files)
            self.style_tensors[style_id] = stack
            self._register_style_index(style_id, files)
            logger.info("  style=%s id=%d count=%d", subdir, style_id, stack.shape[0])

        total_count = sum(int(t.shape[0]) for t in self.style_tensors.values())
        self.content_count = max(1, total_count)
        effective_multiplier = max(1e-6, float(virtual_length_multiplier))
        self.virtual_length_multiplier = effective_multiplier
        self.length = max(1, int(round(self.content_count * effective_multiplier)))
        self.content_style_sampling_weights = self._normalize_style_weights(content_style_sampling_weights, "content_style_sampling_weights")
        self.target_style_sampling_weights = self._normalize_style_weights(target_style_sampling_weights, "target_style_sampling_weights")

        if requested_preload_to_gpu:
            self._try_preload_to_gpu()
        if self.pairing_cache_path:
            self._load_pairing_cache(self.pairing_cache_path)

        # Initialize deterministic caches so __getitem__ is always safe.
        self.set_epoch(0)

    def _manifest_path(self) -> Path:
        return self.latent_cache_dir / "manifest.json"

    def _load_style_captions(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            logger.warning("Style caption file not found: %s", p)
            return
        count = 0
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rel = entry.get("rel_path", "").strip()
                cap = entry.get("caption", "").strip()
                if rel and cap and entry.get("status", "success") == "success":
                    self.style_captions[rel] = cap
                    count += 1
        logger.info("Loaded %d style captions from %s", count, p)

    def _style_cache_name(self, style_id: int, subdir: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(subdir)).strip("_") or f"style_{style_id}"
        return f"{style_id:02d}_{safe}.pt"

    def _scan_style_files(self, subdir: str) -> list[Path]:
        style_dir = self.data_root / subdir
        return sorted(style_dir.glob("*.pt")) + sorted(style_dir.glob("*.npy"))

    def _load_or_build_manifest(self, *, force_refresh: bool = False) -> dict[str, list[Path]]:
        manifest_path = self._manifest_path()
        if not force_refresh and manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    payload.get("schema") == 1
                    and payload.get("data_root") == str(self.data_root)
                    and payload.get("style_subdirs") == self.style_subdirs
                ):
                    styles = payload.get("styles", {})
                    manifest: dict[str, list[Path]] = {}
                    for subdir in self.style_subdirs:
                        rel_files = styles.get(subdir, {}).get("files", [])
                        manifest[subdir] = [self.data_root / str(rel) for rel in rel_files]
                    logger.info("Loaded latent manifest cache: %s", manifest_path)
                    return manifest
            except Exception as exc:
                logger.warning("Ignoring stale latent manifest cache %s (%s).", manifest_path, exc)

        manifest = {subdir: self._scan_style_files(subdir) for subdir in self.style_subdirs}
        self.latent_cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1,
            "data_root": str(self.data_root),
            "style_subdirs": self.style_subdirs,
            "styles": {
                subdir: {
                    "count": len(files),
                    "files": [str(path.relative_to(self.data_root)) for path in files],
                }
                for subdir, files in manifest.items()
            },
        }
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(manifest_path)
        logger.info("Wrote latent manifest cache: %s", manifest_path)
        return manifest

    def _resolve_style_files(self, subdir: str) -> list[Path]:
        # In packed mode, we still need the full file list for stem registration
        # (style_item_stems, style_base_to_indices). The packed cache is loaded
        # separately in _load_style_stack. Returning [packed_path] here would
        # break stem registration (only 1 stem instead of N).
        if self._latent_manifest is not None and subdir in self._latent_manifest:
            return list(self._latent_manifest[subdir])
        return self._scan_style_files(subdir)

    def _load_style_stack(self, style_id: int, subdir: str, files: list[Path]) -> torch.Tensor:
        if self.latent_cache_mode == "packed":
            packed_dir = self.latent_cache_dir / "packed"
            packed_path = packed_dir / self._style_cache_name(style_id, subdir)
            if packed_path.exists():
                try:
                    payload = torch.load(packed_path, map_location="cpu", weights_only=False)
                    if (
                        isinstance(payload, dict)
                        and payload.get("schema") == 1
                        and payload.get("subdir") == subdir
                    ):
                        latents = torch.as_tensor(payload["latents"]).float()
                        if latents.ndim == 4:
                            logger.info("Loaded packed latent cache: %s (count=%d)", packed_path, payload.get("count", -1))
                            # Pin to shared memory so forked/spawned workers can
                            # access the tensor without an extra copy.
                            try:
                                latents.share_memory_()
                            except Exception:
                                pass
                            return latents
                except Exception as exc:
                    logger.warning("Ignoring stale packed latent cache %s (%s).", packed_path, exc)

        latents = [_load_latent_file(p) for p in files]
        stack = torch.stack(latents, dim=0)
        if self.latent_cache_mode == "packed":
            packed_dir = self.latent_cache_dir / "packed"
            packed_dir.mkdir(parents=True, exist_ok=True)
            packed_path = packed_dir / self._style_cache_name(style_id, subdir)
            payload = {
                "schema": 1,
                "subdir": subdir,
                "count": len(files),
                "files": [str(path.relative_to(self.data_root)) for path in files],
                "latents": stack,
            }
            tmp = packed_path.with_suffix(".tmp")
            torch.save(payload, tmp)
            tmp.replace(packed_path)
            logger.info("Wrote packed latent cache: %s", packed_path)
        try:
            stack.share_memory_()
        except Exception:
            pass
        return stack

    def _normalize_style_weights(self, weights: Sequence[float] | None, name: str) -> torch.Tensor | None:
        if weights is None:
            return None
        values = torch.as_tensor(list(weights), dtype=torch.float32)
        if values.numel() != len(self.style_subdirs):
            raise ValueError(f"{name} must have {len(self.style_subdirs)} values, got {values.numel()}")
        if not torch.isfinite(values).all() or torch.any(values < 0):
            raise ValueError(f"{name} must contain finite non-negative values")
        if float(values.sum().item()) <= 0.0:
            raise ValueError(f"{name} must have a positive sum")
        values = values / values.sum()
        logger.info("%s=%s for styles=%s", name, [round(float(v), 4) for v in values.tolist()], self.style_subdirs)
        return values

    def _build_balanced_target_style_ids(
        self,
        *,
        generator: torch.Generator,
        n_styles: int,
        length: int,
    ) -> torch.Tensor:
        if self.batch_size_hint <= 0:
            return torch.randint(0, n_styles, (length,), generator=generator)

        target_style_ids = torch.empty((length,), dtype=torch.long)
        offset = int(torch.randint(0, n_styles, (1,), generator=generator).item()) if n_styles > 1 else 0
        for start in range(0, length, self.batch_size_hint):
            end = min(length, start + self.batch_size_hint)
            block_len = end - start
            block = (torch.arange(block_len, dtype=torch.long) + offset) % n_styles
            perm = torch.randperm(block_len, generator=generator)
            target_style_ids[start:end] = block.index_select(0, perm)
            if n_styles > 1:
                offset = int((offset + torch.randint(1, n_styles, (1,), generator=generator).item()) % n_styles)
        return target_style_ids

    def _sample_style_ids(
        self,
        *,
        generator: torch.Generator,
        length: int,
        weights: torch.Tensor | None,
        n_styles: int,
    ) -> torch.Tensor:
        if weights is None:
            return torch.randint(0, n_styles, (length,), generator=generator)
        return torch.multinomial(weights, num_samples=length, replacement=True, generator=generator).long()

    def _estimate_dataset_bytes(self) -> int:
        total = 0
        for t in self.style_tensors.values():
            total += int(t.numel()) * int(t.element_size())
        return int(total)

    def _try_preload_to_gpu(self) -> None:
        if not torch.cuda.is_available():
            logger.warning("preload_to_gpu=True requested but CUDA is unavailable; using CPU dataset tensors.")
            return
        if not str(self.device).startswith("cuda"):
            logger.warning("preload_to_gpu=True requested but current device=%s is not CUDA; using CPU tensors.", self.device)
            return

        target_device = torch.device(self.device)
        needed_bytes = self._estimate_dataset_bytes()
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except Exception:
            free_bytes, total_bytes = 0, 0

        reserve_bytes = int(float(total_bytes) * self.preload_reserve_ratio) if total_bytes > 0 else 0
        allowed_bytes = max(0, int(free_bytes) - reserve_bytes) if free_bytes > 0 else 0
        if self.preload_max_vram_gb > 0.0:
            allowed_bytes = min(allowed_bytes, int(self.preload_max_vram_gb * (1024**3))) if allowed_bytes > 0 else int(self.preload_max_vram_gb * (1024**3))

        if allowed_bytes > 0 and needed_bytes > allowed_bytes:
            logger.warning(
                "Skip preload_to_gpu: need %.2fGB > allowed %.2fGB (free %.2fGB, reserve_ratio=%.2f).",
                needed_bytes / (1024**3),
                allowed_bytes / (1024**3),
                free_bytes / (1024**3),
                self.preload_reserve_ratio,
            )
            return

        gpu_tensors: Dict[int, torch.Tensor] = {}
        try:
            for style_id, stack in self.style_tensors.items():
                gpu_tensors[style_id] = stack.to(device=target_device, non_blocking=False)
        except RuntimeError as exc:
            logger.warning("preload_to_gpu failed (%s); fallback to CPU tensors.", exc)
            gpu_tensors.clear()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return

        self.style_tensors = gpu_tensors
        self.preload_to_gpu = True
        logger.info(
            "Dataset preloaded to %s: %.2fGB across %d style pools.",
            target_device,
            needed_bytes / (1024**3),
            len(self.style_tensors),
        )

    def _normalize_base_stem(self, stem: str) -> str:
        return _normalize_base_stem_text(stem)

    def _register_style_index(self, style_id: int, files: list[Path]) -> None:
        stems = [p.stem for p in files]
        self.style_item_stems[style_id] = stems
        base_to_indices: dict[str, list[int]] = {}
        style_name = self.style_subdirs[style_id]
        for idx, stem in enumerate(stems):
            for alias in _stem_aliases(style_name, stem):
                base_to_indices.setdefault(alias, []).append(idx)
        self.style_base_to_indices[style_id] = base_to_indices

    def _load_pairing_cache(self, cache_path: str) -> None:
        path = Path(cache_path)
        if not path.is_absolute():
            path = path.resolve()
        if not path.exists():
            logger.warning("pairing cache not found: %s", path)
            return

        # ------------------------------------------------------------------
        # Fast path: try to attach to a pre-built SHM block from a previous
        # process run.  The block name embeds the source file's mtime so it
        # auto-invalidates when the .pt file changes.
        # ------------------------------------------------------------------
        shm_name = _shm_name_for_path("pairmp", path)
        flat_cached = _shm_read_json(shm_name)
        if flat_cached is not None:
            try:
                pair_map: dict[tuple[str, str, str], list[str]] = {}
                for entry in flat_cached:
                    k = (str(entry[0][0]), str(entry[0][1]), str(entry[0][2]))
                    pair_map[k] = entry[1]
                self.offline_pairing_map = pair_map
                logger.info(
                    "Loaded pairing cache from SHM '%s' with %d source-target routes",
                    shm_name,
                    len(self.offline_pairing_map),
                )
                return
            except Exception as exc:
                logger.debug("SHM pairing cache decode failed (%s); reloading from disk.", exc)

        # ------------------------------------------------------------------
        # Slow path: load from disk (.pt pickle or .json).
        # ------------------------------------------------------------------
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)

        raw_pairs = payload.get("pairs", payload if isinstance(payload, dict) else {})
        pair_map = {}
        # flat list for SHM serialisation: [[src_style, src_stem, tgt_style], [tgt1, ...]]
        flat_for_shm: list[list] = []

        if isinstance(raw_pairs, dict):
            for key, value in raw_pairs.items():
                if isinstance(key, str) and "|" in key:
                    src_style, src_stem, tgt_style = key.split("|", 2)
                    targets = [str(x) for x in value][: self.pairing_cache_topk]
                    if targets:
                        k = (src_style, src_stem, tgt_style)
                        pair_map[k] = targets
                        flat_for_shm.append([[src_style, src_stem, tgt_style], targets])
                    continue
                if isinstance(value, dict):
                    src_style = str(key)
                    for src_stem, target_map in value.items():
                        if not isinstance(target_map, dict):
                            continue
                        for tgt_style, targets in target_map.items():
                            target_list = [str(x) for x in targets][: self.pairing_cache_topk]
                            if target_list:
                                k = (src_style, str(src_stem), str(tgt_style))
                                pair_map[k] = target_list
                                flat_for_shm.append([[src_style, str(src_stem), str(tgt_style)], target_list])

        self.offline_pairing_map = pair_map
        logger.info("Loaded pairing cache %s with %d source-target routes", path, len(self.offline_pairing_map))

        # Write-through to SHM so subsequent process starts use the fast path.
        if flat_for_shm:
            ok = _shm_write_json(shm_name, flat_for_shm)
            if ok:
                logger.info(
                    "Wrote pairing cache to SHM '%s' (%d routes, %.1f MB JSON)",
                    shm_name,
                    len(flat_for_shm),
                    sum(len(e[0][0]) + len(e[0][1]) + len(e[0][2]) + sum(len(t) for t in e[1])
                        for e in flat_for_shm) / 1e6,
                )

    def _active_pairing_topk(self, candidate_count: int) -> int:
        max_topk = max(1, min(int(candidate_count), int(self.pairing_cache_topk)))
        if self.pairing_cache_active_topk > 0:
            return max(1, min(max_topk, int(self.pairing_cache_active_topk)))
        min_topk = max(1, min(int(self.pairing_cache_min_topk), max_topk))
        if self.pairing_cache_rank_schedule == "fixed" or self.pairing_cache_curriculum_epochs <= 1:
            return max_topk
        progress = max(0.0, min(1.0, (float(self.epoch) - 1.0) / max(1.0, float(self.pairing_cache_curriculum_epochs - 1))))
        if self.pairing_cache_rank_schedule == "hard_to_easy":
            active = max_topk - progress * float(max_topk - min_topk)
        else:
            active = min_topk + progress * float(max_topk - min_topk)
        return max(1, min(max_topk, int(round(active))))

    def _sample_pairing_rank(self, *, active_topk: int, random_value: float, sample_index: int | None = None) -> int:
        if active_topk <= 1:
            return 0
        mode = self.pairing_cache_sample_mode
        if mode == "top1":
            return 0
        if mode in {"rank_stratified", "rank_biased_stratified", "reverse_rank_biased_stratified"}:
            if sample_index is None:
                u = max(0.0, min(1.0 - 1e-7, float(random_value)))
            else:
                # Deterministic per-epoch rank quantiles reduce batch-to-batch
                # target hardness jitter while still covering all active ranks.
                slot = (int(sample_index) + 9973 * int(self.epoch)) % active_topk
                u = (float(slot) + 0.5) / float(active_topk)
        else:
            u = max(0.0, min(1.0 - 1e-7, float(random_value)))
        if mode in {"rank_biased", "rank_biased_stratified"}:
            u = u ** self.pairing_cache_rank_power
        elif mode in {"reverse_rank_biased", "reverse_rank_biased_stratified"}:
            u = 1.0 - ((1.0 - u) ** self.pairing_cache_rank_power)
        return min(int(u * active_topk), active_topk - 1)

    def current_pairing_active_topk(self) -> int:
        return self._active_pairing_topk(self.pairing_cache_topk)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        
        # Optimization: Pre-compute all random indices for the epoch using vectorized operations.
        # This eliminates the heavy overhead of instantiating random.Random() per sample.
        N = self.length
        g = torch.Generator()
        g.manual_seed((self.epoch + 1) * 1000003)
        
        n_styles = len(self.style_subdirs)
        
        self._cache_content_style_ids = self._sample_style_ids(
            generator=g,
            length=N,
            weights=self.content_style_sampling_weights,
            n_styles=n_styles,
        )
        # Uniform target sampling across all styles (including source style).
        if self.identity_ratio is None:
            if self.target_style_sampling_weights is not None:
                self._cache_target_style_ids = self._sample_style_ids(
                    generator=g,
                    length=N,
                    weights=self.target_style_sampling_weights,
                    n_styles=n_styles,
                )
            elif self.balance_target_styles_per_batch:
                self._cache_target_style_ids = self._build_balanced_target_style_ids(
                    generator=g,
                    n_styles=n_styles,
                    length=N,
                )
            else:
                # Backward compatible behavior: uniform target sampling over all styles.
                self._cache_target_style_ids = torch.randint(0, n_styles, (N,), generator=g)
        else:
            # Controlled identity ratio:
            # - identity samples use target=source
            # - non-identity samples sample uniformly from all other styles.
            identity_mask = torch.rand(N, generator=g) < float(self.identity_ratio)
            target_style_ids = self._cache_content_style_ids.clone()
            if n_styles > 1:
                non_id = ~identity_mask
                non_id_count = int(non_id.sum().item())
                if non_id_count > 0:
                    rand_other = torch.randint(0, n_styles - 1, (non_id_count,), generator=g)
                    src_non_id = self._cache_content_style_ids[non_id]
                    adjusted = rand_other + (rand_other >= src_non_id).long()
                    target_style_ids[non_id] = adjusted
            self._cache_target_style_ids = target_style_ids

        # Random floats for selecting index within the chosen style
        self._cache_content_rands = torch.rand(N, generator=g)
        self._cache_target_rands = torch.rand(N, generator=g)
        self._cache_pairing_rands = torch.rand(N, generator=g)
        self._cache_pairing_explore_rands = torch.rand(N, generator=g)
        self._cache_pairing_dual_rands = torch.rand(N, generator=g)
        self._cache_pairing_aux_rands = torch.rand(N, generator=g)

        if self.allow_hflip:
            self._cache_flip_content = torch.rand(N, generator=g) < 0.5
            self._cache_flip_target = torch.rand(N, generator=g) < 0.5
        else:
            self._cache_flip_content = None
            self._cache_flip_target = None

    def __len__(self) -> int:
        return self.length

    def _maybe_flip(self, x: torch.Tensor, do_flip: torch.Tensor | None, idx: int) -> torch.Tensor:
        if do_flip is not None and do_flip[idx]:
            return torch.flip(x, dims=[-1])
        return x

    def _sample_target_index_from_pairing(
        self,
        *,
        content_style_id: int,
        content_index: int,
        target_style_id: int,
        fallback_index: int,
        random_value: float,
        rank_random_value: float | None = None,
        explore_random_value: float | None = None,
        sample_index: int | None = None,
        force_hard_topk: int = 0,
    ) -> int:
        if not self.offline_pairing_map:
            return fallback_index
        if self.pairing_cache_cross_only and content_style_id == target_style_id:
            return fallback_index

        src_style = self.style_subdirs[content_style_id]
        tgt_style = self.style_subdirs[target_style_id]
        src_stem = self._normalize_base_stem(self.style_item_stems[content_style_id][content_index])
        candidates = self.offline_pairing_map.get((src_style, src_stem, tgt_style))
        if not candidates:
            return fallback_index

        active_topk = self._active_pairing_topk(len(candidates))
        rank_random = float(random_value if rank_random_value is None else rank_random_value)
        explore_random = float(random_value if explore_random_value is None else explore_random_value)
        if force_hard_topk > 0:
            hard_topk = max(1, min(int(force_hard_topk), int(len(candidates)), int(self.pairing_cache_topk)))
            if hard_topk <= active_topk:
                return fallback_index
            hard_width = max(1, hard_topk - active_topk)
            hard_u = max(0.0, min(1.0 - 1e-7, rank_random))
            chosen_idx = active_topk + min(int(hard_u * hard_width), hard_width - 1)
            chosen_stem = candidates[chosen_idx]
            target_indices = self.style_base_to_indices[target_style_id].get(chosen_stem)
            if not target_indices:
                return fallback_index
            picked_variant = min(int(random_value * len(target_indices)), len(target_indices) - 1)
            return int(target_indices[picked_variant])

        explore_topk = self.pairing_cache_explore_topk if self.pairing_cache_explore_topk > 0 else self.pairing_cache_topk
        explore_topk = max(1, min(int(explore_topk), int(len(candidates)), int(self.pairing_cache_topk)))
        use_hard_explore = (
            self.pairing_cache_explore_prob > 0.0
            and explore_topk > active_topk
            and explore_random < self.pairing_cache_explore_prob
        )
        if use_hard_explore:
            hard_width = max(1, explore_topk - active_topk)
            chosen_idx = active_topk + min(int(rank_random * hard_width), hard_width - 1)
        else:
            chosen_idx = self._sample_pairing_rank(
                active_topk=active_topk,
                random_value=rank_random,
                sample_index=sample_index,
            )
        chosen_stem = candidates[chosen_idx]

        target_indices = self.style_base_to_indices[target_style_id].get(chosen_stem)
        if not target_indices:
            return fallback_index
        picked_variant = min(int(random_value * len(target_indices)), len(target_indices) - 1)
        return int(target_indices[picked_variant])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        # Ultra-lightweight getitem using pre-computed indices
        content_style_id = int(self._cache_content_style_ids[index])
        target_style_id = int(self._cache_target_style_ids[index])

        c_pool = self.style_tensors[content_style_id]
        t_pool = self.style_tensors[target_style_id]

        c_idx = int(self._cache_content_rands[index] * c_pool.shape[0])
        t_idx = int(self._cache_target_rands[index] * t_pool.shape[0])
        t_idx = self._sample_target_index_from_pairing(
            content_style_id=content_style_id,
            content_index=c_idx,
            target_style_id=target_style_id,
            fallback_index=t_idx,
            random_value=float(self._cache_target_rands[index]),
            rank_random_value=float(self._cache_pairing_rands[index]),
            explore_random_value=float(self._cache_pairing_explore_rands[index]),
            sample_index=index,
        )

        content = self._maybe_flip(c_pool[c_idx], self._cache_flip_content, index)
        target_style = self._maybe_flip(t_pool[t_idx], self._cache_flip_target, index)
        aux_target_style = None
        aux_target_valid = False
        if self.pairing_cache_dual_target_mix > 0.0:
            hard_topk = self.pairing_cache_dual_target_topk if self.pairing_cache_dual_target_topk > 0 else self.pairing_cache_topk
            hard_idx = self._sample_target_index_from_pairing(
                content_style_id=content_style_id,
                content_index=c_idx,
                target_style_id=target_style_id,
                fallback_index=t_idx,
                random_value=float(self._cache_target_rands[index]),
                rank_random_value=float(self._cache_pairing_dual_rands[index]),
                sample_index=index,
                force_hard_topk=hard_topk,
            )
            if hard_idx != t_idx:
                hard_target = self._maybe_flip(t_pool[hard_idx], self._cache_flip_target, index)
                target_style = torch.lerp(target_style, hard_target, self.pairing_cache_dual_target_mix)
        if self.pairing_cache_aux_target_topk > 0:
            aux_idx = self._sample_target_index_from_pairing(
                content_style_id=content_style_id,
                content_index=c_idx,
                target_style_id=target_style_id,
                fallback_index=t_idx,
                random_value=float(self._cache_target_rands[index]),
                rank_random_value=float(self._cache_pairing_aux_rands[index]),
                sample_index=index,
                force_hard_topk=self.pairing_cache_aux_target_topk,
            )
            aux_target_valid = aux_idx != t_idx
            aux_target_style = self._maybe_flip(t_pool[aux_idx], self._cache_flip_target, index)

        item = {
            "content": content,
            "target_style": target_style,
            "target_style_id": target_style_id,
            "source_style_id": content_style_id,
        }
        if self.pairing_cache_aux_target_topk > 0:
            item["aux_target_style"] = aux_target_style if aux_target_style is not None else target_style
            item["aux_target_valid"] = torch.tensor(float(aux_target_valid), dtype=torch.float32)
        if self.style_caption_path:
            target_style_name = self.style_subdirs[target_style_id]
            target_stem = self.style_item_stems[target_style_id][t_idx]
            target_base = self._normalize_base_stem(target_stem)
            matched_rel = ""
            caption_text = ""
            for ext in [".jpg", ".jpeg", ".png"]:
                rel = f"{target_style_name}/{target_base}{ext}"
                prefix_rel = f"{target_style_name}/{target_style_name}__{target_base}{ext}"
                if self.style_captions:
                    if rel in self.style_captions:
                        caption_text = self.style_captions[rel]
                        matched_rel = rel
                        break
                    if prefix_rel in self.style_captions:
                        caption_text = self.style_captions[prefix_rel]
                        matched_rel = prefix_rel
                        break
                else:
                    matched_rel = prefix_rel
                    break
            else:
                if self.style_captions:
                    matched_rel = ""
                else:
                    matched_rel = f"{target_style_name}/{target_style_name}__{target_base}.jpg"
            item["target_style_caption"] = caption_text
            item["target_style_caption_rel_path"] = matched_rel
        return item
