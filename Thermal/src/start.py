import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch._inductor.config as inductor_config

# 1. 开启 FX 图缓存 (持久化 torch.compile，要求 PyTorch >= 2.2)
inductor_config.fx_graph_cache = True
# 2. 🚀 Infra Optimization: Force Cache to Linux Native FS (Avoid /mnt/c latency)
# 使用用户主目录下的 .cache，确保是 ext4 文件系统
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.expanduser("~/.cache/torch_compile")
# 3. 针对 4070 特性微调：强制唯一内核名称并减少冗余搜索
inductor_config.triton.unique_kernel_names = True
inductor_config.fallback_random = True

from torch.utils.data import DataLoader

from utils.dataset import LatentDataset
from trainer import LGTTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _resolve_num_workers(config) -> int:
    # 🚀 Optimization: If data is on GPU, workers must be 0 to avoid multiprocessing overhead
    if config['training'].get('preload_data_to_gpu', False):
        return 0
        
    cfg_workers = config['training'].get('num_workers')
    if cfg_workers is not None:
        return int(cfg_workers)
    cpu_count = os.cpu_count() or 1
    return max(2, cpu_count // 2)


def main() -> None:
    parser = argparse.ArgumentParser(description='LGT Training with modular pipeline')
    parser.add_argument('--config', type=str, default='config.json', help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        config = json.load(f)

    if args.resume:
        config['training']['resume_checkpoint'] = args.resume
        logger.info(f"Overriding resume checkpoint: {args.resume}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    dataset = LatentDataset(
        data_root=config['data']['data_root'],
        num_styles=config['model']['num_styles'],
        style_subdirs=config['data'].get('style_subdirs'),
        config=config,
    )

    num_workers = _resolve_num_workers(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    trainer = LGTTrainer(config, device=device, config_path=str(config_path))
    trainer.on_training_start(dataloader)

    for epoch in range(trainer.start_epoch, trainer.num_epochs + 1):
        metrics = trainer.train_epoch(dataloader, epoch)
        trainer.step_scheduler()
        current_lr = trainer.optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch}/{trainer.num_epochs} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"SWD: {metrics['style_swd']:.4f} | "
            f"MSE: {metrics['mse']:.4f} | "
            f"W_Str: {metrics.get('m_mse', 0.0):.2f} | "
            f"W_Sty: {metrics.get('m_style', 0.0):.2f} | "
            f"LR: {current_lr:.2e}"
        )

        trainer.log_epoch(epoch, metrics)

        if epoch % trainer.save_interval == 0 or epoch == trainer.num_epochs:
            trainer.save_checkpoint(epoch, metrics)

        if epoch % trainer.eval_interval == 0 or epoch == trainer.num_epochs:
            trainer.evaluate_and_infer(epoch)
            if trainer.full_eval_interval is not None and (
                epoch % trainer.full_eval_interval == 0 or epoch == trainer.num_epochs
            ):
                try:
                    trainer.run_full_evaluation(epoch)
                except Exception as exc:
                    logger.error(f"Full external evaluation failed for epoch {epoch}: {exc}")

    logger.info("✓ Training completed!")


if __name__ == "__main__":
    main()
