"""
Latent AdaCUT (Statistical Version)

Core recipe:
- AdaGN injection
- SWD distribution alignment
- Moment matching
- InfoNCE structure lock
"""

from .model import LatentAdaCUT, count_parameters
from .trainer import AdaCUTTrainer

__all__ = [
    "LatentAdaCUT",
    "AdaCUTTrainer",
    "count_parameters",
]
