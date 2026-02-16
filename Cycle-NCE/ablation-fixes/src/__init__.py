"""
Latent AdaCUT (Statistical Version)

Core recipe:
- AdaGN injection
- 16x16 spatial pre-inject
- Stroke-Gram + Color-Moment style alignment
- Struct/Edge + Delta-TV stability losses
"""

from .model import LatentAdaCUT, count_parameters
from .trainer import AdaCUTTrainer

__all__ = [
    "LatentAdaCUT",
    "AdaCUTTrainer",
    "count_parameters",
]
