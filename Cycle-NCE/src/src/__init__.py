"""
Latent AdaCUT (Statistical Version)

Core recipe:
- AdaGN injection
- 16x16 spatial pre-inject
- Stroke-Gram + Color-Moment style alignment
- Semigroup + Delta regularization
"""

from .model import LatentAdaCUT, count_parameters
from .trainer import AdaCUTTrainer

__all__ = [
    "LatentAdaCUT",
    "AdaCUTTrainer",
    "count_parameters",
]
