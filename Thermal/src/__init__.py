"""
LGT: Latent Geometric Thermodynamics

Geometric free energy optimization for style transfer in VAE latent space.
"""

__version__ = "1.0.0"

from .model import LGTUNetLite, TimestepEmbedding, StyleEmbedding, count_parameters
from .losses import (
    PatchSlicedWassersteinLoss,
    MultiScaleSWDLoss,
    GeometricFreeEnergyLoss,
    VelocityRegularizationLoss
)
from .utils.inference import (
    LangevinSampler,
    LGTInference,
    load_vae,
    encode_image,
    decode_latent,
    tensor_to_pil
)

__all__ = [
    # Model
    "LGTUNetLite",
    "TimestepEmbedding",
    "StyleEmbedding",
    "count_parameters",
    
    # Losses
    "PatchSlicedWassersteinLoss",
    "MultiScaleSWDLoss",
    "GeometricFreeEnergyLoss",
    "VelocityRegularizationLoss",
    
    # Inference
    "LangevinSampler",
    "LGTInference",
    "load_vae",
    "encode_image",
    "decode_latent",
    "tensor_to_pil",
]
