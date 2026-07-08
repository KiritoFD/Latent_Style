import sys, importlib
from pathlib import Path
S = Path(r"I:/GitHub/Latent_Style/SchrodingerBridge/tools")
for p in [str(S), str(S / "zstar"), str(S / "styleshot")]:
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    from zstar.zstar import ReweightCrossAttentionControl
    from zstar.diffuser_utils import ZstarPipeline
    from zstar.zstar_utils import AttentionBase, regiter_attention_editor_diffusers
    import ptp_utils
    print("ZSTAR import OK")
except Exception as e:
    print("ZSTAR import FAIL", repr(e))
try:
    from annotator.hed import SOFT_HEDdetector
    from ip_adapter import StyleShot, StyleContentStableDiffusionControlNetPipeline
    print("STYLESHOT import OK")
except Exception as e:
    print("STYLESHOT import FAIL", repr(e))
