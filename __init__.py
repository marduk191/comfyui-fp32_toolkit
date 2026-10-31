"""ComfyUI FP32 Toolkit — Blackwell-optimized acceleration for FP32 diffusion models.

Target: Tongyi Z-Image-Turbo FP32 on RTX 5090 (sm_120).
"""
from .nodes.global_settings import NODE_CLASS_MAPPINGS as _GS, NODE_DISPLAY_NAME_MAPPINGS as _GSD
from .nodes.loader import NODE_CLASS_MAPPINGS as _LD, NODE_DISPLAY_NAME_MAPPINGS as _LDD
from .nodes.autocast import NODE_CLASS_MAPPINGS as _AC, NODE_DISPLAY_NAME_MAPPINGS as _ACD
from .nodes.selective_cast import NODE_CLASS_MAPPINGS as _SC, NODE_DISPLAY_NAME_MAPPINGS as _SCD
from .nodes.attention import NODE_CLASS_MAPPINGS as _AT, NODE_DISPLAY_NAME_MAPPINGS as _ATD
from .nodes.quantize import NODE_CLASS_MAPPINGS as _QT, NODE_DISPLAY_NAME_MAPPINGS as _QTD
from .nodes.compile_node import NODE_CLASS_MAPPINGS as _CP, NODE_DISPLAY_NAME_MAPPINGS as _CPD
from .nodes.memory import NODE_CLASS_MAPPINGS as _MM, NODE_DISPLAY_NAME_MAPPINGS as _MMD
from .nodes.cache import NODE_CLASS_MAPPINGS as _CA, NODE_DISPLAY_NAME_MAPPINGS as _CAD
from .nodes.preset import NODE_CLASS_MAPPINGS as _PR, NODE_DISPLAY_NAME_MAPPINGS as _PRD
from .nodes.diagnostic import NODE_CLASS_MAPPINGS as _DG, NODE_DISPLAY_NAME_MAPPINGS as _DGD
from .nodes.save_model import NODE_CLASS_MAPPINGS as _SV, NODE_DISPLAY_NAME_MAPPINGS as _SVD

NODE_CLASS_MAPPINGS = {**_GS, **_LD, **_AC, **_SC, **_AT, **_QT, **_CP, **_MM, **_CA, **_PR, **_DG, **_SV}
NODE_DISPLAY_NAME_MAPPINGS = {**_GSD, **_LDD, **_ACD, **_SCD, **_ATD, **_QTD, **_CPD, **_MMD, **_CAD, **_PRD, **_DGD, **_SVD}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
