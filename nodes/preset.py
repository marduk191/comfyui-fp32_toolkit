"""One-shot preset stack tuned for Z-Image-Turbo FP32 on RTX 5090."""
import torch
import comfy.model_management as mm

from .global_settings import FP32GlobalSettings
from .autocast import FP32AutocastWrapper
from .attention import SDPABackendSelect, FP32AttentionPrecision
from .selective_cast import FP32SelectiveCast
from .quantize import BlackwellFP8Quantize, BlackwellMXQuantize
from .compile_node import BlackwellTorchCompile


_PRESETS = {
    "safe":       {"selective": True, "autocast": "bf16", "quant": "none",   "compile": False, "attn_fp32": True},
    "balanced":   {"selective": True, "autocast": "bf16", "quant": "fp8",    "compile": True,  "attn_fp32": True},
    "aggressive": {"selective": True, "autocast": "bf16", "quant": "mxfp8",  "compile": True,  "attn_fp32": False},
    "extreme":    {"selective": True, "autocast": "bf16", "quant": "nvfp4",  "compile": True,  "attn_fp32": False},
}


class ZImageTurboOptimizer:
    """Apply a tuned acceleration stack to a Z-Image-Turbo FP32 model."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (list(_PRESETS.keys()), {"default": "balanced"}),
                "compile_mode": (["default", "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune"],
                                 {"default": "max-autotune-no-cudagraphs"}),
                "sdpa_backend": (["auto", "cudnn+flash", "cudnn", "flash"], {"default": "cudnn+flash"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "FP32Toolkit/preset"

    def apply(self, model, preset, compile_mode, sdpa_backend):
        cfg = _PRESETS[preset]

        # Global flags first
        FP32GlobalSettings().apply(
            passthrough=model,
            matmul_precision="high",
            tf32_matmul=True, tf32_cudnn=True,
            cudnn_benchmark=True,
            allow_fp16_reduced_precision_reduction=True,
            allow_fp16_accumulation=True,
            allow_bf16_reduced_precision_reduction=True,
            deterministic=False,
        )

        m = model

        if cfg["selective"]:
            (m,) = FP32SelectiveCast().patch(
                m, linear_dtype="bf16", conv_dtype="keep",
                preserve_norms_fp32=True,
                preserve_embeddings_fp32=True,
                preserve_final_layer_fp32=True,
            )

        if cfg["quant"] == "fp8":
            (m,) = BlackwellFP8Quantize().patch(m, scheme="per_row", exclude_sensitive_layers=True)
        elif cfg["quant"] == "mxfp8":
            (m,) = BlackwellMXQuantize().patch(m, format="mxfp8_e4m3",
                                                activation_dtype="match_weight",
                                                exclude_sensitive_layers=True)
        elif cfg["quant"] == "nvfp4":
            (m,) = BlackwellMXQuantize().patch(m, format="nvfp4",
                                                activation_dtype="bf16",
                                                exclude_sensitive_layers=True)

        if cfg["attn_fp32"]:
            (m,) = FP32AttentionPrecision().patch(m, attn_precision="fp32")
        else:
            (m,) = FP32AttentionPrecision().patch(m, attn_precision="default")

        (m,) = SDPABackendSelect().patch(m, backend=sdpa_backend)
        (m,) = FP32AutocastWrapper().patch(m, autocast_dtype=cfg["autocast"], cache_enabled=True)

        if cfg["compile"]:
            (m,) = BlackwellTorchCompile().patch(
                m, backend="inductor", mode=compile_mode,
                fullgraph=False, dynamic=False,
                cutlass_backend=False, coordinate_descent_tuning=True,
            )

        return (m,)


NODE_CLASS_MAPPINGS = {"ZImageTurboOptimizer": ZImageTurboOptimizer}
NODE_DISPLAY_NAME_MAPPINGS = {"ZImageTurboOptimizer": "Z-Image-Turbo Optimizer (Blackwell)"}
