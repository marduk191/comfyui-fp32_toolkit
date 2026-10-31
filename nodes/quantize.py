"""On-the-fly weight quantization via torchao. FP32 master weights stay on disk.

FP8 (e4m3) needs sm_89+; MX formats and NVFP4 are Blackwell-native (sm_100 / sm_120).
Supports the current torchao config API (>= 0.13) and falls back to the older names.
"""
import torch
import torch.nn as nn
import comfy.model_management as mm

from ._common import clone_for_weight_edit, cuda_capability, is_plain_tensor
from .selective_cast import _wrap_dtype_bridge

try:
    import torchao.quantization as _tq
    from torchao.quantization import quantize_
    _HAS_TORCHAO = True
except ImportError:
    _tq = None
    _HAS_TORCHAO = False

_SENSITIVE_NAME_KEYS = ("final_layer", "proj_out", "adaln_final", "embedder", "time_embed", "label_emb")


def _quantizable(mod, fqn):
    """Plain (not yet quantized) Linear layers only, so re-running a node never double-quantizes."""
    return isinstance(mod, nn.Linear) and is_plain_tensor(mod.weight)


def _exclude_norm_embed(mod, fqn):
    """Filter fn for torchao quantize_. Signature: (module, fqn_string).
    Returns True for layers we WANT to quantize."""
    if not _quantizable(mod, fqn):
        return False
    if "rmsnorm" in type(mod).__name__.lower():
        return False
    n = (fqn or "").lower()
    return not any(k in n for k in _SENSITIVE_NAME_KEYS)


def _require_torchao():
    if not _HAS_TORCHAO:
        raise RuntimeError("torchao not installed. pip install torchao (must match your torch/CUDA build)")


def _require_cuda(min_cap, what):
    cap = cuda_capability()
    if cap is None:
        raise RuntimeError(f"{what} needs a CUDA GPU (sm_{min_cap[0]}{min_cap[1]}+); none detected")
    if tuple(cap) < tuple(min_cap):
        raise RuntimeError(f"{what} needs sm_{min_cap[0]}{min_cap[1]}+. Detected: sm_{cap[0]}{cap[1]}")
    return cap


def _apply(model, cfg, exclude_sensitive_layers, weight_dtype=None):
    m = clone_for_weight_edit(model)
    mm.load_models_gpu([m])
    diff = m.get_model_object("diffusion_model")
    filter_fn = _exclude_norm_embed if exclude_sensitive_layers else _quantizable
    n = 0
    for fqn, mod in diff.named_modules():
        if filter_fn(mod, fqn):
            n += 1
            if weight_dtype is not None and mod.weight.dtype != weight_dtype:
                mod.to(dtype=weight_dtype)
                _wrap_dtype_bridge(mod)  # keep fp32 callers working outside autocast
    if n == 0:
        print("[FP32Toolkit quantize] no un-quantized Linear layers matched (already quantized?); model left "
              "unchanged. Reload the model to switch quantization format.")
        return m, 0
    quantize_(diff, cfg, filter_fn=filter_fn)
    return m, n


class BlackwellFP8Quantize:
    """Dynamic FP8 e4m3 activation + FP8 e4m3 weight. Native Ada/Hopper/Blackwell tensor cores."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scheme": (["per_tensor", "per_row"], {"default": "per_row"}),
                "exclude_sensitive_layers": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "FP32Toolkit/quantize"

    def patch(self, model, scheme, exclude_sensitive_layers):
        _require_torchao()
        _require_cuda((8, 9), "FP8 dynamic quantization")
        granularity = _tq.PerRow() if scheme == "per_row" else _tq.PerTensor()
        cfg = _tq.Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
        m, n = _apply(model, cfg, exclude_sensitive_layers)
        print(f"[FP8Quantize] {scheme}: {n} Linear layers")
        return (m,)


_MX_ELEM = {
    "mxfp8_e4m3": lambda: torch.float8_e4m3fn,
    "mxfp8_e5m2": lambda: torch.float8_e5m2,
    "mxfp6_e2m3": lambda: _mx_constant("DTYPE_FP6_E2M3"),
    "mxfp6_e3m2": lambda: _mx_constant("DTYPE_FP6_E3M2"),
    "mxfp4": lambda: torch.float4_e2m1fn_x2,
}


def _mx_constant(name):
    from torchao.prototype.mx_formats import constants
    return getattr(constants, name)


def _mx_config(format, activation_dtype):
    """Build the torchao MX/NVFP4 config. New API first (torchao >= 0.13), old names as fallback."""
    try:
        import torchao.prototype.mx_formats as mx
    except ImportError as e:
        raise RuntimeError(f"torchao MX formats unavailable: {e}. Update torchao")

    if format == "nvfp4":
        if activation_dtype == "match_weight":
            cls = getattr(mx, "NVFP4DynamicActivationNVFP4WeightConfig", None)
        else:
            cls = getattr(mx, "NVFP4WeightOnlyConfig", None)
        if cls is not None:
            return cls()
        if hasattr(mx, "NVFP4InferenceConfig"):  # torchao <= 0.12
            return mx.NVFP4InferenceConfig()
        raise RuntimeError("this torchao has no NVFP4 inference config")

    elem = _MX_ELEM[format]()
    if hasattr(mx, "MXDynamicActivationMXWeightConfig"):
        if activation_dtype != "match_weight":
            print(f"[MXQuantize] torchao MX only supports activation dtype == weight dtype; "
                  f"using {format} activations instead of {activation_dtype}")
        return mx.MXDynamicActivationMXWeightConfig(activation_dtype=elem, weight_dtype=elem)
    if hasattr(mx, "MXFPInferenceConfig"):  # torchao <= 0.12
        if activation_dtype == "match_weight":
            ad = elem
        else:
            ad = torch.bfloat16 if activation_dtype == "bf16" else torch.float16
        return mx.MXFPInferenceConfig(activation_dtype=ad, weight_dtype=elem)
    raise RuntimeError("this torchao has no MX inference config")


class BlackwellMXQuantize:
    """Microscale MX/NVFP weight quantization via torchao prototype. Blackwell native.

    Formats:
      - mxfp8_e4m3 / mxfp8_e5m2: 8-bit microscale, block-32 power-of-2 scales
      - mxfp6_e2m3 / mxfp6_e3m2: 6-bit microscale
      - mxfp4:                   4-bit microscale, block-32
      - nvfp4:                   4-bit Nvidia format, block-16, FP8 scales (best 4-bit quality)

    activation_dtype: match_weight = quantize activations too (dynamic); bf16/fp16 = weight-only
    (NVFP4 only; torchao's MX path always quantizes activations to the weight format).
    Quantized layers are cast to bf16 first, which torchao's MX/NVFP4 kernels require.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "format": (["mxfp8_e4m3", "mxfp8_e5m2", "mxfp6_e2m3", "mxfp6_e3m2", "mxfp4", "nvfp4"],
                           {"default": "mxfp8_e4m3"}),
                "activation_dtype": (["bf16", "fp16", "match_weight"], {"default": "match_weight"}),
                "exclude_sensitive_layers": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "FP32Toolkit/quantize"

    def patch(self, model, format, activation_dtype, exclude_sensitive_layers):
        _require_torchao()
        if format in ("mxfp4", "nvfp4"):
            _require_cuda((10, 0), f"{format} (FP4)")
        else:
            cap = _require_cuda((8, 9), f"{format}")
            if cap[0] < 10:
                print(f"[MXQuantize] warning: MX formats run natively on Blackwell (sm_100+). Detected sm_{cap[0]}{cap[1]}")
        cfg = _mx_config(format, activation_dtype)
        m, n = _apply(model, cfg, exclude_sensitive_layers, weight_dtype=torch.bfloat16)
        print(f"[MXQuantize] applied {format} to {n} Linear layers")
        return (m,)


# Backward-compat alias
BlackwellFP4Quantize = BlackwellMXQuantize


class BlackwellInt8Quantize:
    """INT8 weight-only fallback for non-Blackwell or compatibility tests."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "exclude_sensitive_layers": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "FP32Toolkit/quantize"

    def patch(self, model, exclude_sensitive_layers):
        _require_torchao()
        if hasattr(_tq, "Int8WeightOnlyConfig"):
            cfg = _tq.Int8WeightOnlyConfig()
        else:  # torchao < 0.10
            cfg = _tq.int8_weight_only()
        m, n = _apply(model, cfg, exclude_sensitive_layers)
        print(f"[Int8Quantize] {n} Linear layers")
        return (m,)


NODE_CLASS_MAPPINGS = {
    "BlackwellFP8Quantize": BlackwellFP8Quantize,
    "BlackwellMXQuantize": BlackwellMXQuantize,
    "BlackwellInt8Quantize": BlackwellInt8Quantize,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BlackwellFP8Quantize": "Blackwell FP8 Quantize (per-tensor/per-row)",
    "BlackwellMXQuantize": "Blackwell MX Quantize (MXFP8/MXFP6/MXFP4/NVFP4)",
    "BlackwellInt8Quantize": "INT8 Weight-Only Quantize",
}
