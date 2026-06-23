"""On-the-fly weight quantization via torchao. FP32 master weights stay on disk.

FP8 (e4m3) and FP4 (MXFP4 / NVFP4) are Blackwell-native (sm_120).
"""
import torch
import torch.nn as nn
import comfy.model_management as mm

try:
    from torchao.quantization import quantize_
    _HAS_TORCHAO = True
except ImportError:
    _HAS_TORCHAO = False


def _exclude_norm_embed(mod, fqn):
    """Filter fn for torchao quantize_. Signature: (module, fqn_string).
    Returns True for layers we WANT to quantize."""
    if not isinstance(mod, nn.Linear):
        return False
    if isinstance(mod, (nn.LayerNorm, nn.GroupNorm, nn.Embedding)):
        return False
    if "rmsnorm" in type(mod).__name__.lower():
        return False
    n = (fqn or "").lower()
    if any(k in n for k in ("final_layer", "proj_out", "adaln_final", "embedder", "time_embed", "label_emb")):
        return False
    return True


class BlackwellFP8Quantize:
    """Dynamic FP8 e4m3 activation + FP8 e4m3 weight. Native Blackwell tensor cores."""
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
        if not _HAS_TORCHAO:
            raise RuntimeError("torchao not installed. pip install torchao")
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        try:
            from torchao.quantization import PerRow, PerTensor
            granularity = PerRow() if scheme == "per_row" else PerTensor()
            cfg = Float8DynamicActivationFloat8WeightConfig(granularity=granularity)
        except ImportError:
            cfg = Float8DynamicActivationFloat8WeightConfig()

        m = model.clone()
        mm.load_models_gpu([m])
        diff = m.get_model_object("diffusion_model")
        filter_fn = _exclude_norm_embed if exclude_sensitive_layers else None
        quantize_(diff, cfg, filter_fn=filter_fn)
        return (m,)


class BlackwellMXQuantize:
    """Microscale MX/NVFP weight quantization via torchao prototype. Blackwell sm_120 native.

    Formats:
      - mxfp8_e4m3 / mxfp8_e5m2: 8-bit microscale, block-32 power-of-2 scales
      - mxfp6_e2m3 / mxfp6_e3m2: 6-bit microscale
      - mxfp4:                   4-bit microscale, block-32
      - nvfp4:                   4-bit Nvidia format, block-16, FP8 scales (best 4-bit quality)
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

    _WEIGHT_DTYPE = {
        "mxfp8_e4m3": torch.float8_e4m3fn,
        "mxfp8_e5m2": torch.float8_e5m2,
        # FP6 variants exposed as torch dtypes in torchao prototype:
        "mxfp6_e2m3": "float6_e2m3fn",
        "mxfp6_e3m2": "float6_e3m2fn",
        "mxfp4":      "float4_e2m1fn_x2",
    }

    def patch(self, model, format, activation_dtype, exclude_sensitive_layers):
        if not _HAS_TORCHAO:
            raise RuntimeError("torchao not installed. pip install torchao")

        cap = torch.cuda.get_device_capability()
        if cap[0] < 9 and format != "mxfp4" and format != "nvfp4":
            print(f"[MXQuantize] warning: MX FP8/FP6 best on sm_90+, FP4 on sm_120. Detected sm_{cap[0]}{cap[1]}")
        if format in ("mxfp4", "nvfp4") and cap[0] < 12:
            raise RuntimeError(f"FP4 requires Blackwell (sm_120). Detected: sm_{cap[0]}{cap[1]}")

        try:
            from torchao.prototype.mx_formats import MXFPInferenceConfig, NVFP4InferenceConfig
        except ImportError as e:
            raise RuntimeError(f"torchao MX formats unavailable: {e}. Update torchao >= 0.9")

        if format == "nvfp4":
            cfg = NVFP4InferenceConfig()
        else:
            wd = self._WEIGHT_DTYPE[format]
            if isinstance(wd, str):
                wd = getattr(torch, wd, None) or getattr(torch, wd.replace("_x2", ""), None)
                if wd is None:
                    raise RuntimeError(f"torch dtype for {format} not in this PyTorch build")
            if activation_dtype == "match_weight":
                ad = wd
            else:
                ad = torch.bfloat16 if activation_dtype == "bf16" else torch.float16
            cfg = MXFPInferenceConfig(activation_dtype=ad, weight_dtype=wd)

        m = model.clone()
        mm.load_models_gpu([m])
        diff = m.get_model_object("diffusion_model")
        filter_fn = _exclude_norm_embed if exclude_sensitive_layers else None
        quantize_(diff, cfg, filter_fn=filter_fn)
        print(f"[MXQuantize] applied {format}")
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
        if not _HAS_TORCHAO:
            raise RuntimeError("torchao not installed.")
        from torchao.quantization import int8_weight_only
        m = model.clone()
        mm.load_models_gpu([m])
        diff = m.get_model_object("diffusion_model")
        filter_fn = _exclude_norm_embed if exclude_sensitive_layers else None
        quantize_(diff, int8_weight_only(), filter_fn=filter_fn)
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
