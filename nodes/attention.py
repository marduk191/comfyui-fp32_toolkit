"""Attention backend selection + precision control.

All patches go through ComfyUI's ``optimized_attention_override`` / unet wrapper hooks, so they
apply to every model that uses ``comfy.ldm.modules.attention`` (UNets, Lumina2 / Z-Image, Flux, ...).
"""
import inspect
import logging

import torch
import comfy.ops

from ._common import chain_unet_wrapper, chain_attention_override

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _HAS_SDPA_KERNEL = True
    _SDPA_SET_PRIORITY = "set_priority" in inspect.signature(sdpa_kernel).parameters
except ImportError:
    _HAS_SDPA_KERNEL = False
    _SDPA_SET_PRIORITY = False

try:
    import sageattention
    _HAS_SAGE = True
except ImportError:
    sageattention = None
    _HAS_SAGE = False

log = logging.getLogger("FP32Toolkit")


_BACKEND_MAP = {
    "auto": None,
    "cudnn": ["CUDNN_ATTENTION"],
    "flash": ["FLASH_ATTENTION"],
    "efficient": ["EFFICIENT_ATTENTION"],
    "math": ["MATH"],
    "cudnn+flash": ["CUDNN_ATTENTION", "FLASH_ATTENTION"],
    "flash+efficient": ["FLASH_ATTENTION", "EFFICIENT_ATTENTION"],
}


def _resolve_backends(names):
    backends = [getattr(SDPBackend, n) for n in names]
    # MATH as a last resort so an unsupported shape/dtype degrades instead of raising "No available kernel".
    if SDPBackend.MATH not in backends:
        backends.append(SDPBackend.MATH)
    return backends


class SDPABackendSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (list(_BACKEND_MAP.keys()), {"default": "cudnn+flash"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "FP32Toolkit/attention"
    DESCRIPTION = "Pick the PyTorch SDPA kernel priority (listed backends first, MATH as fallback)."

    def patch(self, model, backend):
        m = model.clone()
        if backend == "auto" or not _HAS_SDPA_KERNEL:
            return (m,)
        backends = _resolve_backends(_BACKEND_MAP[backend])
        kernel_kwargs = {"set_priority": True} if _SDPA_SET_PRIORITY else {}

        def factory(call_next):
            def wrapper(apply_model, args):
                # ComfyUI wraps each SDPA call in its own sdpa_kernel(comfy.ops.SDPA_BACKEND_PRIORITY),
                # which would override an outer context, so swap that list for the duration of the call.
                had = hasattr(comfy.ops, "SDPA_BACKEND_PRIORITY")
                prev = getattr(comfy.ops, "SDPA_BACKEND_PRIORITY", None)
                if had:
                    comfy.ops.SDPA_BACKEND_PRIORITY = backends
                try:
                    with sdpa_kernel(backends, **kernel_kwargs):
                        return call_next(apply_model, args)
                finally:
                    if had:
                        comfy.ops.SDPA_BACKEND_PRIORITY = prev
            return wrapper

        chain_unet_wrapper(m, factory)
        return (m,)


class FP32AttentionPrecision:
    """Force the attention upcast precision (softmax/QK^T) even when surrounding compute is bf16/fp16.

    Applies to ComfyUI's basic / split / sub-quadratic attention paths. PyTorch SDPA kernels
    (flash / cuDNN / efficient) already accumulate the softmax in fp32 internally.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "attn_precision": (["fp32", "bf16", "fp16", "default"], {"default": "fp32"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "FP32Toolkit/attention"

    _DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16, "default": None}

    def patch(self, model, attn_precision):
        m = model.clone()
        prec = self._DTYPES[attn_precision]
        to = m.model_options.setdefault("transformer_options", {})
        if prec is None:
            to.pop("attn_precision", None)
            return (m,)
        to["attn_precision"] = prec

        def override(call_next, *args, **kwargs):
            # signature: (q, k, v, heads, mask=None, attn_precision=None, ...)
            if len(args) < 6 and kwargs.get("attn_precision") is None:
                kwargs["attn_precision"] = prec
            return call_next(*args, **kwargs)

        chain_attention_override(m, override)
        return (m,)


# Accept the names used by sageattention 2.x; older pack versions exposed shorthand names.
_SAGE_VARIANTS = {
    "sageattn": "sageattn",
    "sageattn_qk_int8_pv_fp16_cuda": "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton": "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda": "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp16": "sageattn_qk_int8_pv_fp16_cuda",  # legacy alias
}


def _make_sage_override(sage_fn):
    def override(call_next, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
                 skip_output_reshape=False, **kwargs):
        def fallback():
            return call_next(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                             skip_output_reshape=skip_output_reshape, **kwargs)

        # Sage kernels need fp16/bf16 and do not take arbitrary masks.
        if mask is not None or q.dtype not in (torch.float16, torch.bfloat16) or kwargs.get("low_precision_attention", True) is False:
            return fallback()
        try:
            if skip_reshape:  # (B, H, N, D)
                b, h, _, d = q.shape
                if k.shape[1] != h:
                    k, v = comfy.ops.repeat_kv_for_gqa(k, v, h, -3)
                layout = "HND"
            else:  # (B, N, H*D)
                b, _, hd = q.shape
                h = heads
                d = hd // heads
                q = q.view(b, -1, h, d)
                k = k.view(b, -1, k.shape[-1] // d, d)
                v = v.view(b, -1, v.shape[-1] // d, d)
                if k.shape[2] != h:
                    k, v = comfy.ops.repeat_kv_for_gqa(k, v, h, -2)
                layout = "NHD"
            out = sage_fn(q, k, v, tensor_layout=layout, is_causal=False, sm_scale=kwargs.get("scale", None))
        except Exception as e:
            log.warning("[SageAttentionPatch] sage kernel failed (%s); falling back to default attention", e)
            return fallback()

        if layout == "NHD":
            out = out.transpose(1, 2)  # -> (B, H, N, D)
        if skip_output_reshape:
            return out
        return out.transpose(1, 2).reshape(b, -1, h * d)
    return override


class SageAttentionPatch:
    """Replace attention with SageAttention (INT8 QK / FP8 PV kernels, requires the sageattention package)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "variant": (list(_SAGE_VARIANTS.keys()), {"default": "sageattn"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "FP32Toolkit/attention"

    def patch(self, model, variant):
        if not _HAS_SAGE:
            raise RuntimeError("sageattention not installed. pip install sageattention")
        name = _SAGE_VARIANTS.get(variant, variant)
        sage_fn = getattr(sageattention, name, None)
        if sage_fn is None:
            log.warning("[SageAttentionPatch] sageattention has no %s; using sageattn", name)
            sage_fn = sageattention.sageattn
        m = model.clone()
        chain_attention_override(m, _make_sage_override(sage_fn))
        return (m,)


NODE_CLASS_MAPPINGS = {
    "SDPABackendSelect": SDPABackendSelect,
    "FP32AttentionPrecision": FP32AttentionPrecision,
    "SageAttentionPatch": SageAttentionPatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDPABackendSelect": "SDPA Backend Select (Blackwell)",
    "FP32AttentionPrecision": "FP32 Attention Precision",
    "SageAttentionPatch": "SageAttention Patch",
}
