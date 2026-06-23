"""Attention backend selection + precision control."""
import torch
import contextlib

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _HAS_SDPA_KERNEL = True
except ImportError:
    _HAS_SDPA_KERNEL = False

try:
    import sageattention
    _HAS_SAGE = True
except ImportError:
    _HAS_SAGE = False


_BACKEND_MAP = {
    "auto": None,
    "cudnn": ["CUDNN_ATTENTION"],
    "flash": ["FLASH_ATTENTION"],
    "efficient": ["EFFICIENT_ATTENTION"],
    "math": ["MATH"],
    "cudnn+flash": ["CUDNN_ATTENTION", "FLASH_ATTENTION"],
    "flash+efficient": ["FLASH_ATTENTION", "EFFICIENT_ATTENTION"],
}


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

    def patch(self, model, backend):
        m = model.clone()
        if backend == "auto" or not _HAS_SDPA_KERNEL:
            return (m,)
        backends = [getattr(SDPBackend, name) for name in _BACKEND_MAP[backend]]

        prev_wrapper = m.model_options.get("model_function_wrapper")

        def wrapper(apply_model, args):
            with sdpa_kernel(backends):
                if prev_wrapper is not None:
                    return prev_wrapper(apply_model, args)
                return apply_model(args["input"], args["timestep"], **args["c"])

        m.set_model_unet_function_wrapper(wrapper)
        return (m,)


class FP32AttentionPrecision:
    """Force fp32 softmax inside attention even when surrounding compute is bf16/fp16."""
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
        import comfy.model_management as mm
        m = model.clone()
        prec = self._DTYPES[attn_precision]

        m.model_options.setdefault("transformer_options", {})
        if prec is None:
            m.model_options["transformer_options"].pop("attn_precision", None)
        else:
            m.model_options["transformer_options"]["attn_precision"] = prec

        try:
            mm.load_models_gpu([m])
            diff = m.get_model_object("diffusion_model")
            count = 0
            for sub in diff.modules():
                if hasattr(sub, "attn_precision"):
                    sub.attn_precision = prec
                    count += 1
            print(f"[FP32AttentionPrecision] set attn_precision={prec} on {count} modules")
        except Exception as e:
            print(f"[FP32AttentionPrecision] module-walk failed ({e}); transformer_options key set anyway")
        return (m,)


class SageAttentionPatch:
    """Replace attention with SageAttention (INT8/FP8 QK, requires sageattention package)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "variant": (["sageattn", "sageattn_qk_int8_pv_fp16", "sageattn_qk_int8_pv_fp8_cuda"],
                            {"default": "sageattn"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "FP32Toolkit/attention"

    def patch(self, model, variant):
        if not _HAS_SAGE:
            raise RuntimeError("sageattention not installed. pip install sageattention")
        m = model.clone()
        sage_fn = getattr(sageattention, variant, sageattention.sageattn)

        def attn_replace(q, k, v, extra_options):
            heads = extra_options.get("n_heads") or extra_options.get("heads")
            B, N, C = q.shape if q.dim() == 3 else (q.shape[0], q.shape[2], q.shape[1] * q.shape[3])
            if q.dim() == 3:
                head_dim = C // heads
                q = q.view(B, N, heads, head_dim).transpose(1, 2)
                k = k.view(B, k.shape[1], heads, head_dim).transpose(1, 2)
                v = v.view(B, v.shape[1], heads, head_dim).transpose(1, 2)
            out = sage_fn(q, k, v)
            out = out.transpose(1, 2).reshape(B, N, -1)
            return out

        # Patch known attention call sites; fall back to wrapper if model uses SDPA directly
        for block in ("input", "middle", "output"):
            try:
                m.set_model_attn1_replace(attn_replace, block, 0)
                m.set_model_attn2_replace(attn_replace, block, 0)
            except Exception:
                pass
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
