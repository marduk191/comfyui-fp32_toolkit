"""Selective dtype casting. Keep norms/embeddings FP32, downcast Linear/Conv weights."""
import torch
import torch.nn as nn
import comfy.model_management as mm


_PRESERVE_TYPES = (nn.LayerNorm, nn.GroupNorm, nn.Embedding)


def _has_rms(mod):
    name = type(mod).__name__.lower()
    return "rmsnorm" in name or "rms_norm" in name


class FP32SelectiveCast:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "linear_dtype": (["keep", "bf16", "fp16"], {"default": "bf16"}),
                "conv_dtype": (["keep", "bf16", "fp16"], {"default": "keep"}),
                "preserve_norms_fp32": ("BOOLEAN", {"default": True}),
                "preserve_embeddings_fp32": ("BOOLEAN", {"default": True}),
                "preserve_final_layer_fp32": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "FP32Toolkit/precision"

    _DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

    def patch(self, model, linear_dtype, conv_dtype, preserve_norms_fp32,
              preserve_embeddings_fp32, preserve_final_layer_fp32):
        m = model.clone()
        mm.load_models_gpu([m])
        diff = m.get_model_object("diffusion_model")

        for name, sub in diff.named_modules():
            is_norm = isinstance(sub, _PRESERVE_TYPES) or _has_rms(sub)
            is_embed = isinstance(sub, nn.Embedding)
            is_final = preserve_final_layer_fp32 and any(
                k in name.lower() for k in ("final_layer", "out_proj_final", "adaLN_final", "proj_out")
            )

            if is_norm and preserve_norms_fp32:
                sub.to(dtype=torch.float32)
                continue
            if is_embed and preserve_embeddings_fp32:
                sub.to(dtype=torch.float32)
                continue
            if is_final:
                sub.to(dtype=torch.float32)
                continue

            if isinstance(sub, nn.Linear) and linear_dtype != "keep":
                sub.to(dtype=self._DTYPES[linear_dtype])
            elif isinstance(sub, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and conv_dtype != "keep":
                sub.to(dtype=self._DTYPES[conv_dtype])

        return (m,)


NODE_CLASS_MAPPINGS = {"FP32SelectiveCast": FP32SelectiveCast}
NODE_DISPLAY_NAME_MAPPINGS = {"FP32SelectiveCast": "FP32 Selective Cast"}
