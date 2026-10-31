"""Selective dtype casting. Keep norms/embeddings FP32, downcast Linear/Conv weights."""
import torch
import torch.nn as nn
import comfy.model_management as mm

from ._common import clone_for_weight_edit


_NORM_TYPES = (nn.LayerNorm, nn.GroupNorm)
_EMBED_NAME_KEYS = ("embedder", "time_embed", "label_emb", "pos_embed", "patch_embed")
_FINAL_NAME_KEYS = ("final_layer", "out_proj_final", "adaln_final", "proj_out", "norm_out")


def _is_norm(mod):
    if isinstance(mod, _NORM_TYPES):
        return True
    name = type(mod).__name__.lower()
    return "rmsnorm" in name or "rms_norm" in name or "layernorm" in name or "groupnorm" in name


def _is_embed(mod, name):
    return isinstance(mod, nn.Embedding) or any(k in name for k in _EMBED_NAME_KEYS)


_BRIDGE_CLASSES = {}


def _bridge_class(base):
    cls = _BRIDGE_CLASSES.get(base)
    if cls is None:
        class DtypeBridge(base):
            _fp32tk_dtype_bridge = True

            def forward(self, x, *args, **kwargs):
                w = self.weight
                if (isinstance(x, torch.Tensor) and w is not None and x.is_floating_point()
                        and x.dtype != w.dtype and not torch.is_autocast_enabled(x.device.type)):
                    return super().forward(x.to(w.dtype), *args, **kwargs).to(x.dtype)
                return super().forward(x, *args, **kwargs)

        DtypeBridge.__name__ = base.__name__
        DtypeBridge.__qualname__ = base.__qualname__
        cls = _BRIDGE_CLASSES[base] = DtypeBridge
    return cls


def _wrap_dtype_bridge(mod):
    """Make a downcast layer usable without autocast: cast the input to the weight dtype and
    the output back to the caller's dtype. Under autocast it is a no-op (autocast handles it).
    Done by swapping in a cached subclass so deep copies of the model stay correct."""
    if not getattr(mod, "_fp32tk_dtype_bridge", False):
        mod.__class__ = _bridge_class(type(mod))


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
        m = clone_for_weight_edit(model)
        mm.load_models_gpu([m])
        diff = m.get_model_object("diffusion_model")

        counts = {"fp32": 0, "linear": 0, "conv": 0}
        for name, sub in diff.named_modules():
            lname = name.lower()
            if preserve_norms_fp32 and _is_norm(sub):
                sub.to(dtype=torch.float32)
                counts["fp32"] += 1
                continue
            if preserve_embeddings_fp32 and _is_embed(sub, lname):
                sub.to(dtype=torch.float32)
                counts["fp32"] += 1
                continue
            if preserve_final_layer_fp32 and any(k in lname for k in _FINAL_NAME_KEYS):
                sub.to(dtype=torch.float32)
                counts["fp32"] += 1
                continue

            if isinstance(sub, nn.Linear) and linear_dtype != "keep":
                sub.to(dtype=self._DTYPES[linear_dtype])
                _wrap_dtype_bridge(sub)
                counts["linear"] += 1
            elif isinstance(sub, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and conv_dtype != "keep":
                sub.to(dtype=self._DTYPES[conv_dtype])
                _wrap_dtype_bridge(sub)
                counts["conv"] += 1

        print(f"[FP32SelectiveCast] linear->{linear_dtype}: {counts['linear']}, conv->{conv_dtype}: {counts['conv']}, "
              f"kept fp32: {counts['fp32']}")
        return (m,)


NODE_CLASS_MAPPINGS = {"FP32SelectiveCast": FP32SelectiveCast}
NODE_DISPLAY_NAME_MAPPINGS = {"FP32SelectiveCast": "FP32 Selective Cast"}
