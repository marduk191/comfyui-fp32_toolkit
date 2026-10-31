"""Wrap UNet/DiT forward in torch.autocast. FP32 master weights, lower-precision compute."""
import torch

from ._common import chain_unet_wrapper, model_device_type


class FP32AutocastWrapper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "autocast_dtype": (["bf16", "fp16", "off"], {"default": "bf16"}),
                "cache_enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "FP32Toolkit/precision"

    _DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

    def patch(self, model, autocast_dtype, cache_enabled):
        m = model.clone()
        if autocast_dtype == "off":
            return (m,)
        ac_dtype = self._DTYPES[autocast_dtype]
        device_type = model_device_type(m)  # "cuda", "xpu", "cpu", ...

        def factory(call_next):
            def wrapper(apply_model, args):
                with torch.autocast(device_type, dtype=ac_dtype, cache_enabled=cache_enabled):
                    return call_next(apply_model, args)
            return wrapper

        chain_unet_wrapper(m, factory)
        return (m,)


NODE_CLASS_MAPPINGS = {"FP32AutocastWrapper": FP32AutocastWrapper}
NODE_DISPLAY_NAME_MAPPINGS = {"FP32AutocastWrapper": "FP32 Autocast Wrapper"}
