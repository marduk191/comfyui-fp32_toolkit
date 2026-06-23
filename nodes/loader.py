"""FP32 diffusion model loader. Forces dtype=fp32 with optional manual_cast ops."""
import torch
import folder_paths
import comfy.sd
import comfy.ops
import comfy.model_management as mm


class LoadDiffusionModelFP32:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "load_dtype": (["fp32", "bf16", "fp16", "fp8_e4m3fn", "fp8_e5m2"], {"default": "fp32"}),
                "compute_dtype": (["match_load", "fp32", "bf16", "fp16"], {"default": "match_load"}),
                "ops_mode": (["default", "manual_cast", "disable_init"], {"default": "manual_cast"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "FP32Toolkit/loaders"

    _DTYPE_MAP = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }

    def load(self, unet_name, load_dtype, compute_dtype, ops_mode):
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        if unet_path is None:
            raise FileNotFoundError(f"diffusion_model not found: {unet_name}")

        weight_dtype = self._DTYPE_MAP[load_dtype]
        model_options = {"dtype": weight_dtype}

        if load_dtype.startswith("fp8"):
            model_options["fp8_optimizations"] = True

        compute = weight_dtype if compute_dtype == "match_load" else self._DTYPE_MAP[compute_dtype]

        if ops_mode == "manual_cast" and compute != weight_dtype:
            class _ManualCastOps(comfy.ops.manual_cast):
                pass
            model_options["custom_operations"] = _ManualCastOps
        elif ops_mode == "disable_init":
            model_options["custom_operations"] = comfy.ops.disable_weight_init

        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        # stash compute hint for downstream nodes
        model.model_options.setdefault("transformer_options", {})
        model.model_options["transformer_options"]["fp32_toolkit_compute_dtype"] = compute
        return (model,)


NODE_CLASS_MAPPINGS = {"LoadDiffusionModelFP32": LoadDiffusionModelFP32}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadDiffusionModelFP32": "Load Diffusion Model (FP32 Toolkit)"}
