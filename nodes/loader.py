"""FP32 diffusion model loader. Loads weights at a chosen dtype, optionally computing in another."""
import torch
import folder_paths
import comfy.sd
import comfy.ops


_DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


def _load(unet_path, model_options, compute_dtype=None, disable_dynamic=False):
    """Load + apply the compute dtype. Also used as ``cached_patcher_init`` so that
    non-dynamic re-loads (``clone(disable_dynamic=True)``) get the same settings."""
    model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options, disable_dynamic=disable_dynamic)
    if compute_dtype is not None:
        # manual_cast ops cast weights to the input dtype; the input dtype comes from manual_cast_dtype.
        model.model.manual_cast_dtype = compute_dtype
    model.cached_patcher_init = (_load, (unet_path, model_options, compute_dtype))
    return model


class LoadDiffusionModelFP32:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "load_dtype": (list(_DTYPE_MAP.keys()), {"default": "fp32"}),
                "compute_dtype": (["match_load", "fp32", "bf16", "fp16"], {"default": "match_load"}),
                "ops_mode": (["default", "manual_cast", "disable_init"], {"default": "manual_cast"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "FP32Toolkit/loaders"
    DESCRIPTION = ("Load a diffusion model keeping its weights at load_dtype. With ops_mode=manual_cast and a "
                   "different compute_dtype, weights stay in load_dtype and are cast per layer at run time.")

    def load(self, unet_name, load_dtype, compute_dtype, ops_mode):
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        if unet_path is None:
            raise FileNotFoundError(f"diffusion_model not found: {unet_name}")

        weight_dtype = _DTYPE_MAP[load_dtype]
        model_options = {"dtype": weight_dtype}

        if load_dtype.startswith("fp8"):
            model_options["fp8_optimizations"] = True

        compute = weight_dtype if compute_dtype == "match_load" else _DTYPE_MAP[compute_dtype]
        compute_override = None

        if ops_mode == "manual_cast" and compute != weight_dtype:
            model_options["custom_operations"] = comfy.ops.manual_cast
            compute_override = compute
        elif ops_mode == "disable_init":
            model_options["custom_operations"] = comfy.ops.disable_weight_init

        model = _load(unet_path, model_options, compute_override)
        # stash compute hint for downstream nodes
        model.model_options.setdefault("transformer_options", {})
        model.model_options["transformer_options"]["fp32_toolkit_compute_dtype"] = compute
        return (model,)


NODE_CLASS_MAPPINGS = {"LoadDiffusionModelFP32": LoadDiffusionModelFP32}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadDiffusionModelFP32": "Load Diffusion Model (FP32 Toolkit)"}
