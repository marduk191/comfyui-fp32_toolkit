"""Save diffusion model state_dict to .safetensors after toolkit modifications.

Handles:
  - selective-cast modules (fp32/bf16/fp16/fp8 weights via plain state_dict)
  - torch.compile unwrap (._orig_mod)
  - torchao quantized tensor subclasses (optional dequantize-on-save)
  - dtype override on save

Notes:
  - autocast / SDPA / TeaCache patches do not modify weights; they round-trip safely.
  - True quantized save (preserving FP8/FP4 + scales as plain tensors) is format-specific.
    For portability we offer "dequantize" mode → bf16/fp16/fp32 plain weights.
"""
import os
import json
import torch
import folder_paths
import comfy.model_management as mm

from safetensors.torch import save_file

from ._common import __version__


_DTYPE_MAP = {
    "keep":      None,
    "fp32":      torch.float32,
    "bf16":      torch.bfloat16,
    "fp16":      torch.float16,
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e5m2":  torch.float8_e5m2,
}


def _maybe_dequantize(t):
    """Return plain torch.Tensor. Calls .dequantize() on quantized subclasses."""
    if hasattr(t, "dequantize") and callable(t.dequantize):
        try:
            return t.dequantize()
        except Exception:
            pass
    # torchao newer API
    if hasattr(t, "to_original_precision") and callable(t.to_original_precision):
        try:
            return t.to_original_precision()
        except Exception:
            pass
    return t


def _to_plain(t):
    """Coerce tensor subclass to a plain torch.Tensor view (no dequant)."""
    if type(t) is torch.Tensor:
        return t
    try:
        return t.detach().clone().as_subclass(torch.Tensor)
    except Exception:
        return _maybe_dequantize(t)


class SaveDiffusionModel:
    """Save diffusion_model weights to safetensors. Output goes to ComfyUI/output/<prefix>_NNNNN_.safetensors."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "zimage_fp32_toolkit"}),
                "save_dtype": (list(_DTYPE_MAP.keys()), {"default": "keep"}),
                "quantized_handling": (["dequantize", "as_plain_tensor", "error"], {"default": "dequantize"}),
                "save_to": (["output", "diffusion_models"], {"default": "output"}),
                "include_metadata": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    CATEGORY = "FP32Toolkit/save"
    OUTPUT_NODE = True

    def save(self, model, filename_prefix, save_dtype, quantized_handling, save_to, include_metadata):
        if getattr(model, "is_dynamic", lambda: False)():
            # DynamicVRAM patchers stream weights on demand; get a regular patcher holding real tensors.
            model = model.clone(disable_dynamic=True)
        mm.load_models_gpu([model])  # applies LoRA / weight patches
        diff = model.get_model_object("diffusion_model")
        # unwrap torch.compile
        diff = getattr(diff, "_orig_mod", diff)

        target_dtype = _DTYPE_MAP[save_dtype]
        sd = diff.state_dict()

        out_sd = {}
        n_quant = 0
        n_total = 0
        for k, v in sd.items():
            n_total += 1
            is_subclass = type(v) is not torch.Tensor

            if is_subclass:
                n_quant += 1
                if quantized_handling == "error":
                    raise RuntimeError(
                        f"Tensor '{k}' is a quantized subclass ({type(v).__name__}). "
                        f"Set quantized_handling=dequantize or as_plain_tensor."
                    )
                elif quantized_handling == "dequantize":
                    v = _maybe_dequantize(v)
                    if type(v) is not torch.Tensor:
                        # last resort: dense materialize
                        try:
                            v = v.to(torch.bfloat16).to(torch.float32).contiguous()
                        except Exception as e:
                            raise RuntimeError(f"Could not dequantize {k}: {e}")
                else:  # as_plain_tensor
                    v = _to_plain(v)

            if target_dtype is not None and v.is_floating_point():
                if v.dtype != target_dtype:
                    v = v.to(target_dtype)

            out_sd[k] = v.detach().contiguous().cpu()

        # Resolve output path
        if save_to == "diffusion_models":
            base = folder_paths.get_folder_paths("diffusion_models")[0]
            os.makedirs(base, exist_ok=True)
            counter = 1
            while True:
                fname = f"{filename_prefix}_{counter:05d}_.safetensors"
                full = os.path.join(base, fname)
                if not os.path.exists(full):
                    break
                counter += 1
            out_path = full
        else:
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory()
            )
            out_path = os.path.join(full_output_folder, f"{filename}_{counter:05d}_.safetensors")

        metadata = None
        if include_metadata:
            metadata = {
                "format": "pt",
                "fp32_toolkit_version": __version__,
                "save_dtype": save_dtype,
                "quantized_layers": str(n_quant),
                "total_layers": str(n_total),
                "quantized_handling": quantized_handling,
            }
            try:
                cap = torch.cuda.get_device_capability()
                metadata["cuda_capability"] = f"sm_{cap[0]}{cap[1]}"
            except Exception:
                pass

        save_file(out_sd, out_path, metadata=metadata)
        size_mb = os.path.getsize(out_path) / 1e6
        msg = (f"[SaveDiffusionModel] wrote {n_total} tensors ({n_quant} quantized, {quantized_handling}) "
               f"-> {out_path} ({size_mb:.1f} MB)")
        print(msg)
        return (out_path,)


NODE_CLASS_MAPPINGS = {"SaveDiffusionModel": SaveDiffusionModel}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveDiffusionModel": "Save Diffusion Model (FP32 Toolkit)"}
