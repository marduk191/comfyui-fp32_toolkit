"""torch.compile + channels_last patches for the diffusion_model submodule."""
import os
import torch
import comfy.model_management as mm

from ._common import clone_for_weight_edit, log

try:
    from comfy_api.torch_helpers import set_torch_compile_wrapper
    _HAS_COMPILE_HELPER = True
except ImportError:
    _HAS_COMPILE_HELPER = False


_INDUCTOR_ENV = {
    "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
}


def _skip_transformer_options_guards(guard_entries):
    # Same filter as core TorchCompileModel: transformer_options changes must not trigger recompiles.
    return [("transformer_options" not in entry.name) for entry in guard_entries]


def _inductor_options(mode, coordinate_descent_tuning):
    """torch.compile refuses mode= together with options=, so expand the mode into options."""
    opts = {}
    if mode and mode != "default":
        try:
            opts.update(torch._inductor.list_mode_options(mode))
        except Exception as e:
            log.warning("[BlackwellTorchCompile] could not expand mode %s (%s)", mode, e)
    opts["coordinate_descent_tuning"] = bool(coordinate_descent_tuning)
    return opts


class BlackwellTorchCompile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (["inductor", "cudagraphs"], {"default": "inductor"}),
                "mode": (["default", "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune"],
                         {"default": "max-autotune-no-cudagraphs"}),
                "fullgraph": ("BOOLEAN", {"default": False}),
                "dynamic": ("BOOLEAN", {"default": False}),
                "cutlass_backend": ("BOOLEAN", {"default": False}),
                "coordinate_descent_tuning": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "FP32Toolkit/compile"

    def patch(self, model, backend, mode, fullgraph, dynamic, cutlass_backend,
              coordinate_descent_tuning):
        for k, v in _INDUCTOR_ENV.items():
            os.environ.setdefault(k, v)

        options = {}
        if backend == "inductor":
            options = _inductor_options(mode, coordinate_descent_tuning)
            if cutlass_backend:
                options["max_autotune_gemm_backends"] = "ATEN,TRITON,CUTLASS"
        options["guard_filter_fn"] = _skip_transformer_options_guards

        m = clone_for_weight_edit(model)

        if _HAS_COMPILE_HELPER:
            set_torch_compile_wrapper(model=m, backend=backend, options=options, fullgraph=fullgraph, dynamic=dynamic)
        else:
            diff = m.get_model_object("diffusion_model")
            compiled = torch.compile(diff, backend=backend, options=options, fullgraph=fullgraph, dynamic=dynamic)
            m.add_object_patch("diffusion_model", compiled)
        return (m,)


class ChannelsLastPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "memory_format": (["channels_last", "channels_last_3d"], {"default": "channels_last"})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "FP32Toolkit/compile"

    def patch(self, model, memory_format):
        fmt = torch.channels_last if memory_format == "channels_last" else torch.channels_last_3d
        ndim = 4 if fmt is torch.channels_last else 5
        m = clone_for_weight_edit(model)
        mm.load_models_gpu([m])
        diff = m.get_model_object("diffusion_model")
        n = 0
        # Only conv weights of the right rank can take the format; Module.to(memory_format=) raises
        # on the first mismatching (e.g. Linear) parameter and leaves the rest untouched.
        for sub in diff.modules():
            w = getattr(sub, "weight", None)
            if isinstance(sub, (torch.nn.Conv2d, torch.nn.Conv3d)) and isinstance(w, torch.Tensor) and w.dim() == ndim:
                sub.weight.data = w.data.contiguous(memory_format=fmt)
                n += 1
        print(f"[ChannelsLastPatch] {memory_format} applied to {n} conv layers")
        return (m,)


NODE_CLASS_MAPPINGS = {
    "BlackwellTorchCompile": BlackwellTorchCompile,
    "ChannelsLastPatch": ChannelsLastPatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BlackwellTorchCompile": "Blackwell torch.compile",
    "ChannelsLastPatch": "Channels-Last Memory Format",
}
