"""torch.compile + channels_last patches for the diffusion_model submodule."""
import os
import torch
import comfy.model_management as mm

try:
    from comfy_api.torch_helpers import set_torch_compile_wrapper
    _HAS_COMPILE_HELPER = True
except ImportError:
    _HAS_COMPILE_HELPER = False


_INDUCTOR_ENV = {
    "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
}


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

        if cutlass_backend:
            os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "ATEN,TRITON,CUTLASS"
        try:
            torch._inductor.config.coordinate_descent_tuning = coordinate_descent_tuning
        except Exception:
            pass

        try:
            m = model.clone(disable_dynamic=True)
        except TypeError:
            m = model.clone()

        kwargs = {"backend": backend, "fullgraph": fullgraph, "dynamic": dynamic}
        if backend == "inductor":
            kwargs["mode"] = mode

        if _HAS_COMPILE_HELPER:
            set_torch_compile_wrapper(model=m, backend=backend, mode=mode if backend == "inductor" else None,
                                       fullgraph=fullgraph, dynamic=dynamic)
        else:
            diff = m.get_model_object("diffusion_model")
            compiled = torch.compile(diff, **kwargs)
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
        m = model.clone()
        mm.load_models_gpu([m])
        diff = m.get_model_object("diffusion_model")
        try:
            diff.to(memory_format=fmt)
        except RuntimeError:
            pass
        return (m,)


NODE_CLASS_MAPPINGS = {
    "BlackwellTorchCompile": BlackwellTorchCompile,
    "ChannelsLastPatch": ChannelsLastPatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BlackwellTorchCompile": "Blackwell torch.compile",
    "ChannelsLastPatch": "Channels-Last Memory Format",
}
