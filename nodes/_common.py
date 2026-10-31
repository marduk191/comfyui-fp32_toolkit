"""Shared helpers for the FP32 Toolkit nodes."""
import logging

import torch

__version__ = "0.2.0"

log = logging.getLogger("FP32Toolkit")


def clone_for_weight_edit(model):
    """Clone a ModelPatcher so its module weights can be edited in place.

    Current ComfyUI loads models with DynamicVRAM (ModelPatcherDynamic) on CUDA, where
    weights are streamed on demand and must not be mutated. ``disable_dynamic=True``
    returns a regular ModelPatcher (reloading the model from disk if needed), which is
    what core nodes such as TorchCompileModel do before touching the model.
    """
    try:
        return model.clone(disable_dynamic=True)
    except TypeError:  # older ComfyUI without the argument
        return model.clone()
    except RuntimeError as e:  # dynamic patcher without cached_patcher_init
        log.warning("[FP32Toolkit] could not get a non-dynamic clone (%s); editing the shared model", e)
        return model.clone()


def chain_unet_wrapper(model, wrapper_factory):
    """Install a ``model_function_wrapper`` that composes with any existing one.

    ``wrapper_factory(call_next)`` must return ``wrapper(apply_model, args)``; ``call_next``
    runs the previous wrapper (or the model itself).
    """
    prev = model.model_options.get("model_function_wrapper")

    def call_next(apply_model, args):
        if prev is not None:
            return prev(apply_model, args)
        return apply_model(args["input"], args["timestep"], **args["c"])

    model.set_model_unet_function_wrapper(wrapper_factory(call_next))


def chain_attention_override(model, override):
    """Install ``transformer_options['optimized_attention_override']`` composing with any existing one.

    ``override(call_next, *args, **kwargs)``; ``call_next(*args, **kwargs)`` runs the previous
    override (or ComfyUI's selected attention function).
    """
    to = model.model_options.setdefault("transformer_options", {})
    prev = to.get("optimized_attention_override")

    def composed(func, *args, **kwargs):
        if prev is not None:
            def call_next(*a, **k):
                return prev(func, *a, **k)
        else:
            call_next = func
        return override(call_next, *args, **kwargs)

    to["optimized_attention_override"] = composed


def model_device_type(model):
    dev = getattr(model, "load_device", None)
    if dev is None:
        import comfy.model_management as mm
        dev = mm.get_torch_device()
    return torch.device(dev).type


def cuda_capability():
    """(major, minor) of the current CUDA device, or None when CUDA is unavailable."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()
    except Exception:
        pass
    return None


def is_plain_tensor(t):
    return type(t) in (torch.Tensor, torch.nn.Parameter)
