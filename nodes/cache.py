"""Step-skip caching for DiT (TeaCache-style rel-L1).

Wraps the unet function. On each step, compares current input embedding vs.
cached prev. If rel-L1 < threshold, returns cached output (skip compute).
"""
import torch


class TeaCachePatch:
    """Generic timestep-aware DiT cache. Skips full forward when input drift is small."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rel_l1_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.01}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "end_step": ("INT", {"default": 1000, "min": 0, "max": 1000}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "FP32Toolkit/cache"

    def patch(self, model, rel_l1_threshold, start_step, end_step):
        m = model.clone()

        state = {"prev_input": None, "prev_output": None, "step": 0, "skipped": 0}
        prev_wrapper = m.model_options.get("model_function_wrapper")

        def wrapper(apply_model, args):
            x = args["input"]
            t = args["timestep"]
            step = state["step"]
            state["step"] += 1

            in_window = start_step <= step < end_step
            should_skip = False

            if in_window and state["prev_input"] is not None:
                with torch.no_grad():
                    diff = (x - state["prev_input"]).abs().mean()
                    base = state["prev_input"].abs().mean().clamp(min=1e-6)
                    rel = (diff / base).item()
                if rel < rel_l1_threshold and state["prev_output"].shape == x.shape:
                    state["skipped"] += 1
                    return state["prev_output"]

            if prev_wrapper is not None:
                out = prev_wrapper(apply_model, args)
            else:
                out = apply_model(x, t, **args["c"])

            state["prev_input"] = x.detach()
            state["prev_output"] = out.detach()
            return out

        # reset state at the start of each sampling run via on_sample callback
        def reset_on_load(*a, **kw):
            state["prev_input"] = None
            state["prev_output"] = None
            state["step"] = 0
            state["skipped"] = 0

        m.set_model_unet_function_wrapper(wrapper)
        # ComfyUI calls model.cleanup() between runs; hook for state reset
        if hasattr(m, "add_callback"):
            m.add_callback("on_pre_run", reset_on_load)
        return (m,)


NODE_CLASS_MAPPINGS = {"TeaCachePatch": TeaCachePatch}
NODE_DISPLAY_NAME_MAPPINGS = {"TeaCachePatch": "TeaCache (DiT step-skip)"}
