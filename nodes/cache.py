"""Step-skip caching for DiT (TeaCache-style rel-L1).

Wraps the unet function. On each sampling step, compares the current input vs. the cached
previous input for the same cond/uncond batch. If rel-L1 < threshold, returns the cached output
(skips compute).
"""
import torch

from ._common import chain_unet_wrapper


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
        state = {}

        def reset(*_args, **_kwargs):
            if state.get("skipped"):
                print(f"[TeaCache] skipped {state['skipped']}/{state['calls']} model calls")
            state.clear()
            state.update({"entries": {}, "step": -1, "last_t": None, "skipped": 0, "calls": 0})

        reset()

        def factory(call_next):
            def wrapper(apply_model, args):
                x = args["input"]
                t = args["timestep"]
                t_val = float(t.flatten()[0])
                if state["last_t"] != t_val:  # new sampling step (cond and uncond share a timestep)
                    state["step"] += 1
                    state["last_t"] = t_val
                state["calls"] += 1
                # Cache per cond/uncond batch layout so a cond output is never returned for uncond.
                key = (tuple(args.get("cond_or_uncond", ())), tuple(x.shape))
                entry = state["entries"].get(key)

                if entry is not None and start_step <= state["step"] < end_step:
                    with torch.no_grad():
                        prev_x = entry["input"]
                        rel = ((x - prev_x).abs().mean() / prev_x.abs().mean().clamp(min=1e-6)).item()
                    if rel < rel_l1_threshold:
                        state["skipped"] += 1
                        return entry["output"]

                out = call_next(apply_model, args)
                state["entries"][key] = {"input": x.detach(), "output": out.detach()}
                return out
            return wrapper

        chain_unet_wrapper(m, factory)
        # pre_run fires at the start of every sampling run: drop state from the previous run.
        if hasattr(m, "add_callback"):
            m.add_callback("on_pre_run", reset)
        return (m,)


NODE_CLASS_MAPPINGS = {"TeaCachePatch": TeaCachePatch}
NODE_DISPLAY_NAME_MAPPINGS = {"TeaCachePatch": "TeaCache (DiT step-skip)"}
