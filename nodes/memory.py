"""Memory / offload helpers."""
import gc
import torch
import comfy.model_management as mm


class SoftEmptyCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"passthrough": ("MODEL",), "force": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = "FP32Toolkit/memory"

    def run(self, passthrough, force):
        gc.collect()
        mm.soft_empty_cache(force=force)
        return (passthrough,)


class AsyncStreamsConfig:
    """Configure ComfyUI async H2D offload streams. 0 disables async streams."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "passthrough": ("MODEL",),
            "num_streams": ("INT", {"default": 2, "min": 0, "max": 8}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = "FP32Toolkit/memory"

    def run(self, passthrough, num_streams):
        if hasattr(mm, "NUM_STREAMS"):
            mm.NUM_STREAMS = num_streams
            print(f"[AsyncStreamsConfig] mm.NUM_STREAMS = {num_streams}")
        else:
            print("[AsyncStreamsConfig] mm.NUM_STREAMS not present in this ComfyUI build")
        return (passthrough,)


NODE_CLASS_MAPPINGS = {
    "SoftEmptyCache": SoftEmptyCache,
    "AsyncStreamsConfig": AsyncStreamsConfig,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SoftEmptyCache": "Soft Empty Cache",
    "AsyncStreamsConfig": "Async Streams Config",
}
