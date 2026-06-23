"""Global PyTorch / CUDA backend toggles. Apply once at workflow start."""
import torch


class FP32GlobalSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough": ("MODEL",),
                "matmul_precision": (["highest", "high", "medium"], {"default": "high"}),
                "tf32_matmul": ("BOOLEAN", {"default": True}),
                "tf32_cudnn": ("BOOLEAN", {"default": True}),
                "cudnn_benchmark": ("BOOLEAN", {"default": True}),
                "allow_fp16_reduced_precision_reduction": ("BOOLEAN", {"default": True}),
                "allow_fp16_accumulation": ("BOOLEAN", {"default": True}),
                "allow_bf16_reduced_precision_reduction": ("BOOLEAN", {"default": True}),
                "deterministic": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "FP32Toolkit/setup"

    def apply(self, passthrough, matmul_precision, tf32_matmul, tf32_cudnn,
              cudnn_benchmark, allow_fp16_reduced_precision_reduction,
              allow_fp16_accumulation, allow_bf16_reduced_precision_reduction,
              deterministic):
        torch.set_float32_matmul_precision(matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
        torch.backends.cudnn.allow_tf32 = tf32_cudnn
        torch.backends.cudnn.benchmark = cudnn_benchmark and not deterministic
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = allow_fp16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = allow_bf16_reduced_precision_reduction
        # Blackwell-specific: fp16 accumulation in tensor cores
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
            torch.backends.cuda.matmul.allow_fp16_accumulation = allow_fp16_accumulation
        return (passthrough,)


NODE_CLASS_MAPPINGS = {"FP32GlobalSettings": FP32GlobalSettings}
NODE_DISPLAY_NAME_MAPPINGS = {"FP32GlobalSettings": "FP32 Global Settings (Blackwell)"}
