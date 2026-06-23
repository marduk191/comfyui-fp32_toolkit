"""Diagnostic / report nodes."""
import torch
import comfy.model_management as mm


class FP32ToolkitReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "run"
    CATEGORY = "FP32Toolkit/diagnostic"
    OUTPUT_NODE = True

    def run(self, model):
        lines = []
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        lines.append(f"CUDA capability: sm_{cap[0]}{cap[1]}")
        lines.append(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
        lines.append(f"PyTorch: {torch.__version__}")
        lines.append(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
        lines.append(f"TF32 cuDNN: {torch.backends.cudnn.allow_tf32}")
        lines.append(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        lines.append(f"matmul precision: {torch.get_float32_matmul_precision()}")
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
            lines.append(f"fp16 accumulation: {torch.backends.cuda.matmul.allow_fp16_accumulation}")

        try:
            from torchao import __version__ as tao_v
            lines.append(f"torchao: {tao_v}")
        except ImportError:
            lines.append("torchao: not installed")
        try:
            import sageattention
            lines.append(f"sageattention: {getattr(sageattention, '__version__', 'unknown')}")
        except ImportError:
            lines.append("sageattention: not installed")
        try:
            import flash_attn
            lines.append(f"flash_attn: {flash_attn.__version__}")
        except ImportError:
            lines.append("flash_attn: not installed")

        topts = model.model_options.get("transformer_options", {})
        lines.append(f"transformer_options: {dict(topts)}")
        lines.append(f"model_function_wrapper set: {model.model_options.get('model_function_wrapper') is not None}")

        try:
            diff = model.get_model_object("diffusion_model")
            n_params = sum(p.numel() for p in diff.parameters())
            dtypes = {}
            for p in diff.parameters():
                dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
            lines.append(f"params: {n_params/1e9:.2f}B")
            for d, n in dtypes.items():
                lines.append(f"  {d}: {n/1e9:.2f}B ({100*n/n_params:.1f}%)")
        except Exception as e:
            lines.append(f"param scan failed: {e}")

        report = "\n".join(lines)
        print("[FP32Toolkit Report]\n" + report)
        return (model, report)


NODE_CLASS_MAPPINGS = {"FP32ToolkitReport": FP32ToolkitReport}
NODE_DISPLAY_NAME_MAPPINGS = {"FP32ToolkitReport": "FP32 Toolkit Report"}
