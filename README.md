# ComfyUI FP32 Toolkit (Blackwell)

Custom node pack to accelerate **FP32 diffusion model weights** on **RTX 5090 (Blackwell, sm_120)**. Tuned for [Tongyi Z-Image-Turbo FP32](https://huggingface.co/tsqn/Z-Image-Turbo_fp32-fp16-bf16_full_and_ema-only).

Strategy: keep master weights FP32 on disk, run aggressive Blackwell-native acceleration (TF32, bf16 autocast, FP8/FP4 quantization, cuDNN/Flash SDPA, `torch.compile` max-autotune, optional TeaCache step-skipping). Sensitive layers (norms, embeddings, final projection) stay FP32 for quality.

## Install

From the [Comfy Registry](https://registry.comfy.org) (ComfyUI-Manager or comfy-cli):
```
comfy node install comfyui-fp32-toolkit
```

Or manually:
```
cd ComfyUI/custom_nodes
git clone https://github.com/marduk191/comfyui-fp32_toolkit comfyui-fp32_toolkit
```

The pack has no hard Python dependencies; the optional accelerators below are imported only when their node runs.

Tested against ComfyUI 0.43.0 (October 2026), torch 2.15 and torchao 0.18.

PyTorch must be CUDA 13 (cu130) with sm_120 support — confirm with:
```
python -c "import torch; print(torch.version.cuda, torch.cuda.get_device_capability())"
# expect: 13.x  (12, 0)
```
Install nightly if needed:
```
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

Optional accelerators (must match cu13 ABI):
- `torchao>=0.13` — FP8 / MX / NVFP4 / INT8 quant (older releases still work through fallbacks)
- `sageattention>=2.2` — INT8/FP8 attention (build from source on cu13 if no wheel)
- `flash-attn>=3.0` — Blackwell wheels merged Q1-2026; build with `MAX_JOBS=4 pip install flash-attn --no-build-isolation` if needed

## Model

Download:
```
https://huggingface.co/tsqn/Z-Image-Turbo_fp32-fp16-bf16_full_and_ema-only/resolve/main/diffusion_pytorch_model-ema-only-fp32.safetensors
```
Place in `ComfyUI/models/diffusion_models/`.

## Nodes

### Setup
- **FP32 Global Settings (Blackwell)** — TF32, cuDNN benchmark, fp16 accumulation, matmul precision. Run once.

### Loaders
- **Load Diffusion Model (FP32 Toolkit)** — loads UNet/DiT at `load_dtype`. With `ops_mode=manual_cast` and a different `compute_dtype`, weights stay at `load_dtype` and are cast per layer to `compute_dtype` at run time.

### Precision
- **FP32 Autocast Wrapper** — wraps forward in `torch.autocast(bf16/fp16)` on the model's device. FP32 master, low-precision compute.
- **FP32 Selective Cast** — keeps norms/embedders/final layer FP32, downcasts Linear/Conv weights. Works with or without autocast (downcast layers cast their input/output when autocast is off).

### Quantize (Blackwell-native)
- **Blackwell FP8 Quantize** — torchao dynamic FP8 e4m3, per-row or per-tensor scaling. Needs sm_89+ (Ada/Hopper/Blackwell).
- **Blackwell MX Quantize** — microscale formats with block-wise scales (Blackwell-native MX tensor cores). Quantized layers are cast to bf16 first (torchao requirement):
    - `mxfp8_e4m3` / `mxfp8_e5m2` — block-32 FP8, better dynamic range than per-tensor FP8
    - `mxfp6_e2m3` / `mxfp6_e3m2` — block-32 FP6
    - `mxfp4` — block-32 FP4 (sm_100+)
    - `nvfp4` — block-16 FP4 with FP8 scales, best 4-bit quality (sm_100+). `activation_dtype=match_weight` quantizes activations too; `bf16`/`fp16` is weight-only.
    - On current torchao the MX formats always quantize activations to the weight format, so `activation_dtype` only matters for `nvfp4`.
- Quantize nodes skip layers that are already quantized, so re-running them never double-quantizes. To switch format, reload the model.
- **INT8 Weight-Only Quantize** — fallback for non-Blackwell.

### Attention
- **SDPA Backend Select (Blackwell)** — sets the PyTorch SDPA kernel priority (cuDNN / Flash / Efficient). The chosen backends go first and MATH is kept as a last-resort fallback, so an unsupported shape degrades instead of erroring.
- **FP32 Attention Precision** — sets the attention upcast precision. It applies to ComfyUI's basic / split / sub-quadratic attention; PyTorch SDPA kernels already accumulate softmax in fp32.
- **SageAttention Patch** — replaces attention with SageAttention kernels (`sageattn`, `sageattn_qk_int8_pv_fp16_cuda/_triton`, `sageattn_qk_int8_pv_fp8_cuda`) through ComfyUI's attention override, so it works on DiTs such as Z-Image. Falls back to default attention for masks or fp32 inputs.

### Compile
- **Blackwell torch.compile** — `inductor` with `max-autotune-no-cudagraphs` (DiT default).
- **Channels-Last Memory Format** — for conv-heavy submodels.

### Cache
- **TeaCache (DiT step-skip)** — rel-L1 input drift threshold; reuses prev output. ~1.5–2x extra speedup. State is kept per cond/uncond batch and reset at the start of every sampling run.

### Memory
- **Soft Empty Cache** — gc + `mm.soft_empty_cache`.
- **Async Streams Config** — sets `mm.NUM_STREAMS` for async H2D offload pipelines.

### Preset
- **Z-Image-Turbo Optimizer (Blackwell)** — one-shot stack that bundles the toolkit's individual nodes into a single drop-in. Takes a `MODEL`, returns a fully-patched `MODEL`. Three widgets:

  **`preset`** — quality/speed tradeoff. Each preset toggles selective cast + autocast(bf16) + quant + compile + attn-precision:

  | preset | selective cast | quant | compile | attn precision | best for |
  |---|---|---|---|---|---|
  | `safe` | bf16 Linear, FP32 norms/embeds/final | none | off | fp32 softmax | quality reference, debugging |
  | `balanced` | same | FP8 e4m3 per-row | on | fp32 softmax | default — Hopper+ compatible |
  | `aggressive` | same | **MXFP8 e4m3** (block-32 microscale) | on | bf16 softmax | Blackwell native, ~1.7x over balanced |
  | `extreme` | same | **NVFP4** (block-16 + FP8 scales) | on | bf16 softmax | max speed, halves VRAM, slight quality drift |

  All presets keep RMSNorm / LayerNorm / embeddings / final projection in FP32 to preserve quality.

  **`compile_mode`** — passed to `torch.compile` when the preset enables compile:

  | mode | first-run autotune | speedup | notes |
  |---|---|---|---|
  | `default` | fast | ~1.2x | minimal autotune, fewest issues |
  | `reduce-overhead` | medium | ~1.4x | uses CUDA graphs (static shapes only) |
  | `max-autotune-no-cudagraphs` | slow | ~1.7x | **default — best for DiT** without CUDA-graph fragility |
  | `max-autotune` | very slow | ~1.9x | full autotune + CUDA graphs (fixed shape only) |

  Static shapes required for `max-autotune*`. Changing resolution invalidates the compile cache. First run after compile is slow; subsequent runs use `TORCHINDUCTOR_FX_GRAPH_CACHE`.

  **`sdpa_backend`** — selects `torch.nn.attention.sdpa_kernel` priority list:

  | backend | when to pick |
  |---|---|
  | `auto` | let PyTorch decide (no override) |
  | `cudnn+flash` | **default** — cuDNN attention preferred (Blackwell-tuned), Flash as fallback |
  | `cudnn` | force cuDNN only |
  | `flash` | force FlashAttention 2/3 (needs `flash-attn` installed) |

  On RTX 5090, cuDNN attention typically wins for Z-Image's MMDiT shapes; benchmark both if unsure.

  **Suggested starting points:**
  - First run / quality validation → `safe` + `default` + `cudnn+flash`
  - Daily driver → `balanced` + `max-autotune-no-cudagraphs` + `cudnn+flash`
  - Speed runs (quality drift acceptable) → `aggressive` or `extreme` with same compile/sdpa

### Save
- **Save Diffusion Model (FP32 Toolkit)** — writes the modified `diffusion_model` state_dict to `.safetensors`. Options:
    - `save_dtype`: keep / fp32 / bf16 / fp16 / fp8_e4m3fn / fp8_e5m2 — final cast on save.
    - `quantized_handling`: `dequantize` (torchao subclass → plain bf16/fp32, portable), `as_plain_tensor` (try to keep packed bits — risky, format-dependent), `error`.
    - `save_to`: `output/` (default) or `models/diffusion_models/` (overwrite-safe, auto-counter).
    - Embeds metadata: cuda_capability, save_dtype, quantized layer count.

### Diagnostic
- **FP32 Toolkit Report** — prints CUDA caps, dtypes, param distribution, active patches.

## Recommended stack for Z-Image-Turbo on 5090

```
LoadDiffusionModelFP32 (load_dtype=fp32, ops_mode=manual_cast)
  → FP32GlobalSettings (defaults)
  → ZImageTurboOptimizer (preset=balanced or aggressive, compile_mode=max-autotune-no-cudagraphs)
  → KSampler
```

Expected: **~3–4x** over naive FP32 baseline with `balanced`, **~5–8x** with `aggressive` (FP8) or `extreme` (NVFP4) including TeaCache. Quality stays close to FP32 because norms/embeds/final layer are preserved.

## Example workflows (`example_workflows/`)

| File | Purpose |
|---|---|
| `z_image_turbo_fp32_blackwell.json` | Inference only — FP32 load → optimizer (balanced) → KSampler |
| `z_image_turbo_inference_and_save.json` | Inference + save modified weights (bf16 dequantized) in one pass |
| `z_image_turbo_convert_fp16.json` | Offline convert: FP32 → selective-cast → save as FP16 (no inference) |
| `z_image_turbo_convert_mxfp8.json` | Offline convert: FP32 → bf16 selective → MXFP8 quant → dequantize+save bf16 |

The convert workflows write under `models/diffusion_models/` so the result is immediately reloadable via stock `UNETLoader`.

> Convert workflows reference `ShowText|pysssss` (cosmetic). Replace with any STRING display node or just remove if pysssss isn't installed.

## Notes

- **First run after `torch.compile` is slow** (autotune). Subsequent runs use cached graphs. Set `TORCHINDUCTOR_FX_GRAPH_CACHE=1`.
- **FP4 requires Blackwell (sm_100+).** The node refuses on Ada/Hopper, and every quant node gives a clear error when no suitable CUDA GPU is present.
- **DynamicVRAM:** current ComfyUI streams weights on demand (DynamicVRAM). Nodes that edit weights (selective cast, quantize, compile, channels-last) take a regular, non-dynamic copy first, the same way core `TorchCompileModel` does.
- **Static shapes only** for `max-autotune` — change resolution invalidates cache.
- TeaCache trades quality for speed; lower `rel_l1_threshold` = more skips. Start at 0.4.
- Z-Image-Turbo native cfg is 1.0; fewer steps (≈8) so step-skip benefit is smaller.

## Changelog

### 0.2.0
- Updated for ComfyUI 0.43 / DynamicVRAM: weight-editing nodes use `clone(disable_dynamic=True)`.
- torchao 0.13+ API: `MXDynamicActivationMXWeightConfig`, `NVFP4DynamicActivationNVFP4WeightConfig` / `NVFP4WeightOnlyConfig`, `Int8WeightOnlyConfig` (old names kept as fallbacks). The `aggressive` and `extreme` presets and the INT8 node failed on current torchao before this release.
- Selective Cast: fixed the `mat1 and mat2 must have the same dtype` crash without autocast; Z-Image's `*_embedder` layers are now kept FP32; `preserve_embeddings_fp32` and `preserve_norms_fp32` act independently.
- Loader: `compute_dtype` is now applied (it was stored but never used).
- SDPA Backend Select: now takes effect. ComfyUI's own per-call `sdpa_kernel` priority used to override it.
- Attention Precision and SageAttention now go through `optimized_attention_override`, so they apply to DiTs (the old UNet `attn1/attn2` block patches never reached Z-Image).
- TeaCache: cache keyed per cond/uncond, so a cond output is no longer returned for uncond; steps are counted per timestep.
- torch.compile: core-style guard filter (no recompiles when `transformer_options` change); `mode` is expanded into inductor options.
- Autocast follows the model's device type, not hard-coded `cuda`.
- Added Comfy Registry publish workflow. Removed the committed `__pycache__` and the unused `WEB_DIRECTORY`.

## License

Apache-2.0.
