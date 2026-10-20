"""Scaled FP8 weights with optional native FP8 GEMMs and bounded CPU offloading.

Only Linear weights are quantized. Biases and other model parameters retain BF16;
one FP32 scalar per Linear records its quantization scale.
"""
import time

import torch
import torch.nn.functional as F

from diffsynth.vram_management.layers import AutoTorchModule

FP8_DTYPE = torch.float8_e4m3fn
_native_kernels = None


def _native_quantization_kernels():
    # Storage-only inference never needs to import or install Triton.
    global _native_kernels, tl
    if _native_kernels is not None:
        return _native_kernels
    try:
        import triton
        import triton.language as tl
    except ImportError as error:
        raise RuntimeError("Native FP8 activation quantization requires Triton (included with Linux CUDA PyTorch).") from error

    @triton.jit(do_not_specialize=["count"])
    def _activation_amax_blocks(inputs, partials, count, BLOCK: tl.constexpr):
        block = tl.program_id(0)
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(inputs + offsets, offsets < count, other=0).to(tl.float32)
        maximum = tl.max(tl.abs(values), axis=0)
        tl.store(partials + block, maximum)

    @triton.jit(do_not_specialize=["count"])
    def _activation_quantize(inputs, output, scale, count, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(inputs + offsets, offsets < count, other=0).to(tl.float32)
        values = values / tl.load(scale)
        values = tl.minimum(tl.maximum(values, -448.0), 448.0)
        tl.store(output + offsets, values, offsets < count)
    _native_kernels = triton, _activation_amax_blocks, _activation_quantize
    return _native_kernels


def validate_native_fp8(device="cuda"):
    """Probe the actual CUDA FP8 GEMM; fail before loading a large checkpoint."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Native FP8 computation requires a CUDA device.")
    if not torch.cuda.is_available() or not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("Native FP8 requires CUDA and torch._scaled_mm.")
    capability = torch.cuda.get_device_capability(device)
    if capability < (8, 9):
        raise RuntimeError(f"Native FP8 requires SM89 or newer; got SM{capability[0]}{capability[1]}.")
    _native_quantization_kernels()
    try:
        with torch.inference_mode(), torch.cuda.device(device):
            inputs = torch.ones((16, 16), device=device, dtype=torch.bfloat16)
            matrix, input_scale = _quantize_native_input(inputs)
            weight = torch.ones((16, 16), device=device, dtype=FP8_DTYPE)
            scale = torch.ones((), device=device, dtype=torch.float32)
            bias = torch.zeros(16, device=device, dtype=torch.bfloat16)
            output = torch._scaled_mm(
                matrix, weight.t(), scale_a=input_scale, scale_b=scale,
                bias=bias, out_dtype=torch.bfloat16, use_fast_accum=False,
            )
            torch.cuda.synchronize(device)
            if output.dtype != torch.bfloat16 or not torch.all(output == 16).item():
                raise RuntimeError("FP8 GEMM returned an invalid probe result.")
    except (RuntimeError, NotImplementedError) as error:
        raise RuntimeError(f"Native FP8 GEMM is unavailable on {device}: {error}") from error
    return {
        "available": True, "device": str(device),
        "gpu": torch.cuda.get_device_name(device), "capability": list(capability),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "operation": "torch._scaled_mm", "input_dtype": "float8_e4m3fn",
        "output_dtype": "bfloat16", "activation_scaling": "dynamic_per_tensor",
        "input_quantization": "triton_fused",
        "use_fast_accum": False, "alignment": "K and N divisible by 16; arbitrary M",
    }


def _new_compute_stats():
    return {"native_calls": 0, "bf16_calls": 0, "fallback_calls": 0, "fallback_reasons": {}}


def fp8_compute_stats(model, reset=False):
    """Count executed GEMMs; explicit BF16 fallbacks also count as BF16 calls."""
    total = dict(modules=0, native_modules=0, bf16_modules=0, **_new_compute_stats())
    for module in model.modules():
        if not isinstance(module, (ScaledFP8Linear, AutoWrappedScaledFP8Linear)):
            continue
        total["modules"] += 1
        total["native_modules" if module.compute_backend == "native" else "bf16_modules"] += 1
        stats = module._compute_stats
        for key in ("native_calls", "bf16_calls", "fallback_calls"):
            total[key] += stats[key]
        for reason, count in stats["fallback_reasons"].items():
            total["fallback_reasons"][reason] = total["fallback_reasons"].get(reason, 0) + count
        if reset:
            module._compute_stats = _new_compute_stats()
    total["use_fast_accum"] = False if total["native_modules"] else None
    total["input_quantization"] = "triton_fused" if total["native_modules"] else None
    return total


def reset_fp8_compute_stats(model):
    """Exclude warmup calls from the counters of the next measured inference."""
    for module in model.modules():
        if isinstance(module, (ScaledFP8Linear, AutoWrappedScaledFP8Linear)):
            module._compute_stats = _new_compute_stats()


def quantize_linear_weight(weight):
    """Return E4M3 weights and an FP32 scale, without modifying the source tensor."""
    weight = weight.detach()
    if not weight.is_floating_point() or not torch.isfinite(weight).all():
        raise ValueError("Linear weights must contain finite floating-point values.")
    maximum = weight.abs().amax().float()
    scale = maximum / torch.finfo(FP8_DTYPE).max if maximum.item() else torch.ones_like(maximum)
    quantized = (weight.float() / scale).clamp(-448, 448).to(FP8_DTYPE)
    return quantized.contiguous(), scale


def _linear(x, weight, bias, scale):
    # Transfer FP8 bytes first; casting before the transfer would double CPU traffic.
    quantized = weight.to(device=x.device)
    dequantized = quantized.to(dtype=x.dtype)
    dequantized.mul_(scale.to(device=x.device))
    bias = None if bias is None else bias.to(device=x.device, dtype=x.dtype)
    return F.linear(x, dequantized, bias)


def _quantize_native_input(inputs):
    triton, amax_kernel, quantize_kernel = _native_quantization_kernels()
    # A global reduction compiled as one generic kernel scales poorly for large
    # videos. Reduce fixed-size blocks first, then reduce the small partial array.
    # The cast kernel normalizes in FP32 registers and writes FP8 directly.
    count = inputs.numel()
    blocks = triton.cdiv(count, 4096)
    partials = torch.empty(blocks, device=inputs.device, dtype=torch.float32)
    quantized = torch.empty_like(inputs, dtype=FP8_DTYPE)
    with torch.cuda.device(inputs.device):
        amax_kernel[(blocks,)](inputs, partials, count, BLOCK=4096)
        scale = partials.amax().div_(448).clamp_min_(torch.finfo(torch.float32).tiny)
        quantize_kernel[(blocks,)](inputs, quantized, scale, count, BLOCK=4096)
    return quantized, scale


def warmup_native_fp8(device="cuda", input_shapes=None):
    """Compile representative activation sizes separately from inference timing.

    This warms the quantizer, not a complete denoising step or its GEMMs. A full
    discarded inference remains the most complete end-to-end warmup.
    """
    report = validate_native_fp8(device)
    if input_shapes is None:
        input_shapes = ((1, 256), (512, 4096), (257, 1280), (26520, 5120), (26520, 13824))
    timings = []
    with torch.inference_mode():
        for shape in input_shapes:
            inputs = torch.ones(shape, device=device, dtype=torch.bfloat16)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            quantized, scale = _quantize_native_input(inputs)
            torch.cuda.synchronize(device)
            timings.append({"input_shape": list(shape), "seconds": time.perf_counter() - start})
            del inputs, quantized, scale
    report["quantizer_warmup"] = timings
    return report


def _native_linear(x, weight, bias, scale):
    inputs = x.reshape(-1, x.shape[-1]).contiguous()
    quantized_inputs, input_scale = _quantize_native_input(inputs)
    quantized_weight = weight.to(device=x.device).contiguous()
    weight_scale = scale.to(device=x.device, dtype=torch.float32).reshape(())
    bias = None if bias is None else bias.to(device=x.device, dtype=torch.bfloat16).contiguous()
    output = torch._scaled_mm(
        quantized_inputs, quantized_weight.t(), scale_a=input_scale, scale_b=weight_scale,
        bias=bias, out_dtype=torch.bfloat16, use_fast_accum=False,
    )
    return output.reshape(*x.shape[:-1], weight.shape[0])


def _dispatch_linear(module, x):
    if module.compute_backend == "bf16":
        result = _linear(x, module.weight, module.bias, module.weight_scale)
        module._compute_stats["bf16_calls"] += 1
        return result
    if module.compute_backend != "native":
        raise ValueError(f"Unknown FP8 compute backend: {module.compute_backend}")
    if not x.is_cuda:
        raise RuntimeError("Native FP8 computation requires CUDA activations; use the bf16 backend on CPU.")
    if x.dtype != torch.bfloat16 or module.weight.dtype != FP8_DTYPE:
        raise ValueError("Native FP8 requires BF16 activations and E4M3 FP8 weights.")
    if x.ndim == 0 or x.shape[-1] != module.weight.shape[1]:
        raise ValueError("Linear input shape does not match its FP8 weight.")
    if x.numel() == 0 or module.weight.shape[0] == 0:
        reason = "empty_input_or_output"
    elif module.weight.shape[0] % 16 or module.weight.shape[1] % 16:
        reason = "K_or_N_not_divisible_by_16"
    else:
        # Kernel failures are intentionally propagated, never disguised as BF16.
        result = _native_linear(x, module.weight, module.bias, module.weight_scale)
        module._compute_stats["native_calls"] += 1
        return result
    result = _linear(x, module.weight, module.bias, module.weight_scale)
    module._compute_stats["bf16_calls"] += 1
    module._compute_stats["fallback_calls"] += 1
    reasons = module._compute_stats["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1
    return result


class ScaledFP8Linear(torch.nn.Linear):
    """Checkpoint layout marker, also usable without the offloading wrapper."""

    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=torch.bfloat16)
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=FP8_DTYPE),
            requires_grad=False,
        )
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.register_buffer("weight_scale", torch.ones((), dtype=torch.float32, device=device))
        self.compute_backend = "bf16"
        self._compute_stats = _new_compute_stats()

    @classmethod
    def from_linear(cls, module):
        result = cls(module.in_features, module.out_features, module.bias is not None, device="meta")
        weight, scale = quantize_linear_weight(module.weight)
        result.weight = torch.nn.Parameter(weight, requires_grad=False)
        result.weight_scale = scale
        if module.bias is not None:
            result.bias = torch.nn.Parameter(module.bias.detach().to(torch.bfloat16), requires_grad=False)
        return result

    def forward(self, x):
        return _dispatch_linear(self, x)


class AutoWrappedScaledFP8Linear(AutoTorchModule):
    """Keep immutable weights in FP8 and reuse their original CPU backing storage."""

    def __init__(self, module, offload_device, onload_device, computation_device,
                 computation_dtype, vram_limit=None, **kwargs):
        super().__init__()
        if vram_limit is not None:
            raise ValueError("Scaled FP8 requires a fixed persistent parameter count.")
        self.weight = module.weight
        self.bias = module.bias
        self.register_buffer("weight_scale", module.weight_scale)
        self.offload_device = offload_device
        self.onload_device = onload_device
        self.computation_device = computation_device
        self.computation_dtype = computation_dtype
        self.compute_backend = module.compute_backend
        self._compute_stats = _new_compute_stats()
        self.state = 0
        self.to(device=offload_device)
        # Keep Tensor views, not Parameter objects: Module.to() can replace a
        # Parameter's .data while these views must retain the original storage.
        # They are intentionally not buffers, so onload leaves them on the CPU.
        self._offload_weight = self.weight.detach()
        self._offload_bias = None if self.bias is None else self.bias.detach()
        self._offload_weight_scale = self.weight_scale.detach()

    def onload(self):
        if self.state != 1:
            self.to(device=self.onload_device)
            self.state = 1

    def offload(self):
        if self.state != 0:
            # Exported weights are immutable. Restoring their CPU views avoids
            # copying GPU weights into a second allocation beside the mmap.
            self.weight.data = self._offload_weight
            if self.bias is not None:
                self.bias.data = self._offload_bias
            self.weight_scale = self._offload_weight_scale
            self.state = 0

    def keep(self):
        raise RuntimeError("Use fixed residency for scaled FP8 weights.")

    def forward(self, x):
        if x.dtype != self.computation_dtype:
            raise ValueError(f"Expected {self.computation_dtype} activations, got {x.dtype}.")
        return _dispatch_linear(self, x)
