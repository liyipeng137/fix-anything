"""Scaled FP8 weight storage with BF16 computation and bounded CPU offloading.

Only Linear weights are quantized. Biases and other model parameters retain BF16;
one FP32 scalar per Linear records its quantization scale.
"""
import torch
import torch.nn.functional as F

from diffsynth.vram_management.layers import AutoTorchModule


FP8_DTYPE = torch.float8_e4m3fn


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
        return _linear(x, self.weight, self.bias, self.weight_scale)


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
        return _linear(x, self.weight, self.bias, self.weight_scale)
