"""Native FP8 backend checks; CUDA tests require FIXANYTHING_TEST_NATIVE_CUDA=1.

The default test discovery stays CPU-only. Explicit CUDA tests probe the current
GPU and compile the activation quantizer; no production checkpoints are needed.
"""
import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from fixanything.fp8 import (
    FP8_DTYPE,
    AutoWrappedScaledFP8Linear,
    ScaledFP8Linear,
    fp8_compute_stats,
    reset_fp8_compute_stats,
    validate_native_fp8,
)


RUN_CUDA = os.environ.get("FIXANYTHING_TEST_NATIVE_CUDA") == "1" and torch.cuda.is_available()


class NativeFP8ConfigurationTests(unittest.TestCase):
    def test_original_backend_is_default_and_statistics_can_be_reset(self):
        module = ScaledFP8Linear.from_linear(torch.nn.Linear(16, 16, dtype=torch.bfloat16))
        self.assertEqual(module.compute_backend, "bf16")
        module(torch.ones(2, 16, dtype=torch.bfloat16))
        stats = fp8_compute_stats(module)
        self.assertEqual(stats["bf16_calls"], 1)
        self.assertEqual(stats["native_calls"], 0)
        reset_fp8_compute_stats(module)
        self.assertEqual(fp8_compute_stats(module)["bf16_calls"], 0)

    def test_wrapper_inherits_backend_and_preserves_weight_storage(self):
        module = ScaledFP8Linear.from_linear(torch.nn.Linear(16, 16, dtype=torch.bfloat16))
        module.compute_backend = "native"
        wrapper = AutoWrappedScaledFP8Linear(
            module, offload_device="cpu", onload_device="cpu",
            computation_device="cpu", computation_dtype=torch.bfloat16,
        )
        self.assertEqual(wrapper.compute_backend, "native")
        self.assertEqual(wrapper.weight.dtype, FP8_DTYPE)
        self.assertEqual(fp8_compute_stats(wrapper)["native_modules"], 1)
        wrapper.onload()
        wrapper.offload()
        self.assertEqual(wrapper.weight.dtype, FP8_DTYPE)

    def test_native_activation_and_capability_checks_reject_cpu(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            validate_native_fp8("cpu")
        module = ScaledFP8Linear.from_linear(torch.nn.Linear(16, 16, dtype=torch.bfloat16))
        module.compute_backend = "native"
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            module(torch.ones(2, 16, dtype=torch.bfloat16))
        self.assertEqual(fp8_compute_stats(module)["fallback_calls"], 0)

    def test_missing_cuda_is_reported_without_attempting_a_kernel(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                validate_native_fp8("cuda")

    def test_unknown_backend_is_rejected(self):
        module = ScaledFP8Linear.from_linear(torch.nn.Linear(16, 16, dtype=torch.bfloat16))
        module.compute_backend = "misspelled"
        with self.assertRaisesRegex(ValueError, "Unknown"):
            module(torch.ones(2, 16, dtype=torch.bfloat16))


@unittest.skipUnless(RUN_CUDA, "Set FIXANYTHING_TEST_NATIVE_CUDA=1 to run CUDA kernel tests")
class NativeFP8CUDATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        validate_native_fp8("cuda")

    def make_module(self, k=128, n=64, bias=True):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(107)
            source = torch.nn.Linear(k, n, bias=bias, device="cuda", dtype=torch.bfloat16)
            result = ScaledFP8Linear.from_linear(source)
        result.compute_backend = "native"
        return result

    @torch.inference_mode()
    def test_actual_scaled_mm_supports_small_and_irregular_m_with_bias(self):
        for m in (1, 7, 16, 17):
            with self.subTest(m=m):
                module = self.make_module()
                x = torch.linspace(-2, 2, m * 128, device="cuda", dtype=torch.bfloat16).reshape(m, 128)
                actual = module(x)
                scale = x.abs().amax().float().div(448).clamp_min(torch.finfo(torch.float32).tiny)
                quantized_x = (x.float() / scale).clamp(-448, 448).to(FP8_DTYPE)
                expected = F.linear(
                    quantized_x.float() * scale,
                    module.weight.float() * module.weight_scale,
                    module.bias.float(),
                )
                relative_l2 = (actual.float() - expected).norm() / expected.norm()
                self.assertLess(relative_l2.item(), 0.01)
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertEqual(fp8_compute_stats(module)["native_calls"], 1)
                self.assertEqual(fp8_compute_stats(module)["fallback_calls"], 0)

    @torch.inference_mode()
    def test_noncontiguous_and_batched_inputs_restore_their_output_shape(self):
        module = self.make_module()
        x = torch.randn(2, 128, 3, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        actual = module(x)
        self.assertEqual(actual.shape, (2, 3, 64))
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertEqual(fp8_compute_stats(module)["native_calls"], 1)

    @torch.inference_mode()
    def test_all_zero_activations_remain_finite_and_return_bias(self):
        module = self.make_module()
        actual = module(torch.zeros(5, 128, device="cuda", dtype=torch.bfloat16))
        torch.testing.assert_close(actual, module.bias.expand(5, 64), rtol=0, atol=0)

    @torch.inference_mode()
    def test_unaligned_shapes_fall_back_explicitly_and_are_counted(self):
        for k, n in ((15, 16), (16, 15)):
            with self.subTest(k=k, n=n):
                module = self.make_module(k=k, n=n, bias=False)
                x = torch.randn(2, k, device="cuda", dtype=torch.bfloat16)
                actual = module(x)
                module.compute_backend = "bf16"
                expected = module(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                stats = fp8_compute_stats(module)
                self.assertEqual(stats["native_calls"], 0)
                self.assertEqual(stats["fallback_calls"], 1)
                self.assertEqual(stats["fallback_reasons"], {"K_or_N_not_divisible_by_16": 1})

    @torch.inference_mode()
    def test_kernel_failure_is_propagated_without_silent_fallback(self):
        module = self.make_module()
        x = torch.ones(16, 128, device="cuda", dtype=torch.bfloat16)
        with patch("torch._scaled_mm", side_effect=RuntimeError("injected kernel failure")):
            with self.assertRaisesRegex(RuntimeError, "injected kernel failure"):
                module(x)
        stats = fp8_compute_stats(module)
        self.assertEqual(stats["native_calls"], 0)
        self.assertEqual(stats["fallback_calls"], 0)


if __name__ == "__main__":
    unittest.main()
