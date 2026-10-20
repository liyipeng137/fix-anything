"""Small, offline CPU checks for the compact deployment checkpoint format.

Run with: python -m unittest discover -s tests -p 'test_deployment.py'
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from fixanything.fp8 import (
    FP8_DTYPE,
    AutoWrappedScaledFP8Linear,
    ScaledFP8Linear,
    quantize_linear_weight,
)


def make_linear():
    with torch.random.fork_rng():
        torch.manual_seed(73)
        return torch.nn.Linear(128, 64, dtype=torch.bfloat16)


class TinyMixedModel(torch.nn.Module):
    """Enough layer variety to catch accidental blanket dtype conversion."""

    def __init__(self, device="cpu", compact=False):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(128, 64, dtype=torch.bfloat16, device=device),
            torch.nn.LayerNorm(64, dtype=torch.bfloat16, device=device),
            torch.nn.Linear(64, 8, bias=False, dtype=torch.bfloat16, device=device),
        )
        self.modulation = torch.nn.Parameter(
            torch.ones((2, 8), dtype=torch.bfloat16, device=device)
        )
        if compact:
            self.layers[0] = ScaledFP8Linear(128, 64, device=device)
            self.layers[2] = ScaledFP8Linear(64, 8, bias=False, device=device)

    def forward(self, x):
        return self.layers(x) + self.modulation[0]


class LicensePathTests(unittest.TestCase):
    def test_wheel_uses_license_from_real_distribution_metadata(self):
        from importlib.metadata import PathDistribution
        from fixanything import deployment

        with tempfile.TemporaryDirectory() as temporary:
            site_packages = Path(temporary) / "site-packages"
            module = site_packages / "fixanything" / "deployment.py"
            module.parent.mkdir(parents=True)
            module.touch()
            dist_info = site_packages / "fixanything-0.1.0.dist-info"
            license_file = dist_info / "licenses" / "LICENSE"
            license_file.parent.mkdir(parents=True)
            license_file.write_text("Fixture Apache-2.0 license")
            (dist_info / "METADATA").write_text("Name: fixanything\nVersion: 0.1.0\n")
            (dist_info / "RECORD").write_text(
                "fixanything/deployment.py,,\n"
                "fixanything-0.1.0.dist-info/licenses/LICENSE,,\n"
            )
            # Real importlib metadata parses the wheel RECORD and resolves its
            # dist-info license; no checkout-level LICENSE is available.
            installed = PathDistribution(dist_info)
            self.assertFalse((site_packages / "LICENSE").exists())
            with (
                patch.object(deployment, "__file__", str(module)),
                patch.object(deployment, "distribution", return_value=installed) as lookup,
            ):
                actual = deployment._license_path()
            lookup.assert_called_once_with("fixanything")
            self.assertEqual(actual, license_file)
            self.assertEqual(actual.read_text(), "Fixture Apache-2.0 license")

    def test_uninstalled_source_checkout_falls_back_to_its_license(self):
        from importlib.metadata import PackageNotFoundError
        from fixanything import deployment

        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            module = checkout / "fixanything" / "deployment.py"
            module.parent.mkdir()
            module.touch()
            license_file = checkout / "LICENSE"
            license_file.write_text("Fixture source license")
            with (
                patch.object(deployment, "__file__", str(module)),
                patch.object(deployment, "distribution", side_effect=PackageNotFoundError("fixanything")),
            ):
                self.assertEqual(deployment._license_path(), license_file)

    def test_missing_license_stops_export_before_output_or_model_loading(self):
        from fixanything import deployment

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / deployment.BASE_MODEL_ID
            (base / "google" / "umt5-xxl").mkdir(parents=True)
            for filename in (
                "diffusion_pytorch_model-00001-of-00001.safetensors",
                "models_t5_umt5-xxl-enc-bf16.pth",
                "Wan2.1_VAE.pth",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ):
                (base / filename).touch()
            lora = root / "fixanything_lora.safetensors"
            lora.touch()
            output = root / "deployment-output"
            with (
                patch.object(deployment, "_license_path", side_effect=FileNotFoundError("Missing license")),
                patch(
                    "fixanything.pipelines.WanVideoPipeline.from_pretrained",
                    side_effect=AssertionError("Model loading must follow license validation"),
                ) as load,
                self.assertRaisesRegex(FileNotFoundError, "Missing license"),
            ):
                deployment.export_deployment(output, model_dir=root, lora_path=lora, device="cpu")
            load.assert_not_called()
            self.assertFalse(output.exists())


class ScaledFP8Tests(unittest.TestCase):
    def test_zero_weight_has_unit_scale_and_exact_zero_output(self):
        source = torch.zeros((9, 13), dtype=torch.bfloat16)
        quantized, scale = quantize_linear_weight(source)
        self.assertEqual(quantized.dtype, FP8_DTYPE)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(scale.shape, torch.Size([]))
        self.assertEqual(scale.item(), 1.0)
        self.assertTrue(torch.equal(quantized.float(), source.float()))

    def test_normal_and_outlier_weights_are_finite_and_bounded(self):
        source = make_linear().weight.detach().clone()
        source[0, 0] = 1000.0
        before = source.clone()
        quantized, scale = quantize_linear_weight(source)
        restored = quantized.float() * scale
        self.assertTrue(torch.equal(source, before))
        self.assertTrue(torch.isfinite(restored).all().item())
        self.assertGreater(scale.item(), 0.0)
        self.assertLessEqual(quantized.float().abs().max().item(), 448.0)
        # E4M3 normal rounding has at most 1/16 relative error. Values below
        # its normal range need half a subnormal step as an absolute allowance.
        torch.testing.assert_close(
            restored, source.float(), rtol=0.07, atol=scale.item() / 1024
        )

    def test_quantizer_rejects_nonfinite_and_nonfloating_weights(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quantize_linear_weight(torch.tensor([[value]]))
        with self.assertRaises(ValueError):
            quantize_linear_weight(torch.tensor([[1]], dtype=torch.int64))

    def test_marker_preserves_bias_and_persistent_scale(self):
        source = make_linear()
        compact = ScaledFP8Linear.from_linear(source)
        self.assertEqual(compact.weight.dtype, FP8_DTYPE)
        self.assertEqual(compact.bias.dtype, torch.bfloat16)
        self.assertEqual(compact.weight_scale.dtype, torch.float32)
        self.assertFalse(compact.weight.requires_grad)
        self.assertFalse(compact.bias.requires_grad)
        torch.testing.assert_close(compact.bias, source.bias, rtol=0, atol=0)
        self.assertIn("weight_scale", compact.state_dict())
        original_bytes = sum(t.numel() * t.element_size() for t in source.state_dict().values())
        compact_bytes = sum(t.numel() * t.element_size() for t in compact.state_dict().values())
        self.assertLess(compact_bytes, original_bytes * 0.6)

    def test_marker_and_offload_wrapper_match_dequantized_bf16_linear(self):
        compact = ScaledFP8Linear.from_linear(make_linear())
        x = torch.linspace(-1, 1, 3 * 128, dtype=torch.bfloat16).reshape(3, 128)
        restored = compact.weight.to(torch.bfloat16)
        restored.mul_(compact.weight_scale)
        expected = F.linear(x, restored, compact.bias)
        wrapper = AutoWrappedScaledFP8Linear(
            compact,
            offload_device="cpu",
            onload_device="cpu",
            computation_device="cpu",
            computation_dtype=torch.bfloat16,
        )
        for actual in (compact(x), wrapper(x)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.dtype, torch.bfloat16)
        wrapper.onload()
        self.assertEqual(wrapper.state, 1)
        wrapper.offload()
        self.assertEqual(wrapper.state, 0)
        self.assertEqual(wrapper.weight.dtype, FP8_DTYPE)
        self.assertEqual(wrapper.weight_scale.dtype, torch.float32)
        with self.assertRaises(ValueError):
            wrapper(x.float())

    def test_bias_free_linear_stays_bias_free(self):
        source = torch.nn.Linear(8, 4, bias=False, dtype=torch.bfloat16)
        compact = ScaledFP8Linear.from_linear(source)
        self.assertIsNone(compact.bias)
        output = compact(torch.ones((2, 8), dtype=torch.bfloat16))
        self.assertEqual(output.shape, (2, 4))
        self.assertTrue(torch.isfinite(output).all().item())


class OffloadStorageTests(unittest.TestCase):
    def check_roundtrips(self, bias):
        compact = ScaledFP8Linear.from_linear(
            torch.nn.Linear(16, 4, bias=bias, dtype=torch.bfloat16)
        )
        wrapper = AutoWrappedScaledFP8Linear(
            compact,
            offload_device="cpu",
            onload_device="cpu",
            computation_device="cpu",
            computation_dtype=torch.bfloat16,
        )
        names = ["weight", "weight_scale"] + (["bias"] if bias else [])
        pointers = {name: getattr(wrapper, name).data_ptr() for name in names}
        parameter_ids = {name: id(value) for name, value in wrapper.named_parameters()}
        dtypes = {name: getattr(wrapper, name).dtype for name in names}
        x = torch.linspace(-1, 1, 32, dtype=torch.bfloat16).reshape(2, 16)
        expected = wrapper(x).clone()

        def simulate_device_copy(*args, **kwargs):
            # Same-device Module.to() would reuse CPU storage, so clone each
            # tensor to simulate a transfer without allocating on a GPU.
            wrapper.weight.data = wrapper.weight.detach().clone()
            if wrapper.bias is not None:
                wrapper.bias.data = wrapper.bias.detach().clone()
            wrapper.weight_scale = wrapper.weight_scale.detach().clone()
            return wrapper

        for cycle in range(3):
            with self.subTest(cycle=cycle):
                with patch.object(wrapper, "to", side_effect=simulate_device_copy) as transfer:
                    wrapper.onload()
                    wrapper.onload()
                transfer.assert_called_once_with(device="cpu")
                self.assertEqual(wrapper.state, 1)
                for name in names:
                    self.assertNotEqual(getattr(wrapper, name).data_ptr(), pointers[name], name)
                self.assertEqual(
                    {name: id(value) for name, value in wrapper.named_parameters()}, parameter_ids
                )
                torch.testing.assert_close(wrapper(x), expected, rtol=0, atol=0)

                with patch.object(wrapper, "to", side_effect=AssertionError("Offload must reuse CPU storage")):
                    wrapper.offload()
                    wrapper.offload()
                self.assertEqual(wrapper.state, 0)
                self.assertEqual(
                    {name: id(value) for name, value in wrapper.named_parameters()}, parameter_ids
                )
                for name in names:
                    self.assertEqual(getattr(wrapper, name).data_ptr(), pointers[name], name)
                    self.assertEqual(getattr(wrapper, name).dtype, dtypes[name], name)
                self.assertEqual(wrapper.weight.dtype, FP8_DTYPE)
                self.assertEqual(wrapper.weight_scale.dtype, torch.float32)
                if bias:
                    self.assertEqual(wrapper.bias.dtype, torch.bfloat16)
                else:
                    self.assertIsNone(wrapper.bias)
                torch.testing.assert_close(wrapper(x), expected, rtol=0, atol=0)

    def test_biased_linear_reuses_original_cpu_storage_after_repeated_offload(self):
        self.check_roundtrips(bias=True)

    def test_bias_free_linear_reuses_original_cpu_storage_after_repeated_offload(self):
        self.check_roundtrips(bias=False)


class ImmutableDeploymentTests(unittest.TestCase):
    def test_lora_rejected_before_file_loading_for_marker_and_wrapped_linear(self):
        from fixanything.pipelines.wan_video_v2v_mask_dynamic import WanVideoPipeline

        marker = ScaledFP8Linear.from_linear(make_linear())
        wrapped = AutoWrappedScaledFP8Linear(
            ScaledFP8Linear.from_linear(make_linear()),
            offload_device="cpu",
            onload_device="cpu",
            computation_device="cpu",
            computation_dtype=torch.bfloat16,
        )
        pipe = SimpleNamespace(torch_dtype=torch.bfloat16, device="cpu")
        with tempfile.TemporaryDirectory() as temporary:
            missing_lora = Path(temporary) / "does-not-exist.safetensors"
            self.assertFalse(missing_lora.exists())
            for layer in (marker, wrapped):
                with (
                    self.subTest(layer=type(layer).__name__),
                    patch(
                        "fixanything.pipelines.wan_video_v2v_mask_dynamic.load_state_dict",
                        side_effect=AssertionError("Immutable FP8 weights must reject LoRA before file access"),
                    ) as load,
                    self.assertRaisesRegex(ValueError, "immutable"),
                ):
                    WanVideoPipeline.load_lora(pipe, torch.nn.Sequential(layer), str(missing_lora))
                load.assert_not_called()


class ComponentTests(unittest.TestCase):
    def setUp(self):
        from fixanything.deployment import (
            load_component,
            quantized_state_dict,
            save_sharded_state_dict,
        )

        self.load_component = load_component
        self.quantized_state_dict = quantized_state_dict
        self.save_sharded_state_dict = save_sharded_state_dict
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        with torch.random.fork_rng():
            torch.manual_seed(101)
            self.source = TinyMixedModel()
        self.state, self.quantized_names = self.quantized_state_dict(self.source)

    def save(self, state=None, prefix="dit"):
        return self.save_sharded_state_dict(
            self.state if state is None else state,
            self.directory,
            prefix,
            max_shard_bytes=4096,
        )

    def empty_model(self):
        return TinyMixedModel(device="meta", compact=True)

    def test_mixed_model_roundtrip_preserves_dtypes_scales_and_output(self):
        self.assertEqual(set(self.quantized_names), {"layers.0", "layers.2"})
        for name, tensor in self.state.items():
            expected_dtype = (
                FP8_DTYPE
                if name in {"layers.0.weight", "layers.2.weight"}
                else torch.float32 if name.endswith("weight_scale") else torch.bfloat16
            )
            self.assertEqual(tensor.dtype, expected_dtype, name)
        component = self.save()
        self.assertGreater(len(component["files"]), 1)
        restored = self.empty_model()
        self.load_component(restored, component, self.directory, verify_hashes=True)
        for name, tensor in restored.state_dict().items():
            self.assertEqual(tensor.device.type, "cpu", name)
            self.assertEqual(tensor.dtype, self.state[name].dtype, name)
            # Compare storage bytes too: float8 does not support every CPU
            # comparison kernel, and conversion could hide a dtype regression.
            self.assertTrue(
                torch.equal(
                    tensor.reshape(-1).view(torch.uint8),
                    self.state[name].reshape(-1).view(torch.uint8),
                ),
                name,
            )
        reference = TinyMixedModel(compact=True)
        reference.load_state_dict(self.state, strict=True, assign=True)
        x = torch.linspace(-1, 1, 256, dtype=torch.bfloat16).reshape(2, 128)
        torch.testing.assert_close(restored(x), reference(x), rtol=0, atol=0)

    def test_quantized_shards_are_smaller_than_original_bf16_payload(self):
        compact = self.save()
        original = self.save(self.source.state_dict(), prefix="bf16")
        compact_bytes = sum((self.directory / f["name"]).stat().st_size for f in compact["files"])
        original_bytes = sum((self.directory / f["name"]).stat().st_size for f in original["files"])
        self.assertLess(compact_bytes, original_bytes * 0.65)

    def test_missing_shard_is_rejected(self):
        component = self.save()
        (self.directory / component["files"][0]["name"]).unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.load_component(self.empty_model(), component, self.directory)

    def test_bf16_payload_in_fp8_linear_is_rejected(self):
        invalid = dict(self.state)
        invalid["layers.0.weight"] = invalid["layers.0.weight"].to(torch.bfloat16)
        component = self.save(invalid)
        with self.assertRaises(ValueError):
            self.load_component(self.empty_model(), component, self.directory)

    def test_weight_shape_mismatch_is_rejected(self):
        invalid = dict(self.state)
        invalid["layers.0.weight"] = invalid["layers.0.weight"].T.contiguous()
        component = self.save(invalid)
        with self.assertRaises(ValueError):
            self.load_component(self.empty_model(), component, self.directory)

    def test_wrong_scale_dtype_or_value_is_rejected(self):
        for scale in (
            torch.tensor(1.0, dtype=torch.bfloat16),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(-1.0, dtype=torch.float32),
            torch.tensor(float("nan"), dtype=torch.float32),
            torch.tensor(float("inf"), dtype=torch.float32),
        ):
            with self.subTest(scale=scale):
                invalid = dict(self.state)
                invalid["layers.0.weight_scale"] = scale
                component = self.save(invalid)
                with self.assertRaises(ValueError):
                    self.load_component(self.empty_model(), component, self.directory)

    def test_missing_scale_is_rejected(self):
        invalid = dict(self.state)
        del invalid["layers.0.weight_scale"]
        component = self.save(invalid)
        with self.assertRaises(ValueError):
            self.load_component(self.empty_model(), component, self.directory)

    def test_manifest_dtype_mismatch_is_rejected(self):
        component = self.save()
        component["tensors"]["layers.0.weight"]["dtype"] = "bfloat16"
        with self.assertRaises(ValueError):
            self.load_component(self.empty_model(), component, self.directory)

    def test_checksum_failure_is_rejected_before_loading(self):
        component = self.save()
        shard = self.directory / component["files"][0]["name"]
        with shard.open("r+b") as stream:
            stream.seek(-1, 2)
            value = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([value[0] ^ 1]))
        with self.assertRaises(ValueError):
            self.load_component(self.empty_model(), component, self.directory, verify_hashes=True)


class CachedPromptTests(unittest.TestCase):
    def setUp(self):
        from fixanything.pipelines.wan_video_v2v_mask_dynamic import WanVideoUnit_PromptEmbedder

        self.unit = WanVideoUnit_PromptEmbedder()
        self.pipe = SimpleNamespace(
            cached_prompt_embeddings={
                "positive": torch.full((1, 3, 4), 0.25, dtype=torch.bfloat16),
                "negative": torch.full((1, 3, 4), -0.5, dtype=torch.bfloat16),
            },
            cached_prompts={"positive": "fixed positive", "negative": "fixed negative"},
            torch_dtype=torch.bfloat16,
            device="cpu",
            text_encoder=None,
            load_models_to_device=Mock(side_effect=AssertionError("Cached prompts must not load T5")),
            prompter=SimpleNamespace(
                encode_prompt=Mock(side_effect=AssertionError("Cached prompts must not tokenize"))
            ),
        )

    def test_both_cached_contexts_work_without_tokenizer_or_text_encoder(self):
        for positive, key in ((True, "positive"), (False, "negative")):
            with self.subTest(key=key):
                result = self.unit.process(self.pipe, self.pipe.cached_prompts[key], positive)
                self.assertEqual(set(result), {"context"})
                torch.testing.assert_close(
                    result["context"], self.pipe.cached_prompt_embeddings[key], rtol=0, atol=0
                )
                self.assertEqual(result["context"].dtype, torch.bfloat16)
        self.pipe.load_models_to_device.assert_not_called()
        self.pipe.prompter.encode_prompt.assert_not_called()

    def test_changed_positive_or_negative_prompt_is_rejected(self):
        for positive, prompt in ((True, "changed positive"), (False, "changed negative")):
            with self.subTest(positive=positive), self.assertRaises(ValueError):
                self.unit.process(self.pipe, prompt, positive)
        self.pipe.load_models_to_device.assert_not_called()

    def test_missing_cached_context_is_rejected(self):
        del self.pipe.cached_prompt_embeddings["positive"]
        with self.assertRaises(ValueError):
            self.unit.process(self.pipe, "fixed positive", True)

    def test_missing_prompt_metadata_is_rejected(self):
        self.pipe.cached_prompts = None
        with self.assertRaises(ValueError):
            self.unit.process(self.pipe, "fixed positive", True)

    def test_missing_cache_and_text_encoder_is_rejected(self):
        self.pipe.cached_prompt_embeddings = None
        with self.assertRaises(ValueError):
            self.unit.process(self.pipe, "fixed positive", True)


class DeploymentLoaderTests(unittest.TestCase):
    def test_unknown_compute_backend_is_rejected_before_reading_weights(self):
        from fixanything.deployment import load_deployment

        with self.assertRaisesRegex(ValueError, "fp8_compute"):
            load_deployment("/nonexistent-deployment", fp8_compute="automatic")

    def test_package_loads_through_meta_factories_without_originals_or_downloads(self):
        from fixanything.deployment import (
            FORMAT_VERSION,
            load_deployment,
            quantized_state_dict,
            save_sharded_state_dict,
        )

        class FakePipeline(torch.nn.Module):
            def __init__(self, device, torch_dtype):
                super().__init__()
                self.device, self.torch_dtype = device, torch_dtype

            def enable_vram_management(self, num_persistent_param_in_dit):
                self.resident_count = num_persistent_param_in_dit

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dit_state, targets = quantized_state_dict(TinyMixedModel())
            image_encoder = torch.nn.Linear(8, 4, dtype=torch.bfloat16)
            vae = torch.nn.Conv2d(1, 1, 1, dtype=torch.bfloat16)
            contexts = {
                "positive": torch.zeros((1, 512, 4096), dtype=torch.bfloat16),
                "negative": torch.ones((1, 512, 4096), dtype=torch.bfloat16),
            }
            components = {}
            for name, state in (
                ("dit", dit_state),
                ("image_encoder", image_encoder.state_dict()),
                ("vae", vae.state_dict()),
                ("prompt_embeddings", contexts),
            ):
                components[name] = save_sharded_state_dict(state, directory, name)
            manifest = {
                "format": FORMAT_VERSION,
                "computation_dtype": "bfloat16",
                "quantization": {"method": "scaled_fp8_weight_only", "modules": targets},
                "dit_config": {},
                "vae_config": {"z_dim": 16},
                "components": components,
                "prompts": {"positive": "fixed positive", "negative": "fixed negative"},
            }
            (directory / "deployment.json").write_text(json.dumps(manifest))
            # The real loader orchestration is exercised, but its three model
            # factories are tiny so this never allocates the production model.
            with (
                patch("diffsynth.models.wan_video_dit.WanModel", TinyMixedModel),
                patch("diffsynth.models.wan_video_image_encoder.WanImageEncoder", lambda: torch.nn.Linear(8, 4)),
                patch("diffsynth.models.wan_video_vae.WanVideoVAE", lambda z_dim: torch.nn.Conv2d(1, 1, 1)),
                patch("fixanything.pipelines.WanVideoPipeline", FakePipeline),
                patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("Offline loader attempted download")),
                patch("huggingface_hub.snapshot_download", side_effect=AssertionError("Offline loader attempted download")),
            ):
                pipe = load_deployment(directory, device="cpu", num_persistent_param_in_dit=0, verify_hashes=True)
            self.assertEqual(pipe.resident_count, 0)
            self.assertEqual(pipe.fp8_compute, "bf16")
            self.assertIsNone(pipe.fp8_capability)
            self.assertFalse(pipe.training)
            self.assertIsInstance(pipe.dit.layers[0], ScaledFP8Linear)
            self.assertEqual(pipe.dit.layers[0].weight.dtype, FP8_DTYPE)
            self.assertEqual(pipe.dit.layers[0].compute_backend, "bf16")
            self.assertEqual(pipe.dit.layers[0].weight_scale.dtype, torch.float32)
            self.assertEqual(pipe.dit.layers[0].bias.dtype, torch.bfloat16)
            self.assertEqual(pipe.image_encoder.weight.dtype, torch.bfloat16)
            self.assertEqual(pipe.vae.weight.dtype, torch.bfloat16)
            self.assertEqual(pipe.cached_prompts, manifest["prompts"])
            for key, value in contexts.items():
                torch.testing.assert_close(pipe.cached_prompt_embeddings[key], value, rtol=0, atol=0)
            self.assertTrue(all(not parameter.is_meta for parameter in pipe.parameters()))


if __name__ == "__main__":
    unittest.main()
