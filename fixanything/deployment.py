"""Offline export and loading of self-contained FixAnything FP8 deployment packages."""
import gc
import hashlib
import json
import shutil
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .fp8 import ScaledFP8Linear, quantize_linear_weight
from .prompts import DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT


FORMAT_VERSION = "fixanything-scaled-fp8-v1"
BASE_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
DIFFSYNTH_COMMIT = "afd101f3452c9ecae0c87b79adfa2e22d65ffdc3"


def _license_path():
    # Wheels put licenses in dist-info, outside the importable package directory.
    try:
        installed = distribution("fixanything")
        for filename in installed.files or []:
            if filename.name in ("LICENSE", "LICENSE.txt") and ".dist-info/" in str(filename):
                candidate = installed.locate_file(filename)
                if candidate.is_file():
                    return candidate
    except PackageNotFoundError:
        pass
    candidate = Path(__file__).resolve().parents[1] / "LICENSE"
    if not candidate.is_file():
        raise FileNotFoundError("FixAnything's Apache-2.0 license is missing from this installation.")
    return candidate


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def quantized_state_dict(model):
    """Quantize Linear weights; keep biases, normalization and convolution in BF16."""
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prefix = name + "." if name else ""
            state[prefix + "weight"], state[prefix + "weight_scale"] = quantize_linear_weight(state[prefix + "weight"])
            targets.append(name)
    return state, targets


def save_sharded_state_dict(state, directory, prefix, max_shard_bytes=2_000_000_000):
    """Save bounded safetensors shards and record content hashes and tensor schemas."""
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    groups, group, size = [], {}, 0
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        nbytes = value.numel() * value.element_size()
        if group and size + nbytes > max_shard_bytes:
            groups.append(group)
            group, size = {}, 0
        group[key] = value
        size += nbytes
    if group:
        groups.append(group)
    component = {"files": [], "weight_map": {}, "tensors": {}}
    for index, shard in enumerate(groups, 1):
        filename = f"{prefix}-{index:05d}-of-{len(groups):05d}.safetensors"
        path = directory / filename
        save_file(shard, str(path), metadata={"format": "pt"})
        component["files"].append({"name": filename, "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
        for key, value in shard.items():
            component["weight_map"][key] = filename
            component["tensors"][key] = {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
        print(f"Saved {filename}: {path.stat().st_size / 1e9:.3f} GB", flush=True)
    return component


def _component_state(component, directory, expected=None, verify_hashes=False):
    directory = Path(directory)
    if set(component["tensors"]) != set(component["weight_map"]):
        raise ValueError("Tensor schema and shard index do not agree.")
    if expected is not None and set(expected) != set(component["weight_map"]):
        raise ValueError("Checkpoint keys do not match the model architecture.")
    file_names = [entry["name"] for entry in component["files"]]
    if len(set(file_names)) != len(file_names) or set(file_names) != set(component["weight_map"].values()):
        raise ValueError("Shard file list does not match the weight index.")
    for entry in component["files"]:
        filename = entry["name"]
        if Path(filename).name != filename:
            raise ValueError("Shard names must be plain filenames.")
        path = directory / filename
        if path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Incorrect shard size: {filename}")
        if verify_hashes and _sha256(path) != entry["sha256"]:
            raise ValueError(f"Incorrect shard hash: {filename}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            expected_keys = {key for key, value in component["weight_map"].items() if value == filename}
            if set(handle.keys()) != expected_keys:
                raise ValueError(f"Unexpected or missing tensors in {filename}")
            shard = {}
            for key in handle.keys():
                value = handle.get_tensor(key)
                schema = component["tensors"][key]
                if list(value.shape) != schema["shape"] or str(value.dtype).removeprefix("torch.") != schema["dtype"]:
                    raise ValueError(f"Tensor schema mismatch: {key}")
                if expected is not None and (value.shape != expected[key].shape or value.dtype != expected[key].dtype):
                    raise ValueError(f"Tensor dtype or shape does not match the model: {key}")
                if key.endswith("weight_scale") and (value.numel() != 1 or not torch.isfinite(value).all() or value.item() <= 0):
                    raise ValueError(f"Invalid FP8 scale: {key}")
                shard[key] = value
            yield shard


def load_component(model, component, directory, verify_hashes=False):
    """Assign validated shards into a model initialized on the meta device."""
    expected = model.state_dict()
    for shard in _component_state(component, directory, expected, verify_hashes):
        model.load_state_dict(shard, strict=False, assign=True)
    if any(tensor.is_meta for tensor in model.state_dict().values()):
        raise ValueError("Checkpoint left uninitialized model tensors.")
    return model.eval()


def _dit_config(model):
    block = model.blocks[0]
    return dict(
        dim=model.dim, in_dim=model.in_dim, ffn_dim=block.ffn_dim, out_dim=model.head.head.out_features // 4,
        text_dim=model.text_embedding[0].in_features, freq_dim=model.freq_dim,
        eps=block.self_attn.norm_q.eps, patch_size=list(model.patch_size),
        num_heads=block.self_attn.num_heads, num_layers=len(model.blocks),
        has_image_input=model.has_image_input, has_image_pos_emb=model.has_image_pos_emb,
        require_vae_embedding=model.require_vae_embedding, require_clip_embedding=model.require_clip_embedding,
    )


@torch.no_grad()
def export_deployment(output_dir, model_dir="checkpoints", lora_path="checkpoints/fixanything_lora.safetensors",
                      device="cuda", lora_alpha=1.0, prompt=DEFAULT_PROMPT,
                      negative_prompt=DEFAULT_NEGATIVE_PROMPT, max_shard_bytes=2_000_000_000):
    """Build a standalone package from local originals; never download missing files."""
    from diffsynth.utils import ModelConfig
    from .pipelines import WanVideoPipeline

    base = Path(model_dir) / BASE_MODEL_ID
    patterns = ["diffusion_pytorch_model*.safetensors", "models_t5_umt5-xxl-enc-bf16.pth",
                "Wan2.1_VAE.pth", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"]
    configs = []
    for pattern in patterns:
        files = sorted(str(path) for path in base.glob(pattern))
        if not files:
            raise FileNotFoundError(f"Missing local checkpoint: {base / pattern}")
        configs.append(ModelConfig(path=files[0] if len(files) == 1 else files, offload_device="cpu"))
    tokenizer = base / "google/umt5-xxl"
    if not tokenizer.is_dir() or not Path(lora_path).is_file():
        raise FileNotFoundError("Export requires the original tokenizer and FixAnything LoRA.")
    license_path = _license_path()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format": FORMAT_VERSION,
        "base_model": BASE_MODEL_ID,
        "diffsynth_commit": DIFFSYNTH_COMMIT,
        "computation_dtype": "bfloat16",
        "quantization": {"method": "scaled_fp8_weight_only", "dtype": "float8_e4m3fn",
                         "scale_dtype": "float32", "granularity": "per_linear_tensor"},
        "lora": {"merged": True, "alpha": lora_alpha, "sha256": _sha256(lora_path)},
        "prompts": {"positive": prompt, "negative": negative_prompt},
        "components": {},
    }
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device, model_configs=configs,
        tokenizer_config=ModelConfig(path=str(tokenizer)), redirect_common_files=False,
    )
    pipe.load_lora(pipe.dit, str(lora_path), alpha=lora_alpha)
    pipe.eval()
    print("Merged LoRA in BF16; caching the fixed positive/negative prompts.", flush=True)
    # Only the text encoder is moved to GPU during this offline build. It is omitted from the package.
    pipe.text_encoder.to(device)
    embeddings = {
        "positive": pipe.prompter.encode_prompt(prompt, positive=True, device=device).cpu().contiguous(),
        "negative": pipe.prompter.encode_prompt(negative_prompt, positive=False, device=device).cpu().contiguous(),
    }
    pipe.prompter.text_encoder = None
    pipe.text_encoder = None
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    manifest["components"]["prompt_embeddings"] = save_sharded_state_dict(embeddings, output, "prompts", max_shard_bytes)
    manifest["dit_config"] = _dit_config(pipe.dit)
    print("Quantizing DiT Linear weights with FP32 per-tensor scales.", flush=True)
    state, targets = quantized_state_dict(pipe.dit)
    manifest["quantization"]["modules"] = targets
    manifest["components"]["dit"] = save_sharded_state_dict(state, output, "dit", max_shard_bytes)
    del state
    pipe.dit = None
    gc.collect()
    for name in ("image_encoder", "vae"):
        model = getattr(pipe, name)
        state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
        manifest["components"][name] = save_sharded_state_dict(state, output, name, max_shard_bytes)
        del state
    manifest["vae_config"] = {"z_dim": pipe.vae.model.z_dim}
    manifest["weight_bytes"] = sum(entry["size_bytes"] for part in manifest["components"].values() for entry in part["files"])
    shutil.copyfile(license_path, output / "LICENSE")
    (output / "NOTICE").write_text(
        "FixAnything by Khiem Vuong, Deva Ramanan and Srinivasa Narasimhan.\n"
        "Based on Wan2.1 by the Wan team. Both released under Apache-2.0.\n"
        "This package merges the FixAnything LoRA and quantizes DiT Linear weights to scaled FP8.\n"
        "https://github.com/kvuong2711/fix-anything\nhttps://github.com/Wan-Video/Wan2.1\n"
    )
    (output / "deployment.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Deployment ready: {output} ({manifest['weight_bytes'] / 1e9:.3f} GB weights)", flush=True)
    return manifest


def load_deployment(directory, device="cuda", num_persistent_param_in_dit=12_000_000_000,
                    verify_hashes=False, fp8_compute="bf16"):
    """Load only the exported package, without original weights, tokenizer or T5."""
    from diffsynth.models.utils import init_weights_on_device
    from diffsynth.models.wan_video_dit import WanModel
    from diffsynth.models.wan_video_image_encoder import WanImageEncoder
    from diffsynth.models.wan_video_vae import WanVideoVAE
    from .pipelines import WanVideoPipeline

    if fp8_compute not in ("bf16", "native"):
        raise ValueError("fp8_compute must be 'bf16' or 'native'.")
    capability = None
    if fp8_compute == "native":
        from .fp8 import validate_native_fp8
        capability = validate_native_fp8(device)
    if num_persistent_param_in_dit is None or num_persistent_param_in_dit < 0:
        raise ValueError("FP8 requires a nonnegative fixed persistent parameter count.")
    directory = Path(directory)
    manifest = json.loads((directory / "deployment.json").read_text())
    if manifest["format"] != FORMAT_VERSION:
        raise ValueError(f"Unsupported deployment format: {manifest['format']}")
    if manifest["quantization"]["method"] != "scaled_fp8_weight_only" or manifest["computation_dtype"] != "bfloat16":
        raise ValueError("Unsupported deployment quantization or computation dtype.")
    with init_weights_on_device():
        dit = WanModel(**manifest["dit_config"])
        for name in manifest["quantization"]["modules"]:
            module = dit.get_submodule(name)
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f"FP8 target is not a Linear: {name}")
            replacement = ScaledFP8Linear(module.in_features, module.out_features, module.bias is not None, device="meta")
            replacement.compute_backend = fp8_compute
            parent_name, _, leaf = name.rpartition(".")
            setattr(dit.get_submodule(parent_name), leaf, replacement)
        image_encoder = WanImageEncoder()
        vae = WanVideoVAE(**manifest["vae_config"])
    # Build BF16 metadata for unquantized tensors without materializing the model.
    for model in (dit, image_encoder, vae):
        for module in model.modules():
            for name, param in list(module.named_parameters(recurse=False)):
                if param.dtype != torch.float8_e4m3fn:
                    module._parameters[name] = torch.nn.Parameter(param.to(torch.bfloat16), requires_grad=False)
    pipe = WanVideoPipeline(device=device, torch_dtype=torch.bfloat16)
    for name, model in (("dit", dit), ("image_encoder", image_encoder), ("vae", vae)):
        print(f"Loading deployment component: {name}", flush=True)
        setattr(pipe, name, load_component(model, manifest["components"][name], directory, verify_hashes))
    embeddings = {}
    for shard in _component_state(manifest["components"]["prompt_embeddings"], directory, verify_hashes=verify_hashes):
        embeddings.update(shard)
    if set(embeddings) != {"positive", "negative"} or any(
        value.shape != (1, 512, 4096) or value.dtype != torch.bfloat16 or not torch.isfinite(value).all()
        for value in embeddings.values()
    ):
        raise ValueError("Invalid cached prompt embeddings.")
    pipe.cached_prompt_embeddings = embeddings
    pipe.cached_prompts = manifest["prompts"]
    pipe.deployment_manifest = manifest
    pipe.fp8_compute = fp8_compute
    pipe.fp8_capability = capability
    pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_param_in_dit)
    return pipe.eval()
