"""Run FixAnything on a rendered video (a folder of frames or a video file).

Example:
    python scripts/run_inference.py --input examples/dl3dv_3dgs/input.mp4 --output_dir outputs/dl3dv_3dgs

`load_pipeline()` and `fix_video()` below are the whole inference API.
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import torch
from diffsynth.utils import ModelConfig

from fixanything.pipelines import WanVideoPipeline
from fixanything.data import load_frames, save_video, crop_and_resize, side_by_side
from fixanything.prompts import DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT

# Base model (same layout on ModelScope and Hugging Face).
WAN_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
WAN_MODEL_FILES = [
    "diffusion_pytorch_model*.safetensors",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
]
WAN_TOKENIZER_FILES = "google/*"


def _model_config(model_dir, pattern, download_source="huggingface", **kwargs):
    # Skip the remote download check when the files are already there (so this also works offline).
    present = len(glob.glob(os.path.join(model_dir, WAN_MODEL_ID, pattern))) > 0
    return ModelConfig(
        model_id=WAN_MODEL_ID, origin_file_pattern=pattern, local_model_path=model_dir,
        download_resource=download_source, skip_download=present, **kwargs,
    )


def load_pipeline(lora_path, model_dir="checkpoints", device="cuda", torch_dtype=torch.bfloat16,
                  lora_alpha=1.0, download_source="huggingface",
                  num_persistent_param_in_dit=None):
    """Build the Wan2.1-I2V-14B pipeline, apply the FixAnything LoRA and enable VRAM offloading.

    Base-model files are looked up in `<model_dir>/Wan-AI/Wan2.1-I2V-14B-480P/` and downloaded
    from `download_source` ("huggingface" or "modelscope") if missing.
    Six billion DiT parameters stay on the GPU by default; the remaining layers
    are streamed from CPU memory. Pass zero to stream all wrapped DiT layers.
    """
    if num_persistent_param_in_dit is None:
        num_persistent_param_in_dit = 6_000_000_000
    if num_persistent_param_in_dit < 0:
        raise ValueError("num_persistent_param_in_dit must be nonnegative.")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"FixAnything LoRA not found: {lora_path}")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype, device=device,
        model_configs=[_model_config(model_dir, p, download_source, offload_device="cpu") for p in WAN_MODEL_FILES],
        tokenizer_config=_model_config(model_dir, WAN_TOKENIZER_FILES, download_source),
        redirect_common_files=False,
    )
    pipe.load_lora(pipe.dit, lora_path, alpha=lora_alpha)
    pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_param_in_dit)
    return pipe


def fix_video(
    pipe, frames, clean_frame_indices=None,
    num_frames=61, num_repeat_last=4, height=480, width=832,
    seed=1, num_inference_steps=10, cfg_scale=5.0,
    prompt=None, negative_prompt=None, tiled=True,
):
    """Refine a rendered video.

    Args:
        frames: list of PIL images (the degraded rendering); the first `num_frames` are used.
        clean_frame_indices: indices of input frames that are already clean (e.g. rendered at the input
            views); the model keeps them and refines the rest. List of ints or whitespace-separated
            string. Default: first and last frame. Pass [] to refine every frame.
        prompt, negative_prompt: use the deployment's cached prompts when omitted,
            or the original defaults for a pipeline with a text encoder.
    The model runs on `num_frames + num_repeat_last` frames (the last frame is repeated, as in training)
    and the output is cropped back to `num_frames`.
    Returns:
        (generated_frames, input_frames): two lists of `num_frames` PIL images of size (width, height).
    """
    cached_prompts = getattr(pipe, "cached_prompts", None)
    if isinstance(cached_prompts, dict):
        default_prompt = cached_prompts.get("positive", DEFAULT_PROMPT)
        default_negative_prompt = cached_prompts.get("negative", DEFAULT_NEGATIVE_PROMPT)
    elif isinstance(cached_prompts, (tuple, list)) and len(cached_prompts) == 2:
        default_prompt, default_negative_prompt = cached_prompts
    else:
        default_prompt, default_negative_prompt = DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT
    if prompt is None:
        prompt = default_prompt
    if negative_prompt is None:
        negative_prompt = default_negative_prompt
    if len(frames) < num_frames:
        raise ValueError(f"Need {num_frames} input frames, got {len(frames)}.")
    if len(frames) > num_frames:
        print(f"[fix_video] Using the first {num_frames} of {len(frames)} frames.")
    frames = [crop_and_resize(f, height, width) for f in frames[:num_frames]]
    ref_video = frames + [frames[-1]] * num_repeat_last

    if clean_frame_indices is None:
        clean_frame_indices = [0, num_frames - 1]
    elif isinstance(clean_frame_indices, str):
        clean_frame_indices = [int(x) for x in clean_frame_indices.split()]
    clean = set(clean_frame_indices)
    if num_frames - 1 in clean:   # the repeats of the last frame inherit its flag
        clean.update(range(num_frames, num_frames + num_repeat_last))

    gen_video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=ref_video[0],
        reference_video=ref_video,
        clean_frame_indices=sorted(clean),
        num_frames=num_frames + num_repeat_last, height=height, width=width,
        seed=seed, tiled=tiled,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
    )
    return gen_video[:num_frames], ref_video[:num_frames]


def _nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def parse_args():
    p = argparse.ArgumentParser(description="FixAnything inference")
    p.add_argument("--input", type=str, required=True,
                   help="Rendered video file, or a folder of frames.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--deployment_dir", type=str, default=None,
                   help="Load an exported compact FP8 package instead of the base model and LoRA.")
    p.add_argument("--verify_deployment_hashes", action="store_true",
                   help="Verify every deployment shard's SHA-256 before use (adds file-reading time).")
    p.add_argument("--fp8_compute", choices=("bf16", "native"), default="bf16",
                   help="Compact package Linear backend: dequantize to BF16 (default), or use native FP8 GEMM with dynamically quantized activations.")
    p.add_argument("--lora_path", type=str, default="checkpoints/fixanything_lora.safetensors")
    p.add_argument("--model_dir", type=str, default="checkpoints",
                   help="Folder containing Wan-AI/Wan2.1-I2V-14B-480P/ (downloaded there if missing).")
    p.add_argument("--clean_frame_indices", type=str, default=None,
                   help='Space-separated indices of input frames to keep as-is (default: first and last, "0 60"). '
                        'Use "" to refine every frame.')
    p.add_argument("--num_frames", type=int, default=61, help="Input frames used (the first ones).")
    p.add_argument("--num_repeat_last", type=int, default=4,
                   help="The last frame is repeated this many times for the model (as in training); the output is cropped back.")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--num_persistent_param_in_dit", type=_nonnegative_int, default=None,
                   help="DiT parameters kept on GPU; defaults to 6 billion for BF16 or 12 billion for a compact FP8 package. Use 0 to stream all wrapped layers.")
    p.add_argument("--profile", action="store_true",
                   help="Save loading/inference timings and CUDA peak memory to run_metrics.json.")
    p.add_argument("--save_frames", action="store_true",
                   help="Also save lossless generated/input PNG frames for quality comparisons (outside inference timing).")
    p.add_argument("--warmup_steps", type=_nonnegative_int, default=0,
                   help="Run a separate warmup before measured inference, e.g. 1 to compile native FP8 kernels. Defaults to 0.")
    args = p.parse_args()
    if args.verify_deployment_hashes and not args.deployment_dir:
        p.error("--verify_deployment_hashes requires --deployment_dir")
    if args.fp8_compute == "native" and not args.deployment_dir:
        p.error("--fp8_compute native requires --deployment_dir")
    return args


def main():
    args = parse_args()
    frames, _ = load_frames(args.input, args.height, args.width, num_frames=args.num_frames)
    print(f"Loaded {len(frames)} frames from {args.input}")
    clean = None if args.clean_frame_indices is None else [int(x) for x in args.clean_frame_indices.split()]
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_frames:
        for label in ("generated_frames", "input_frames"):
            directory = Path(args.output_dir) / label
            if directory.exists() and any(directory.iterdir()):
                raise FileExistsError(f"Frame output directory must be empty: {directory}")

    persistent_params = args.num_persistent_param_in_dit
    if persistent_params is None:
        persistent_params = 12_000_000_000 if args.deployment_dir else 6_000_000_000
    profile_cuda = args.profile and torch.cuda.is_available()
    metrics = {
        "arguments": vars(args),
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "dit_storage_dtype": "float8_e4m3fn" if args.deployment_dir else "bfloat16",
        "fp8_compute": args.fp8_compute if args.deployment_dir else None,
        "num_persistent_param_in_dit": persistent_params,
    }

    def start_measurement():
        if profile_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        return time.perf_counter()

    def finish_measurement(start):
        if profile_cuda:
            torch.cuda.synchronize()
        result = {"seconds": time.perf_counter() - start}
        if profile_cuda:
            result.update(
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            )
        return result

    def save_metrics():
        if args.profile:
            with open(os.path.join(args.output_dir, "run_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

    if profile_cuda:
        metrics["gpu"] = torch.cuda.get_device_name()
        metrics["gpu_total_gib"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    start = start_measurement()
    if args.deployment_dir:
        from fixanything.deployment import load_deployment
        pipe = load_deployment(
            args.deployment_dir, device="cuda",
            num_persistent_param_in_dit=persistent_params,
            verify_hashes=args.verify_deployment_hashes,
            fp8_compute=args.fp8_compute,
        )
    else:
        pipe = load_pipeline(
            args.lora_path, args.model_dir,
            num_persistent_param_in_dit=persistent_params,
        )
    metrics["load"] = finish_measurement(start)
    if args.deployment_dir:
        metrics["fp8_capability"] = pipe.fp8_capability
    save_metrics()
    if args.profile:
        print(f"Model loading: {metrics['load']}")
    def infer(steps):
        return fix_video(
            pipe, frames, clean_frame_indices=clean,
            num_frames=args.num_frames, num_repeat_last=args.num_repeat_last,
            height=args.height, width=args.width,
            seed=args.seed, num_inference_steps=steps,
        )

    if args.warmup_steps:
        start = start_measurement()
        infer(args.warmup_steps)
        metrics["warmup"] = finish_measurement(start)
        if args.deployment_dir:
            from fixanything.fp8 import reset_fp8_compute_stats
            reset_fp8_compute_stats(pipe.dit)
        save_metrics()
        if args.profile:
            print(f"Warmup: {metrics['warmup']}")
    start = start_measurement()
    gen_video, ref_video = infer(args.num_inference_steps)
    metrics["inference"] = finish_measurement(start)
    if args.deployment_dir:
        from fixanything.fp8 import fp8_compute_stats
        metrics["fp8_compute_stats"] = fp8_compute_stats(pipe.dit)
    save_metrics()

    if args.save_frames:
        for label, video in (("generated_frames", gen_video), ("input_frames", ref_video)):
            directory = Path(args.output_dir) / label
            directory.mkdir(parents=True, exist_ok=True)
            for index, frame in enumerate(video):
                frame.save(directory / f"{index:05d}.png")
    save_video(gen_video, os.path.join(args.output_dir, "generated.mp4"), fps=args.fps)
    save_video(ref_video, os.path.join(args.output_dir, "input.mp4"), fps=args.fps)
    save_video(side_by_side(ref_video, gen_video), os.path.join(args.output_dir, "side_by_side.mp4"), fps=args.fps)
    if args.profile:
        print(f"Inference: {metrics['inference']}")
    print(f"Saved results to {args.output_dir}/ (generated.mp4, input.mp4, side_by_side.mp4 = input | generated)")


if __name__ == "__main__":
    main()
