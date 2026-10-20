<div align="center">

# FixAnything: 3D-Consistent Rendering Refinement via Video Generative Priors

[Khiem Vuong](https://www.khiemvuong.com/), [Deva Ramanan*](https://www.cs.cmu.edu/~deva), [Srinivasa Narasimhan*](https://www.cs.cmu.edu/~srinivas)

**ECCV 2026**

[[`arXiv`](https://arxiv.org/abs/2608.23549)]
[[`Project Page`](https://fix-anything.github.io/)]
[[`Model (HF)`](https://huggingface.co/kvuong2711/fix-anything)]
[[`Bibtex`](#-citation)]

</div>

<p align="center">
  <img src="assets/teaser.png" width="100%">
</p>

## 📖 Overview

FixAnything is a single generalist video model that repairs rendering artifacts from **any** 3D representation — 3DGS, NeRF, meshes, or sparse point clouds — by repurposing a pretrained video diffusion model with minimal modification and finetuning. 

This repository contains the inference code.

## Table of Contents

- [Installation](#-installation)
- [Checkpoints](#-checkpoints)
- [Run Inference on a Rendered Video](#-run-inference-on-a-rendered-video)
- [Compact FP8 Deployment](#compact-fp8-deployment)
- [Run on Your Own Captures](#-run-on-your-own-captures)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Issues](#-issues)
- [Citation](#-citation)

## 🔧 Installation

```bash
git clone https://github.com/kvuong2711/fix-anything.git
cd fix-anything

conda create -n fixanything python=3.10 -y
conda activate fixanything

# PyTorch for your CUDA version
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126

# FixAnything + the pinned DiffSynth-Studio commit it builds on
# (DiffSynth's setup.py needs pkg_resources, hence setuptools<82 and --no-build-isolation)
pip install "setuptools<82" wheel
pip install --no-build-isolation -e .

# Optional: MapAnything + pyrender, to reconstruct and render your own captures (see below)
pip install --no-build-isolation -e ".[mapanything]"

# Optional but recommended: FlashAttention-2 (used automatically when installed; prebuilt wheel for torch 2.6 / CUDA 12 / Python 3.10)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

## 📦 Checkpoints

`scripts/download_models.py` downloads everything into `checkpoints/`: the Wan2.1-I2V-14B-480P base model (~82 GB, from Hugging Face or `--source modelscope`) and the FixAnything LoRA from [kvuong2711/fix-anything](https://huggingface.co/kvuong2711/fix-anything):

```bash
python scripts/download_models.py --model_dir checkpoints
```

The expected layout is:

```
checkpoints/
├── Wan-AI/Wan2.1-I2V-14B-480P/
│   ├── diffusion_pytorch_model-0000?-of-00007.safetensors
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   ├── Wan2.1_VAE.pth
│   ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
│   └── google/umt5-xxl/
└── fixanything_lora.safetensors
```

If you already have the Wan2.1 files elsewhere, point `--model_dir` to the folder that contains `Wan-AI/`.

## 🚀 Run Inference on a Rendered Video

`run_inference.py` refines a rendered video (a video file, or a folder of frames). `examples/` contains one DL3DV clip rendered by 3D Gaussian Splatting and by a sparse point-track renderer:

```bash
# 3DGS rendering
python scripts/run_inference.py --input examples/dl3dv_3dgs/input.mp4   --output_dir outputs/dl3dv_3dgs

# sparse point-track rendering
python scripts/run_inference.py --input examples/dl3dv_tracks/input.mp4 --output_dir outputs/dl3dv_tracks
```

Each run writes `generated.mp4` (refined video), `input.mp4` (the resized input) and `side_by_side.mp4` (input | refined) to the output folder.

To run on your own rendering (any 3D representation), pass a video of 61 frames rendered along a camera path, ideally starting and ending at views used to build the representation. Frames are resized to 832×480.

- **Caveat on the number of frames.** Internally the last frame is repeated 4 times (65 frames): the Wan VAE encodes the first frame on its own but compresses the rest temporally, so this (hopefully) gives the last frame more influence to the generation. The output is cropped back to 61 frames.
- **`--clean_frame_indices`** lists the frames to keep as-is; by default the first and last frames (`"0 60"`).

## Compact FP8 deployment

Export a compact package once, then use that directory for later inference. The exporter merges the FixAnything LoRA into the base DiT before quantization and includes the VAE, image encoder, and cached positive/negative prompt embeddings. Inference from this package does not need the original base-model files, LoRA, tokenizer, or UMT5 text encoder.

Use the installed FixAnything environment for both commands. The export step requires the original checkpoints and sufficient CPU memory. Choose a new output directory for each build:

```bash
python scripts/export_deployment.py --output_dir deployments/fixanything_fp8
```

Once export finishes, run directly from the package, including offline:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/run_inference.py \
  --deployment_dir deployments/fixanything_fp8 \
  --input examples/dl3dv_3dgs/input.mp4 \
  --output_dir outputs/fp8_deployment --profile
```

The compact package stores DiT Linear weights in E4M3 FP8 with one FP32 scale per Linear layer. Biases, normalization parameters, and convolution weights remain BF16. By default, each Linear weight is transferred in FP8, dequantized, and computed in BF16; this reduces weight storage and CPU transfer volume.

To use native FP8 matrix multiplication on the same package, add `--fp8_compute native`. No new export or weight download is needed:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/run_inference.py \
  --deployment_dir deployments/fixanything_fp8 --fp8_compute native \
  --input examples/dl3dv_3dgs/input.mp4 \
  --output_dir outputs/native_fp8_use --profile
```

The native backend dynamically scales each Linear input tensor into E4M3 FP8 and calls `torch._scaled_mm` with FP8 inputs/weights, FP32 scales, BF16 outputs, and `use_fast_accum=False`. Attention, normalization, CLIP, and VAE retain their existing precision. Activation quantization uses fused Triton kernels to avoid full FP32 activation allocations. The tested environment is RTX 4090 (SM89), PyTorch 2.6.0+cu126, and Triton 3.2.0. A CUDA kernel probe runs before checkpoint loading; unsupported native configurations fail explicitly. Linear layers with unaligned K/N dimensions use a counted BF16 fallback, and kernel errors are propagated. `run_metrics.json` records native/BF16 call counts and fallback reasons.

For steady-state comparisons, use `--warmup_steps 1` in **every** mode. This discards a separate one-step inference to warm kernels at the actual input shape, records its time and CUDA peaks under `warmup`, then resets counters/peaks before the requested inference. The default is no warmup; first-use costs then occur in loading (including the native capability probe) and inference. `--save_frames` also writes lossless PNG sequences for quality comparisons, outside inference timing, and requires empty frame output directories.

The default GPU residency budget is 12 billion DiT parameters for the compact package and 6 billion for the original BF16 model. Override it with `--num_persistent_param_in_dit`; `0` streams all wrapped DiT layers from CPU memory. Higher values can reduce transfers but leave less memory for activations. `--profile` writes loading/inference timings and CUDA peak allocated/reserved memory to `run_metrics.json` in the output directory.

The package supports the positive and negative prompts cached during export. `fix_video()` automatically uses those prompts when no explicit prompts are supplied. To change the text conditioning, export a new package with the desired prompts; an explicit mismatched prompt is rejected instead of silently reusing a different embedding. Input videos, clean-frame indices, frame counts, seeds, and denoising steps remain runtime inputs.

The exported package occupies approximately **17.93 GB (16.70 GiB)**, including 12 safetensors shards, the manifest, and license notices. Copy the entire directory to the deployment machine. Add `--verify_deployment_hashes` on first use to check all SHA-256 hashes. Python dependencies are separate from the weight package. Quantization can change generated details; validate it on your own inputs.

Frame dimensions must be divisible by 16, and `num_frames + num_repeat_last` must equal 1 modulo 4 (the default is 61 + 4). Changing the cached prompts or merged LoRA requires exporting a new package.

A local RTX 4090 comparison used the same 3DGS example, 832×480 resolution, 61 input frames (65 internally), 10 steps, CFG 5, seed 1, PyTorch 2.6.0+cu126 and FlashAttention 2.7.4.post1. Each mode ran twice in separate processes, in interleaved order, after a discarded one-step warmup:

| Mode | Inference time, both runs | CUDA peak allocated | Sampled device peak | Process peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Original BF16, 6 billion resident parameters | 257.4 / 272.6 s | 14.52 GiB | 15.53 GiB | 55.86 GiB |
| Compact FP8, BF16 compute, 12 billion resident parameters | 208.1 / 207.7 s | 14.54 GiB | 15.55 GiB | 19.00 GiB |
| Compact FP8, native FP8 compute, 12 billion resident parameters | 180.7 / 180.6 s | 14.54 GiB | 15.77 GiB | 18.86 GiB |

Native FP8 reduced median inference time by **13.1%** relative to the same compact package with BF16 computation (1.15× throughput). The original BF16 comparison also includes prompt caching and a different parameter residency count; it does not isolate GEMM precision. The native runs executed 9,760 FP8 Linear calls each across 488 modules, with no BF16 fallback. A CUDA profiler trace identified an `sm89_xmma_gemm_e4m3...` cuBLAS kernel.

Inference includes conditioning, denoising and VAE decoding; loading, warmup, PNG writing and MP4 encoding are excluded. CUDA inference peaks were reset after warmup. Device memory was sampled once per second across the complete process; the lowest sampled free memory for native FP8 was 7.75 GiB. File caches may be warm; these are single-clip measurements, not a cold-start guarantee. Both compact modes completed offline from empty working directories with deliberately nonexistent original checkpoint paths.

Quality metrics use full-resolution lossless PNGs and exclude the two clean anchor frames (59 repaired frames). BF16 output is a comparison reference, **not ground truth**:

| Compared with original BF16 | Pooled PSNR | Mean RGB SSIM |
| --- | ---: | ---: |
| Compact FP8 weights, BF16 compute | 33.322 dB | 0.96859 |
| Compact FP8 weights, native FP8 compute | 28.932 dB | 0.92775 |

Each mode reproduced all 61 output frames byte-for-byte across its two runs. Native FP8 retained the main scene layout but produced more visible changes in product and fabric details. These scores measure agreement with BF16, not absolute rendering quality or proven temporal consistency. BF16 computation remains the default; native FP8 is an explicit speed/precision choice. Machine-readable settings, source hashes and results are recorded in [the benchmark report](docs/benchmarks/rtx4090_native_fp8.json). Full local videos, frames and samples are under `outputs/native_fp8/` (gitignored).

The runtime reuses immutable CPU copies of quantized weights during offloading. Do not modify deployment weights in place; merge LoRA into the BF16 originals and export a new package instead.

CPU-only format and numerical tests: `python -m unittest discover -s tests -v`.

Native CUDA kernel tests (no production weights required): `FIXANYTHING_TEST_NATIVE_CUDA=1 python -m unittest discover -s tests -p 'test_native_fp8.py' -v`.

## 📷 Run on Your Own Captures

If you only have a few photos of a scene, `scripts/run_mapanything.py` reconstructs them with [MapAnything](https://github.com/facebookresearch/map-anything), renders the reconstruction along a camera path from the first to the last image, and FixAnything turns that rendering into a clean video. `examples/microkitchen_2views` and `examples/chair_2views` each contain two photos of a scene (credit: thanks to Peter Hedman for the photos!):

```bash
# 1) MapAnything reconstruction + rendering (61 frames; first/last frame = the input photos)
python scripts/run_mapanything.py --images examples/microkitchen_2views/images --output_dir outputs/microkitchen_2views

# 2) FixAnything
python scripts/run_inference.py --input outputs/microkitchen_2views/rendered.mp4 --output_dir outputs/microkitchen_2views
```

Step 1 writes `rendered.mp4` and `reconstruction.glb`; step 2 adds `generated.mp4` (the result) and `side_by_side.mp4`.

- `--trajectory`: `interpolate` (default) moves the camera on a straight path between the two input views; `push_in` additionally dollies toward the scene mid-path and back.
- `--render`: render the reconstruction as a triangle `mesh` (default) or as a point cloud (`pcd`).

Example on `examples/microkitchen_2views` with `--trajectory push_in`:

<p align="center">
  <img src="assets/microkitchen_push_in.gif" width="100%">
  <br>
  <em>MapAnything rendering (left), FixAnything output (right).</em>
</p>

## 📄 License

The code and the FixAnything weights are released under the [Apache 2.0 License](LICENSE), the license of the underlying [Wan2.1](https://github.com/Wan-Video/Wan2.1) model.

## 🙏 Acknowledgments

This codebase builds upon many excellent open-source projects, such as [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [Wan2.1](https://github.com/Wan-Video/Wan2.1), [MapAnything](https://github.com/facebookresearch/map-anything), [gsplat](https://github.com/nerfstudio-project/gsplat), [DL3DV](https://dl3dv-10k.github.io/DL3DV-10K/), etc. We thank the respective authors for making their work publicly available.

## 🐛 Issues

If you have any problem/question/suggestion, feel free to create an issue or reach out directly to me via [email](mailto:kvuong@andrew.cmu.edu).

## 📝 Citation

```bibtex
@inproceedings{vuong2026fixanything,
  title     = {FixAnything: 3D-Consistent Rendering Refinement via Video Generative Priors},
  author    = {Vuong, Khiem and Ramanan, Deva and Narasimhan, Srinivasa},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}
```
