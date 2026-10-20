"""Merge the local FixAnything LoRA and export a self-contained scaled FP8 package."""
import argparse

from fixanything.deployment import export_deployment
from fixanything.prompts import DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default="checkpoints")
    parser.add_argument("--lora_path", default="checkpoints/fixanything_lora.safetensors")
    parser.add_argument("--output_dir", required=True, help="A new directory for the deployment package.")
    parser.add_argument("--device", default="cuda", help="Device used for offline LoRA merging and text encoding.")
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--max_shard_gb", type=float, default=2.0)
    args = parser.parse_args()
    if args.max_shard_gb <= 0:
        parser.error("--max_shard_gb must be positive")
    export_deployment(
        args.output_dir, model_dir=args.model_dir, lora_path=args.lora_path,
        device=args.device, lora_alpha=args.lora_alpha, prompt=args.prompt,
        negative_prompt=args.negative_prompt, max_shard_bytes=int(args.max_shard_gb * 1e9),
    )


if __name__ == "__main__":
    main()
