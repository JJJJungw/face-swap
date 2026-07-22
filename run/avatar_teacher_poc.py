#!/usr/bin/env python3
"""Prompt-only teacher dataset generator for 3D avatar face anonymization.

This script turns realistic face crops into stylized 3D avatar targets with a
high-quality teacher model. It is intended for EC2/L4 PoC runs before training a
fast student model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_PROMPT = (
    "original generic stylized 3D animated avatar face, non-photorealistic, "
    "large expressive eyes, soft rounded cheeks, small simplified nose, "
    "smooth clean skin, gentle toon shading, detailed stylized hair, "
    "friendly neutral expression, face crop portrait, soft studio lighting, "
    "high quality 3D render, clean background"
)

DEFAULT_NEGATIVE = (
    "Disney, Pixar, celebrity, famous person, real person, photorealistic, "
    "realistic skin pores, anime, manga, oil painting, sketch, low quality, "
    "uncanny, distorted eyes, asymmetrical face, extra teeth, changed gender, "
    "changed age, watermark, logo, text"
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTS else []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def resize_to_multiple(img: Image.Image, size: int, multiple: int = 16) -> Image.Image:
    from PIL import Image

    w, h = img.size
    scale = size / max(w, h)
    nw = max(256, int(round(w * scale / multiple)) * multiple)
    nh = max(256, int(round(h * scale / multiple)) * multiple)
    return img.resize((nw, nh), Image.LANCZOS)


def stable_seed(path: Path, base_seed: int, mode: str, index: int) -> int:
    if mode == "fixed":
        return base_seed
    if mode == "incremental":
        return base_seed + index
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
    return base_seed + (int(digest[:8], 16) % 1_000_000)


def safe_stem(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{path.stem}_{digest}"


def load_flux_pipe(repo: str, quantization: str):
    import torch
    from diffusers import BitsAndBytesConfig as DiffusersBnb
    from diffusers import FluxImg2ImgPipeline, FluxTransformer2DModel
    from transformers import BitsAndBytesConfig as HFBnb
    from transformers import T5EncoderModel

    if quantization == "nf4":
        nf4 = DiffusersBnb(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            repo,
            subfolder="transformer",
            quantization_config=nf4,
            torch_dtype=torch.bfloat16,
        )
        nf4h = HFBnb(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            repo,
            subfolder="text_encoder_2",
            quantization_config=nf4h,
            torch_dtype=torch.bfloat16,
        )
        pipe = FluxImg2ImgPipeline.from_pretrained(
            repo,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16,
        )
        pipe.vae.to("cuda")
        pipe.text_encoder.to("cuda")
    else:
        pipe = FluxImg2ImgPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
        pipe.to("cuda")

    pipe.enable_vae_tiling()
    return pipe


def load_sdxl_turbo_pipe(repo: str):
    import torch
    from diffusers import AutoPipelineForImage2Image

    pipe = AutoPipelineForImage2Image.from_pretrained(
        repo,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")
    pipe.enable_vae_tiling()
    return pipe


def generate_one(
    pipe,
    model: str,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    strength: float,
    steps: int,
    seed: int,
    guidance_scale: float,
) -> Image.Image:
    import torch

    generator = torch.Generator("cpu").manual_seed(seed)
    if model == "flux-schnell":
        # FLUX schnell is distilled. guidance_scale is intentionally zero.
        run_steps = max(steps, int(steps / max(strength, 0.05)) + 1)
        return pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=0.0,
            num_inference_steps=run_steps,
            generator=generator,
        ).images[0]

    # SDXL Turbo img2img normally works with 1-4 denoising steps and guidance 0.
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate prompt-only 3D avatar teacher targets from face crops."
    )
    parser.add_argument("--input", required=True, help="Input face image or directory.")
    parser.add_argument("--outdir", default="out/avatar_teacher_poc")
    parser.add_argument(
        "--model",
        choices=("flux-schnell", "sdxl-turbo"),
        default="flux-schnell",
    )
    parser.add_argument("--repo", default=None, help="Override Hugging Face repo id.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--strength", type=float, default=0.58)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--seed-mode",
        choices=("hash", "fixed", "incremental"),
        default="hash",
        help="hash is deterministic per file; fixed is useful for one track/person.",
    )
    parser.add_argument(
        "--quantization",
        choices=("nf4", "none"),
        default="nf4",
        help="FLUX only. nf4 is recommended for a single L4.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--copy-input", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    input_out = outdir / "input"
    target_out = outdir / "target"
    manifest_path = outdir / "manifest.jsonl"
    images = list_images(input_path)
    if args.limit:
        images = images[: args.limit]

    if not images:
        raise SystemExit(f"No images found: {input_path}")

    print(f"images: {len(images)}")
    print(f"outdir: {outdir}")
    if args.dry_run:
        for p in images[:20]:
            print(p)
        return

    from PIL import Image

    input_out.mkdir(parents=True, exist_ok=True)
    target_out.mkdir(parents=True, exist_ok=True)

    repo = args.repo
    if repo is None:
        repo = (
            "black-forest-labs/FLUX.1-schnell"
            if args.model == "flux-schnell"
            else "stabilityai/sdxl-turbo"
        )

    print(f"loading {args.model}: {repo}")
    t0 = time.perf_counter()
    if args.model == "flux-schnell":
        pipe = load_flux_pipe(repo, args.quantization)
    else:
        pipe = load_sdxl_turbo_pipe(repo)
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    with manifest_path.open("a", encoding="utf-8") as manifest:
        start = time.perf_counter()
        for index, src in enumerate(images):
            stem = safe_stem(src)
            input_file = input_out / f"{stem}.png"
            target_file = target_out / f"{stem}.png"
            if args.skip_existing and target_file.exists():
                print(f"skip existing: {target_file.name}")
                continue

            image = Image.open(src).convert("RGB")
            prepared = resize_to_multiple(image, args.size)
            if args.copy_input:
                prepared.save(input_file)
            else:
                input_file = src

            seed = stable_seed(src, args.seed, args.seed_mode, index)
            generated = generate_one(
                pipe=pipe,
                model=args.model,
                image=prepared,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                strength=args.strength,
                steps=args.steps,
                seed=seed,
                guidance_scale=args.guidance_scale,
            )
            generated.save(target_file)

            record = {
                "source": str(src),
                "input": str(input_file),
                "target": str(target_file),
                "model": args.model,
                "repo": repo,
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "size": args.size,
                "strength": args.strength,
                "steps": args.steps,
                "seed": seed,
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()
            avg = (time.perf_counter() - start) / (index + 1)
            print(f"{index + 1}/{len(images)} saved {target_file.name}  {avg:.1f}s/img")

    prompt_file = outdir / "prompt.json"
    prompt_file.write_text(
        json.dumps(
            {
                "model": args.model,
                "repo": repo,
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "size": args.size,
                "strength": args.strength,
                "steps": args.steps,
                "seed": args.seed,
                "seed_mode": args.seed_mode,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.copy_input:
        readme = outdir / "README.txt"
        readme.write_text(
            "input/ contains resized realistic face crops.\n"
            "target/ contains prompt-only teacher avatar targets.\n"
            "manifest.jsonl maps each input to its target and generation settings.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
