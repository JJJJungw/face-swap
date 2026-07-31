#!/usr/bin/env python3
"""Reproduce the Anime-V2 path used by the public Hugging Face Space."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import re

import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps


SPACE_REPO = "prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast"
BASE_MODEL = "Qwen/Qwen-Image-Edit-2511"
RAPID_MODEL = "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19"
ANIME_REPO = "prithivMLmods/Qwen-Image-Edit-2511-Anime"
ANIME_FILE = "Qwen-Image-Edit-2511-Anime-2000.safetensors"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
NEGATIVE_PROMPT = (
    "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, "
    "extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"
)


def space_size(image):
    width, height = image.size
    if width > height:
        out_width = 1024
        out_height = int(out_width * height / width)
    else:
        out_height = 1024
        out_width = int(out_height * width / height)
    return (out_width // 8) * 8, (out_height // 8) * 8


def memory_snapshot(label):
    meminfo = {}
    with open("/proc/meminfo", encoding="ascii") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            meminfo[key] = int(value.strip().split()[0]) * 1024
    rss = 0
    with open("/proc/self/status", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
                break
    gpu_free, gpu_total = torch.cuda.mem_get_info()
    gib = 1024**3
    print(
        f"[memory:{label}] "
        f"gpu_alloc={torch.cuda.memory_allocated() / gib:.2f}GiB "
        f"gpu_reserved={torch.cuda.memory_reserved() / gib:.2f}GiB "
        f"gpu_free={gpu_free / gib:.2f}/{gpu_total / gib:.2f}GiB "
        f"rss={rss / gib:.2f}GiB "
        f"ram_available={meminfo.get('MemAvailable', 0) / gib:.2f}GiB "
        f"swap_free={meminfo.get('SwapFree', 0) / gib:.2f}GiB",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="official_sample.jpeg")
    parser.add_argument("--out", default="out/space_exact/official_sample_anime_v2.png")
    parser.add_argument("--prompt", default="Transform into anime.")
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        metavar="TAG::PROMPT",
        help=(
            "Run multiple prompts after one model load. Directory inputs only; outputs are "
            "written under <out>/<tag>/target. Repeat for each prompt."
        ),
    )
    parser.add_argument(
        "--style-variant",
        action="append",
        default=None,
        metavar="TAG::SCALE",
        help=(
            "Run multiple Anime LoRA scales after one model load. Directory inputs only; "
            "repeat for each scale."
        ),
    )
    parser.add_argument(
        "--include-file",
        help="Optional text file containing one input basename per line (directory inputs only).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument(
        "--style-scale",
        type=float,
        default=1.0,
        help="Anime LoRA scale for a single batch run.",
    )
    parser.add_argument("--n", type=int, default=0, help="Directory mode image limit (0=all).")
    parser.add_argument(
        "--sample-mode",
        choices=("uniform", "head"),
        default="uniform",
        help="Select directory samples across the full sorted list or only from its start.",
    )
    parser.add_argument(
        "--seed-mode",
        choices=("fixed", "increment"),
        default="fixed",
        help="Use the same seed for every image or add the batch index.",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Keep inactive pipeline modules in system RAM for GPUs below 48 GiB.",
    )
    parser.add_argument(
        "--int8-transformer",
        action="store_true",
        help="Load only the Rapid transformer in bitsandbytes INT8 (MIT licensed).",
    )
    parser.add_argument(
        "--space-revision",
        default="main",
        help="Space git revision. Use a commit hash to freeze an exact version.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip batch outputs that already have both PNG and JSON metadata.",
    )
    args = parser.parse_args()

    if args.cpu_offload and args.int8_transformer:
        parser.error("--cpu-offload and --int8-transformer are mutually exclusive")
    if args.variant and args.style_variant:
        parser.error("--variant and --style-variant cannot be combined")
    if args.style_scale <= 0:
        parser.error("--style-scale must be positive")
    if (args.variant or args.style_variant) and args.style_scale != 1.0:
        parser.error("--style-scale cannot be combined with --variant/--style-variant")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")

    input_path = Path(args.input)
    batch_mode = input_path.is_dir()
    if (args.variant or args.style_variant) and not batch_mode:
        parser.error("--variant/--style-variant requires --input to be a directory")
    if args.include_file and not batch_mode:
        parser.error("--include-file requires --input to be a directory")

    variants = []
    if args.variant:
        seen_tags = set()
        for value in args.variant:
            if "::" not in value:
                parser.error(f"--variant must use TAG::PROMPT: {value}")
            tag, prompt = value.split("::", 1)
            if not re.fullmatch(r"[A-Za-z0-9._-]+", tag):
                parser.error(f"--variant tag must be filesystem-safe: {tag}")
            if tag in seen_tags:
                parser.error(f"duplicate --variant tag: {tag}")
            if not prompt.strip():
                parser.error(f"empty prompt for --variant tag: {tag}")
            seen_tags.add(tag)
            variants.append((tag, prompt.strip(), 1.0))
    elif args.style_variant:
        seen_tags = set()
        for value in args.style_variant:
            if "::" not in value:
                parser.error(f"--style-variant must use TAG::SCALE: {value}")
            tag, scale_text = value.split("::", 1)
            if not re.fullmatch(r"[A-Za-z0-9._-]+", tag):
                parser.error(f"--style-variant tag must be filesystem-safe: {tag}")
            if tag in seen_tags:
                parser.error(f"duplicate --style-variant tag: {tag}")
            try:
                style_scale = float(scale_text)
            except ValueError:
                parser.error(f"invalid --style-variant scale: {scale_text}")
            if style_scale <= 0:
                parser.error(f"--style-variant scale must be positive: {style_scale}")
            seen_tags.add(tag)
            variants.append((tag, args.prompt, style_scale))
    else:
        variants = [(None, args.prompt, args.style_scale)]

    if batch_mode:
        sources = sorted(
            path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        source_count = len(sources)
        if args.include_file:
            include_path = Path(args.include_file)
            requested = [
                line.strip()
                for line in include_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            by_name = {path.name: path for path in sources}
            missing = [name for name in requested if name not in by_name]
            if missing:
                parser.error(
                    f"--include-file entries not found in {input_path}: {', '.join(missing)}"
                )
            sources = [by_name[name] for name in requested]
        selected_count = len(sources)
        if 0 < args.n < selected_count:
            if args.sample_mode == "uniform" and args.n > 1:
                indices = [
                    round(index * (selected_count - 1) / (args.n - 1))
                    for index in range(args.n)
                ]
                sources = [sources[index] for index in indices]
            else:
                sources = sources[: args.n]
        if not sources:
            raise SystemExit(f"No supported images found in {input_path}")
        output_root = Path(args.out)
        if output_root.suffix:
            parser.error("--out must be a directory when --input is a directory")
        input_output_dir = output_root / "input"
        input_output_dir.mkdir(parents=True, exist_ok=True)
        variant_outputs = []
        for tag, prompt, style_scale in variants:
            variant_root = output_root if tag is None else output_root / tag
            target_output_dir = variant_root / "target"
            target_output_dir.mkdir(parents=True, exist_ok=True)
            variant_outputs.append(
                (tag, prompt, style_scale, variant_root, target_output_dir)
            )
        print(
            f"[batch] input={input_path} candidates={source_count} images={len(sources)} "
            f"variants={len(variants)} sample_mode={args.sample_mode} "
            f"seed_mode={args.seed_mode} output={output_root}"
        )
    else:
        if not input_path.is_file():
            raise SystemExit(f"Input does not exist: {input_path}")
        sources = [input_path]
        variant_outputs = [(None, args.prompt, args.style_scale, None, None)]

    # The Space carries patched Qwen pipeline modules outside the pip package.
    space_dir = snapshot_download(
        repo_id=SPACE_REPO,
        repo_type="space",
        revision=args.space_revision,
        allow_patterns=["qwenimage/*.py"],
    )
    sys.path.insert(0, space_dir)

    from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
    from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

    try:
        from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3
    except Exception as exc:
        QwenDoubleStreamAttnProcessorFA3 = None
        fa3_import_error = exc
    else:
        fa3_import_error = None

    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(f"[env] torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
    print(f"[space] repo={SPACE_REPO} revision={args.space_revision} snapshot={space_dir}")
    memory_snapshot("start")

    transformer_kwargs = {"torch_dtype": dtype}
    if args.int8_transformer:
        try:
            from diffusers import BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit(
                "--int8-transformer requires a Diffusers build with BitsAndBytesConfig"
            ) from exc
        transformer_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    if not args.cpu_offload:
        transformer_kwargs["device_map"] = "cuda"
    transformer = QwenImageTransformer2DModel.from_pretrained(RAPID_MODEL, **transformer_kwargs)
    memory_snapshot("transformer-loaded")
    pipeline_kwargs = {}
    if args.int8_transformer:
        pipeline_kwargs["device_map"] = "cuda"
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        BASE_MODEL,
        transformer=transformer,
        torch_dtype=dtype,
        **pipeline_kwargs,
    )
    memory_snapshot("pipeline-loaded")

    if QwenDoubleStreamAttnProcessorFA3 is None:
        attention = f"SDPA fallback (FA3 import failed: {fa3_import_error})"
    else:
        try:
            pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
            if importlib.util.find_spec("kernels") is None:
                attention = "SDPA fallback (kernels package unavailable)"
            else:
                attention = "FA3"
        except Exception as exc:
            attention = f"SDPA fallback ({exc})"
    print(f"[attention] {attention}")

    pipe.load_lora_weights(ANIME_REPO, weight_name=ANIME_FILE, adapter_name="anime-v2")
    pipe.set_adapters(["anime-v2"], adapter_weights=[1.0])
    memory_snapshot("lora-loaded")
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=0)
        print("[memory] model CPU offload enabled")
    elif args.int8_transformer:
        memory_snapshot("pipeline-cuda")
        print("[memory] INT8 transformer pipeline resident on CUDA")
    else:
        pipe = pipe.to(device)
        memory_snapshot("pipeline-cuda")
        print("[memory] full pipeline resident on CUDA")

    records_by_variant = {tag: [] for tag, _, _, _, _ in variant_outputs}
    total_started = time.time()
    total_jobs = len(sources) * len(variant_outputs)
    completed_jobs = 0
    for tag, prompt, style_scale, variant_root, target_output_dir in variant_outputs:
        pipe.set_adapters(["anime-v2"], adapter_weights=[style_scale])
        if tag is not None:
            print(
                f"[variant] tag={tag} style_scale={style_scale:g} prompt={prompt!r}"
            )
        for index, source in enumerate(sources):
            if batch_mode:
                output = target_output_dir / f"{source.stem}.png"
            else:
                output = Path(args.out)
            meta_path = output.with_suffix(".json")
            if batch_mode and args.resume and output.is_file() and meta_path.is_file():
                if not (input_output_dir / source.name).exists():
                    shutil.copy2(source, input_output_dir / source.name)
                try:
                    with meta_path.open(encoding="utf-8") as handle:
                        metadata = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    print(f"[resume-invalid] regenerating {output}")
                else:
                    completed_jobs += 1
                    records_by_variant[tag].append(metadata)
                    print(f"[resume {completed_jobs}/{total_jobs}] {output}")
                    continue
            image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
            width, height = space_size(image)
            seed = args.seed if args.seed_mode == "fixed" else args.seed + index
            generator = torch.Generator(device=device).manual_seed(seed)
            completed_jobs += 1
            print(
                f"[run {completed_jobs}/{total_jobs}] input={source} output_size={width}x{height} "
                f"seed={seed} steps={args.steps} cfg={args.cfg} prompt={prompt!r}"
            )

            started = time.time()
            result = pipe(
                image=[image],
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                height=height,
                width=width,
                num_inference_steps=args.steps,
                generator=generator,
                true_cfg_scale=args.cfg,
            ).images[0]

            output.parent.mkdir(parents=True, exist_ok=True)
            result.save(output)
            if batch_mode and not (input_output_dir / source.name).exists():
                shutil.copy2(source, input_output_dir / source.name)
            with output.open("rb") as output_file:
                digest = hashlib.sha256(output_file.read()).hexdigest()
            metadata = {
                "input": str(source),
                "output": str(output),
                "variant": tag,
                "prompt": prompt,
                "style_scale": style_scale,
                "negative_prompt": NEGATIVE_PROMPT,
                "seed": seed,
                "seed_mode": args.seed_mode,
                "steps": args.steps,
                "true_cfg_scale": args.cfg,
                "size": [width, height],
                "space_repo": SPACE_REPO,
                "space_revision": args.space_revision,
                "space_commit": os.path.basename(space_dir),
                "base_model": BASE_MODEL,
                "transformer": RAPID_MODEL,
                "adapter": f"{ANIME_REPO}/{ANIME_FILE}",
                "attention": attention,
                "cpu_offload": args.cpu_offload,
                "int8_transformer": args.int8_transformer,
                "seconds": round(time.time() - started, 3),
                "sha256": digest,
            }
            with meta_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=True, indent=2)
            records_by_variant[tag].append(metadata)
            print(
                f"[done {completed_jobs}/{total_jobs}] {output} "
                f"({metadata['seconds']:.1f}s) sha256={digest}"
            )

    if batch_mode:
        for tag, _, _, variant_root, _ in variant_outputs:
            manifest_path = variant_root / "manifest.jsonl"
            with manifest_path.open("w", encoding="utf-8") as handle:
                for record in records_by_variant[tag]:
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            print(f"[manifest] tag={tag or 'default'} path={manifest_path}")
        print(
            f"[batch-done] images={len(sources)} variants={len(variant_outputs)} "
            f"runs={total_jobs} seconds={time.time() - total_started:.1f}"
        )
    else:
        record = records_by_variant[None][0]
        print(f"[meta] {Path(record['output']).with_suffix('.json')}")


if __name__ == "__main__":
    main()
