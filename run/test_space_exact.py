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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.0)
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
    args = parser.parse_args()

    if args.cpu_offload and args.int8_transformer:
        parser.error("--cpu-offload and --int8-transformer are mutually exclusive")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")

    input_path = Path(args.input)
    batch_mode = input_path.is_dir()
    if batch_mode:
        sources = sorted(
            path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        source_count = len(sources)
        if 0 < args.n < source_count:
            if args.sample_mode == "uniform" and args.n > 1:
                indices = [round(index * (source_count - 1) / (args.n - 1)) for index in range(args.n)]
                sources = [sources[index] for index in indices]
            else:
                sources = sources[: args.n]
        if not sources:
            raise SystemExit(f"No supported images found in {input_path}")
        output_root = Path(args.out)
        if output_root.suffix:
            parser.error("--out must be a directory when --input is a directory")
        input_output_dir = output_root / "input"
        target_output_dir = output_root / "target"
        input_output_dir.mkdir(parents=True, exist_ok=True)
        target_output_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(source, target_output_dir / f"{source.stem}.png") for source in sources]
        print(
            f"[batch] input={input_path} candidates={source_count} images={len(jobs)} "
            f"sample_mode={args.sample_mode} seed_mode={args.seed_mode} output={output_root}"
        )
    else:
        if not input_path.is_file():
            raise SystemExit(f"Input does not exist: {input_path}")
        jobs = [(input_path, Path(args.out))]

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

    records = []
    total_started = time.time()
    for index, (source, output) in enumerate(jobs):
        image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
        width, height = space_size(image)
        seed = args.seed if args.seed_mode == "fixed" else args.seed + index
        generator = torch.Generator(device=device).manual_seed(seed)
        print(
            f"[run {index + 1}/{len(jobs)}] input={source} output_size={width}x{height} "
            f"seed={seed} steps={args.steps} cfg={args.cfg} prompt={args.prompt!r}"
        )

        started = time.time()
        result = pipe(
            image=[image],
            prompt=args.prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_inference_steps=args.steps,
            generator=generator,
            true_cfg_scale=args.cfg,
        ).images[0]

        output.parent.mkdir(parents=True, exist_ok=True)
        result.save(output)
        if batch_mode:
            shutil.copy2(source, input_output_dir / source.name)
        with output.open("rb") as output_file:
            digest = hashlib.sha256(output_file.read()).hexdigest()
        metadata = {
            "input": str(source),
            "output": str(output),
            "prompt": args.prompt,
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
        meta_path = output.with_suffix(".json")
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=True, indent=2)
        records.append(metadata)
        print(f"[done {index + 1}/{len(jobs)}] {output} ({metadata['seconds']:.1f}s) sha256={digest}")

    if batch_mode:
        manifest_path = output_root / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        print(
            f"[batch-done] images={len(records)} seconds={time.time() - total_started:.1f} "
            f"manifest={manifest_path}"
        )
    else:
        print(f"[meta] {Path(records[0]['output']).with_suffix('.json')}")


if __name__ == "__main__":
    main()
