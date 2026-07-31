#!/usr/bin/env python3
"""Reproduce the Anime-V2 path used by the public Hugging Face Space."""

import argparse
import hashlib
import json
import os
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="official_sample.jpeg")
    parser.add_argument("--out", default="out/space_exact/official_sample_anime_v2.png")
    parser.add_argument("--prompt", default="Transform into anime.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument(
        "--space-revision",
        default="main",
        help="Space git revision. Use a commit hash to freeze an exact version.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")

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

    transformer = QwenImageTransformer2DModel.from_pretrained(
        RAPID_MODEL,
        torch_dtype=dtype,
        device_map="cuda",
    )
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        BASE_MODEL,
        transformer=transformer,
        torch_dtype=dtype,
    ).to(device)

    if QwenDoubleStreamAttnProcessorFA3 is None:
        attention = f"SDPA fallback (FA3 import failed: {fa3_import_error})"
    else:
        try:
            pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
            attention = "FA3"
        except Exception as exc:
            attention = f"SDPA fallback ({exc})"
    print(f"[attention] {attention}")

    pipe.load_lora_weights(ANIME_REPO, weight_name=ANIME_FILE, adapter_name="anime-v2")
    pipe.set_adapters(["anime-v2"], adapter_weights=[1.0])

    image = ImageOps.exif_transpose(Image.open(args.input)).convert("RGB")
    width, height = space_size(image)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    print(
        f"[run] input={args.input} output_size={width}x{height} seed={args.seed} "
        f"steps={args.steps} cfg={args.cfg} prompt={args.prompt!r}"
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

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    result.save(args.out)
    with open(args.out, "rb") as output_file:
        digest = hashlib.sha256(output_file.read()).hexdigest()
    metadata = {
        "input": args.input,
        "output": args.out,
        "prompt": args.prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "seed": args.seed,
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
        "seconds": round(time.time() - started, 3),
        "sha256": digest,
    }
    meta_path = os.path.splitext(args.out)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2)
    print(f"[done] {args.out} ({metadata['seconds']:.1f}s) sha256={digest}")
    print(f"[meta] {meta_path}")


if __name__ == "__main__":
    main()
