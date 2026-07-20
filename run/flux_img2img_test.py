#!/usr/bin/env python3
"""
FLUX.1-schnell img2img — 사진 → 2.5D 반실사 룩 (4bit 양자화로 가속)
- 라이선스: FLUX.1-schnell(Apache 2.0), diffusers(Apache 2.0)
- 24GB(L4): 4bit 양자화 → 트랜스포머 ~7GB → sequential offload 제거 → 대폭 가속
- 필요: bitsandbytes  (uv pip install bitsandbytes)

사용법:
  python run/flux_img2img_test.py --image face.jpg
  python run/flux_img2img_test.py --image face.jpg --strengths 0.4 --size 1024
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline, FluxTransformer2DModel
from diffusers import BitsAndBytesConfig as DiffusersBnb
from transformers import T5EncoderModel, BitsAndBytesConfig as HFBnb

REPO = "black-forest-labs/FLUX.1-schnell"
PROMPT = (
    "beautiful semi-realistic 2.5d portrait, glossy detailed eyes, smooth flawless skin, "
    "subtle illustration polish, soft cinematic lighting, highly detailed, "
    "preserve the same face pose and expression, no text, no watermark"
)

def prep(img, target):
    w, h = img.size
    scale = target / max(w, h)
    nw = max(256, int(round(w * scale / 16)) * 16)
    nh = max(256, int(round(h * scale / 16)) * 16)
    return img.resize((nw, nh), Image.LANCZOS)

def load_pipe():
    # 트랜스포머 4bit
    nf4 = DiffusersBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_compute_dtype=torch.bfloat16)
    transformer = FluxTransformer2DModel.from_pretrained(
        REPO, subfolder="transformer", quantization_config=nf4, torch_dtype=torch.bfloat16)
    # T5 텍스트 인코더 4bit
    nf4h = HFBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                 bnb_4bit_compute_dtype=torch.bfloat16)
    te2 = T5EncoderModel.from_pretrained(
        REPO, subfolder="text_encoder_2", quantization_config=nf4h, torch_dtype=torch.bfloat16)
    pipe = FluxImg2ImgPipeline.from_pretrained(
        REPO, transformer=transformer, text_encoder_2=te2, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()   # 양자화 상태라 빠르고 24GB에 안전
    return pipe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--strengths", default="0.4")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pipe = load_pipe()
    init = prep(Image.open(args.image).convert("RGB"), args.size)
    print("input size:", init.size)

    for st in [float(x) for x in args.strengths.split(",")]:
        gen = torch.Generator("cpu").manual_seed(0)
        out = pipe(prompt=args.prompt, image=init, strength=st, guidance_scale=0.0,
                   num_inference_steps=max(args.steps, int(args.steps / st) + 1),
                   generator=gen).images[0]
        p = os.path.join(args.out, f"result_{st:.2f}.jpg")
        out.save(p); print("saved", p)

    print("\n완료 → out/ 에서 result_*.jpg 확인")

if __name__ == "__main__":
    main()
