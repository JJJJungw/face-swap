#!/usr/bin/env python3
"""FLUX.1-schnell 배치 카툰화 — 모델 1회 로드 후 폴더 내 모든 이미지 처리 (재로드 없음)
- 프롬프트 임베딩도 1회 계산 → 재사용 → 장당 속도↑
- 증류용 데이터셋 생성에도 그대로 사용
사용법:
  python run/flux_batch.py --indir input --outdir out/batch --strength 0.7
  python run/flux_batch.py --indir myfaces --strength 0.7 --size 768   # 768로 더 빠르게
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, time, glob, torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline, FluxTransformer2DModel, BitsAndBytesConfig as DBnb
from transformers import T5EncoderModel, BitsAndBytesConfig as HBnb

REPO = "black-forest-labs/FLUX.1-schnell"
PROMPT = (
    "original generic stylized 3D animated avatar face, non-photorealistic, "
    "large expressive eyes, soft rounded cheeks, small simplified nose, "
    "smooth clean skin, gentle toon shading, detailed stylized hair, "
    "friendly neutral expression, face crop portrait, soft studio lighting, "
    "high quality 3D render, no text, no watermark"
)

def prep(p, size):
    img = Image.open(p).convert("RGB"); w, h = img.size; sc = size/max(w, h)
    nw = max(256, int(round(w*sc/16))*16); nh = max(256, int(round(h*sc/16))*16)
    return img.resize((nw, nh), Image.LANCZOS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="input")
    ap.add_argument("--outdir", default="out/batch")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--strength", type=float, default=0.7)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    imgs = sorted([p for e in ("*.jpg", "*.jpeg", "*.png") for p in glob.glob(os.path.join(args.indir, e))])
    print(len(imgs), "images in", args.indir)
    if not imgs:
        print("이미지 없음 — --indir 확인"); return

    print("loading model (1회만)...")
    t0 = time.perf_counter()
    nf4 = DBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tr  = FluxTransformer2DModel.from_pretrained(REPO, subfolder="transformer", quantization_config=nf4, torch_dtype=torch.bfloat16)
    nf4h= HBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    te2 = T5EncoderModel.from_pretrained(REPO, subfolder="text_encoder_2", quantization_config=nf4h, torch_dtype=torch.bfloat16)
    pipe = FluxImg2ImgPipeline.from_pretrained(REPO, transformer=tr, text_encoder_2=te2, torch_dtype=torch.bfloat16)
    pipe.vae.to("cuda"); pipe.text_encoder.to("cuda"); pipe.enable_vae_tiling()
    print(f"loaded in {time.perf_counter()-t0:.0f}s")

    # 프롬프트 임베딩 1회 계산 → 프레임마다 텍스트 인코딩 생략
    with torch.no_grad():
        pe, ppe, _ = pipe.encode_prompt(prompt=args.prompt, prompt_2=args.prompt, device="cuda", num_images_per_prompt=1)

    steps = max(args.steps, int(args.steps/args.strength)+1)
    t1 = time.perf_counter()
    for i, p in enumerate(imgs):
        g = torch.Generator("cpu").manual_seed(0)
        out = pipe(prompt_embeds=pe, pooled_prompt_embeds=ppe, image=prep(p, args.size),
                   strength=args.strength, guidance_scale=0.0, num_inference_steps=steps, generator=g).images[0]
        name = os.path.splitext(os.path.basename(p))[0]
        out.save(os.path.join(args.outdir, f"{name}_cartoon.jpg"))
        print(f"  {i+1}/{len(imgs)} {name}  {(time.perf_counter()-t1)/(i+1):.1f}s/img")
    print(f"완료 {time.perf_counter()-t1:.0f}s ({(time.perf_counter()-t1)/len(imgs):.1f}s/img) -> {args.outdir}")

if __name__ == "__main__":
    main()
