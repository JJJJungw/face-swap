#!/usr/bin/env python3
"""[실험] Chroma1-HD img2img — schnell 대비 guidance/negative가 살아있어 스타일을 강하게 따름.
라이선스: Chroma = Apache-2.0(모델카드 재확인 권장) · diffusers = Apache-2.0.
4bit 양자화로 L4 24GB에 상주(cpu_offload 회피 → 예전 RAM 프리즈 방지).
필요: diffusers>=0.36, bitsandbytes.

  python run/chroma_img2img_test.py --image out/testface.png --strengths 0.6,0.7,0.8
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, torch
from PIL import Image
from diffusers import ChromaImg2ImgPipeline, ChromaTransformer2DModel
from diffusers import BitsAndBytesConfig as DiffusersBnb
from transformers import T5EncoderModel, BitsAndBytesConfig as HFBnb

MODEL = "lodestones/Chroma1-HD"
CKPT = "https://huggingface.co/lodestones/Chroma1-HD/blob/main/Chroma1-HD.safetensors"
PROMPT = ("cute stylized 3D animated character render of this face, smooth stylized skin, "
          "large expressive eyes, soft studio lighting, clean polished non-photorealistic 3D look, "
          "friendly, keep the same pose and expression, plain background")
NEG = ("photorealistic, realistic skin pores, real photo, wrinkles, harsh shadows, text, watermark, "
       "logo, deformed, extra fingers, blurry, low quality")

def prep(img, target):
    w, h = img.size; s = target/max(w, h)
    nw = max(256, int(round(w*s/16))*16); nh = max(256, int(round(h*s/16))*16)
    return img.resize((nw, nh), Image.LANCZOS)

def load_pipe():
    nf4 = DiffusersBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    transformer = ChromaTransformer2DModel.from_single_file(CKPT, quantization_config=nf4, torch_dtype=torch.bfloat16)
    nf4h = HFBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    te = T5EncoderModel.from_pretrained(MODEL, subfolder="text_encoder", quantization_config=nf4h, torch_dtype=torch.bfloat16)
    pipe = ChromaImg2ImgPipeline.from_pretrained(MODEL, transformer=transformer, text_encoder=te, torch_dtype=torch.bfloat16)
    pipe.vae.to("cuda"); pipe.enable_vae_tiling()
    return pipe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--negative-prompt", default=NEG, dest="neg")
    ap.add_argument("--strengths", default="0.6,0.7,0.8")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--out", default="out/chroma_test")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pipe = load_pipe()
    init = prep(Image.open(args.image).convert("RGB"), args.size)
    print("input size:", init.size)
    for st in [float(x) for x in args.strengths.split(",")]:
        gen = torch.Generator("cpu").manual_seed(0)
        out = pipe(prompt=args.prompt, negative_prompt=args.neg, image=init, strength=st,
                   guidance_scale=args.guidance, num_inference_steps=args.steps, generator=gen).images[0]
        p = os.path.join(args.out, f"chroma_s{st:.2f}.jpg"); out.save(p); print("saved", p)
    print("\n완료 → out/chroma_test/ 에서 chroma_s*.jpg 확인 (guidance/neg 살아있어 schnell보다 스타일 강함)")

if __name__ == "__main__":
    main()
