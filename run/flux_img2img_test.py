#!/usr/bin/env python3
"""
FLUX.1-schnell img2img — 사진 → 반실사 페인터리 애니 룩 테스트
- 라이선스: FLUX.1-schnell(Apache 2.0), diffusers(Apache 2.0)
- 24GB(L4): sequential_cpu_offload (안전하지만 느림)
- Flux는 1024 해상도 기준 → 512는 흐릿해짐. 기본 1024 권장.

사용법:
  python run/flux_img2img_test.py --image face.jpg
  python run/flux_img2img_test.py --image face.jpg --strengths 0.5,0.6,0.7 --size 1024
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline

PROMPT = (
    "semi-realistic anime illustration, soft painterly cel shading, clean lineart, "
    "detailed expressive eyes, smooth stylized skin, preserve the same face, pose and expression, "
    "korean webtoon style, cinematic soft lighting, highly detailed, high quality"
)

def prep(img, target):
    """비율 유지 리사이즈 (긴 변 = target, 양변 16의 배수) — 크롭/왜곡 없음"""
    w, h = img.size
    scale = target / max(w, h)
    nw = max(256, int(round(w * scale / 16)) * 16)
    nh = max(256, int(round(h * scale / 16)) * 16)
    return img.resize((nw, nh), Image.LANCZOS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--strengths", default="0.6", help="스타일 강도(콤마로 여러 개)")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--size", type=int, default=1024, help="Flux 기준 1024 권장")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pipe = FluxImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()   # L4 24GB 대응 (느림 → 이후 4bit 양자화로 가속)

    init = prep(Image.open(args.image).convert("RGB"), args.size)
    print("input size:", init.size)

    for st in [float(x) for x in args.strengths.split(",")]:
        gen = torch.Generator("cpu").manual_seed(0)
        out = pipe(
            prompt=args.prompt, image=init, strength=st, guidance_scale=0.0,
            num_inference_steps=max(args.steps, int(args.steps / st) + 1),
            generator=gen,
        ).images[0]
        # 결과 한 장만 저장
        p = os.path.join(args.out, f"result_{st:.2f}.jpg")
        out.save(p)
        print("saved", p)

    print("\n완료 → out/ 에서 result_*.jpg 확인")

if __name__ == "__main__":
    main()
