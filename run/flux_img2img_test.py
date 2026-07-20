#!/usr/bin/env python3
"""
FLUX.1-schnell img2img — 사진 → 반실사 페인터리 애니 룩 테스트
- 목적: denoising(strength) 스윕으로 "원하는 화풍 + 표정 유지" 프리셋 찾기
- 라이선스: FLUX.1-schnell(Apache 2.0), diffusers(Apache 2.0)
- 24GB GPU: enable_model_cpu_offload()로 안정 구동

사용법:
  python flux_img2img_test.py --image face.jpg
  python flux_img2img_test.py --image face.jpg --strengths 0.35,0.45,0.55,0.65
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline

# 반실사 페인터리 애니 룩 프롬프트 (원하는 화풍에 맞춰 조정)
PROMPT = (
    "semi-realistic anime illustration portrait, soft painterly shading, "
    "clean subtle lineart, detailed expressive eyes, smooth skin rendering, "
    "keep the same face and expression, cinematic soft lighting, high quality"
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="입력 얼굴 사진")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--strengths", default="0.35,0.45,0.55,0.65",
                    help="denoising 강도 스윕(낮을수록 원본 유지)")
    ap.add_argument("--steps", type=int, default=4, help="schnell은 4스텝 권장")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dtype = torch.bfloat16
    pipe = FluxImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=dtype)
    # 24GB(L4)에서 12B 트랜스포머 전체는 안 들어감 → 레이어 단위 오프로드로 VRAM 절약
    # (더 빠르게: bitsandbytes 4bit 양자화로 교체 가능 — 영상용으로 권장)
    pipe.enable_sequential_cpu_offload()

    init = Image.open(args.image).convert("RGB")
    # 정사각 중심 크롭 후 리사이즈 (얼굴 중심 권장)
    w, h = init.size; s = min(w, h)
    init = init.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2)).resize((args.size, args.size))

    gen = torch.Generator("cpu").manual_seed(0)  # 재현성 고정
    for st in [float(x) for x in args.strengths.split(",")]:
        img = pipe(
            prompt=args.prompt,
            image=init,
            strength=st,
            guidance_scale=0.0,          # schnell은 distilled → CFG 미사용
            num_inference_steps=max(args.steps, int(args.steps/st)+1),
            generator=gen,
        ).images[0]
        # 원본|결과 비교 저장
        cmp = Image.new("RGB", (args.size*2, args.size), "white")
        cmp.paste(init, (0,0)); cmp.paste(img, (args.size,0))
        p = os.path.join(args.out, f"strength_{st:.2f}.jpg")
        cmp.save(p); print("saved", p)

    print("\n완료. out/ 폴더에서 strength별 비교 확인 → 원하는 값이 프리셋.")

if __name__ == "__main__":
    main()
