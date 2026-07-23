#!/usr/bin/env python3
"""[스타일 생성] Chroma text2img로 3D 아바타 스타일 얼굴 생성 = 클린 학습데이터 씨앗.
핀터레스트는 '느낌 참조'용으로만, 실제 이미지는 Chroma(Apache) 출력이라 상업 클린.
GGUF Q6 + to(cuda). 다양한 나이·헤어로 얼굴 다양화.
  python run/chroma_text2img_gen.py --n 12 --out out/style_probe
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, torch
from diffusers import ChromaPipeline, ChromaTransformer2DModel, GGUFQuantizationConfig

MODEL = "lodestones/Chroma1-HD"
GGUF = "https://huggingface.co/silveroxides/Chroma1-HD-GGUF/blob/main/Chroma1-HD-Q6_K.gguf"

# 레퍼런스 화풍을 '말로' 기술(이미지 복사 아님 = 클린). 픽사 시그니처(대형 글로시 눈·feature-animation
# 렌더·무비스틸)는 의도적으로 제거 → 매트 피부·적당한 눈·플랫 파스텔의 중립 3D 아바타로 유도.
STYLE = ("cute stylized 3D character portrait of a {subj} with {skin} skin, clean modern 3D avatar look, "
         "smooth soft matte skin with clear rosy blushed cheeks, large friendly rounded eyes with warm "
         "irises and soft catchlights, delicate eyelashes, small soft rounded nose, gentle closed-mouth "
         "smile, rounded youthful face, {hair} hair, {expr}, soft even almost shadowless studio lighting, "
         "simple flat pastel solid background, head and shoulders portrait, polished stylized 3D avatar "
         "character design, high detail")
NEG = ("photorealistic, real photo, realistic skin pores, plastic glossy skin, harsh shadows, "
       "cinematic film still, movie poster, deformed, extra limbs, bad hands, blurry, low quality, "
       "watermark, text, logo, nsfw")

SUBJ = ["young woman", "young man", "teenage girl", "teenage boy", "little girl",
        "little boy", "middle-aged woman", "middle-aged man", "elderly woman", "elderly man"]
HAIR = ["short curly brown", "long wavy blonde", "black top bun", "auburn bob", "messy ginger",
        "straight dark", "braided brown", "silver short", "wavy red", "black twin buns"]
SKIN = ["fair", "light", "medium", "tan", "brown", "dark", "olive"]
EXPR = ["gentle friendly smile", "happy laughing expression", "calm neutral expression", "cheerful grin",
        "shy soft smile", "surprised open-mouth expression", "warm gentle smile", "playful expression"]

def load_pipe(gguf):
    t = ChromaTransformer2DModel.from_single_file(
        gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16)
    pipe = ChromaPipeline.from_pretrained(MODEL, transformer=t, torch_dtype=torch.bfloat16)
    pipe.to("cuda"); pipe.vae.enable_tiling()
    return pipe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gguf", default=GGUF)
    ap.add_argument("--out", default="out/style_probe")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pipe = load_pipe(args.gguf)
    for i in range(args.n):
        subj = SUBJ[i % len(SUBJ)]; hair = HAIR[(i * 3) % len(HAIR)]
        skin = SKIN[(i * 2) % len(SKIN)]; expr = EXPR[(i * 5) % len(EXPR)]   # 축마다 독립적으로 섞기
        prompt = STYLE.format(subj=subj, hair=hair, skin=skin, expr=expr)
        gen = torch.Generator("cpu").manual_seed(args.seed + i)
        img = pipe(prompt=prompt, negative_prompt=NEG, num_inference_steps=args.steps,
                   guidance_scale=args.guidance, height=args.size, width=args.size, generator=gen).images[0]
        p = os.path.join(args.out, f"style_{i:03d}.png"); img.save(p)
        print(f"saved {p}  ({subj}, {hair})")
    print(f"\n완료 → {args.out} 에서 핀터레스트 4장과 비교")

if __name__ == "__main__":
    main()
