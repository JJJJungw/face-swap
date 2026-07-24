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

# [화풍 타깃] 2.5D 반실사 애니(원래 기획안). 3D 렌더 계열을 아예 배제해 픽사/디즈니색을 근본적으로 차단.
# 핵심: 3D가 아니라 '반실사 애니 일러스트' + 자연스러운 비율 + 아몬드형 눈(초대형 X) + 소프트 페인터리 셰이딩.
STYLE = ("semi-realistic 2.5D anime portrait of a {subj} with {skin} skin, modern high-quality anime "
         "illustration, natural stylized facial proportions, expressive almond-shaped anime eyes with "
         "detailed irises and soft catchlights, refined realistic nose and lips, smooth skin with soft "
         "painterly anime shading and subtle natural blush, {hair} hair, {expr}, wearing a plain "
         "crew-neck t-shirt, {angle}, soft natural lighting, simple flat pastel background, head and "
         "shoulders portrait, detailed semi-realistic anime key-visual rendering with gentle depth")
# 3D/픽사/디즈니 계열을 네거티브로 강하게 배제 + chibi·초대형눈(픽사눈) 차단 + 노출 방지.
NEG = ("3D render, 3D model, CGI, octane render, blender render, Pixar style, Disney style, claymation, "
       "video game character, plastic glossy skin, photorealistic, real photo, realistic skin pores, "
       "chibi, oversized round eyes, harsh shadows, cinematic film still, deformed, extra limbs, bad "
       "hands, blurry, low quality, watermark, text, logo, nsfw, bare shoulders, nude, shirtless")

SUBJ = ["young woman", "young man", "teenage girl", "teenage boy", "little girl",
        "little boy", "middle-aged woman", "middle-aged man", "elderly woman", "elderly man"]
HAIR = ["short curly brown", "long wavy blonde", "black top bun", "auburn bob", "messy ginger",
        "straight dark", "braided brown", "silver short", "wavy red", "black twin buns"]
SKIN = ["fair", "light", "medium", "tan", "brown", "dark", "olive"]
EXPR = ["gentle friendly smile", "happy laughing expression", "calm neutral expression", "cheerful grin",
        "shy soft smile", "surprised expression", "warm gentle smile", "playful winking expression",
        "curious thoughtful look", "big bright smile", "slightly pouting expression", "soft serene face"]
# 각도 다양화 → 학생 모델이 다양한 머리 포즈에 강해짐(정면/3-4/측면/상하 틸트).
ANGLE = ["front view", "slight three-quarter view", "three-quarter view", "gentle side angle",
         "head tilted slightly", "looking slightly up", "looking slightly down", "turned slightly to the side"]

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
        angle = ANGLE[(i * 7) % len(ANGLE)]                                  # 각도도 독립 축
        prompt = STYLE.format(subj=subj, hair=hair, skin=skin, expr=expr, angle=angle)
        gen = torch.Generator("cpu").manual_seed(args.seed + i)
        img = pipe(prompt=prompt, negative_prompt=NEG, num_inference_steps=args.steps,
                   guidance_scale=args.guidance, height=args.size, width=args.size, generator=gen).images[0]
        p = os.path.join(args.out, f"style_{i:03d}.png"); img.save(p)
        print(f"saved {p}  ({subj}, {hair}, {expr}, {angle})")
    print(f"\n완료 → {args.out} 에서 핀터레스트 4장과 비교")

if __name__ == "__main__":
    main()
