#!/usr/bin/env python3
"""[③ 페어 생성] LoRA 입힌 Chroma로 실사 얼굴(SFHQ) → 2.5D 애니 변환 → (실사, 애니) 페어.
학생 모델 학습용 페어 데이터셋. 표정·포즈는 강도로 유지, 신원은 화풍으로 흐림.
GGUF Q6(7.5GB)라 메모리 가벼움. ② 스타일 LoRA를 load_lora_weights로 적용.
  # 테스트 5장(tiny 샘플):
  python run/chroma_pairgen.py \
    --input input/sfhq_t2i/a_tiny_sample_new \
    --lora ai-toolkit/output/chroma_style_lora/chroma_style_lora.safetensors \
    --out out/pairs_test --n 5 --strength 0.6
  # 대량(전부): --n 0 --out out/pairs
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, glob, json, torch
from PIL import Image
from diffusers import ChromaImg2ImgPipeline, ChromaTransformer2DModel, GGUFQuantizationConfig

MODEL = "lodestones/Chroma1-HD"
GGUF = "https://huggingface.co/silveroxides/Chroma1-HD-GGUF/blob/main/Chroma1-HD-Q6_K.gguf"

# 콘텐츠(사람)는 입력 이미지가 주니까, 프롬프트는 화풍 트리거 + '표정/포즈 유지'만.
PROMPT = ("s2anime, semi-realistic 2.5D anime portrait, soft painterly anime shading, "
          "keep the same pose and facial expression, plain simple background")
NEG = ("3D render, Pixar style, Disney style, photorealistic, real photo, flat 2D, chibi, "
       "deformed, extra fingers, bad hands, blurry, low quality, watermark, text")
EXTS = (".png", ".jpg", ".jpeg", ".webp")

def prep(img, target):
    img = img.convert("RGB"); w, h = img.size; s = target / max(w, h)
    nw = max(256, int(round(w * s / 16)) * 16); nh = max(256, int(round(h * s / 16)) * 16)
    return img.resize((nw, nh), Image.LANCZOS)

def load_pipe(gguf, lora, lora_scale):
    t = ChromaTransformer2DModel.from_single_file(
        gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16)
    pipe = ChromaImg2ImgPipeline.from_pretrained(MODEL, transformer=t, torch_dtype=torch.bfloat16)
    pipe.to("cuda"); pipe.vae.enable_tiling()
    pipe.load_lora_weights(lora, adapter_name="style")          # ② 스타일 LoRA
    pipe.set_adapters(["style"], adapter_weights=[lora_scale])  # LoRA 세기(>1 = base 눌러 화풍 강화)
    print(f"LoRA 적용: {os.path.basename(lora)} (scale={lora_scale})")
    return pipe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="실사 얼굴 폴더(SFHQ)")
    ap.add_argument("--lora", required=True, help="chroma_style_lora.safetensors 경로")
    ap.add_argument("--out", default="out/pairs")
    ap.add_argument("--n", type=int, default=0, help="처리 장수(0=전부)")
    ap.add_argument("--strength", type=float, default=0.7, help="변환 강도(↑=스타일강/원본약, 과하면 환각)")
    ap.add_argument("--lora-scale", type=float, default=1.2, dest="lora_scale",
                    help="LoRA 세기(>1 = 학습화풍 강화, base 눌러줌)")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gguf", default=GGUF)
    args = ap.parse_args()

    imgs = sorted(p for p in glob.glob(os.path.join(args.input, "*")) if p.lower().endswith(EXTS))
    if args.n > 0: imgs = imgs[:args.n]
    if not imgs: raise SystemExit(f"입력 얼굴 없음: {args.input}")

    din = os.path.join(args.out, "input"); dtg = os.path.join(args.out, "target")
    os.makedirs(din, exist_ok=True); os.makedirs(dtg, exist_ok=True)
    pipe = load_pipe(args.gguf, args.lora, args.lora_scale)
    man = open(os.path.join(args.out, "manifest.jsonl"), "w")

    for i, p in enumerate(imgs):
        init = prep(Image.open(p), args.size)
        gen = torch.Generator("cpu").manual_seed(args.seed + i)
        out = pipe(prompt=PROMPT, negative_prompt=NEG, image=init, strength=args.strength,
                   guidance_scale=args.guidance, num_inference_steps=args.steps, generator=gen).images[0]
        name = f"pair_{i:05d}.png"
        init.save(os.path.join(din, name)); out.save(os.path.join(dtg, name))
        man.write(json.dumps({"name": name, "src": os.path.basename(p),
                              "strength": args.strength, "steps": args.steps}) + "\n")
        print(f"[{i+1}/{len(imgs)}] {name}  <- {os.path.basename(p)}")
    man.close()
    print(f"\n완료 → {args.out}  (input/=실사, target/=애니, manifest.jsonl)")
    print("확인: input↔target 나란히 보고 '표정유지 + 화풍' 판정. 좋으면 --n 0 로 대량 생성.")

if __name__ == "__main__":
    main()
