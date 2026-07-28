#!/usr/bin/env python3
"""[③ 카툰 페어 생성 — Qwen] Qwen-Image-Edit-2509 + photo-to-anime LoRA로 (실사→카툰) 페어.

라이선스: 베이스 Qwen-Image-Edit-2509 = Apache 2.0, LoRA(autoweeb) = MIT → 클린.
※ 20B diffusion이라 느림(teacher 전용, 런타임 아님). 이미지당 수십 초~1분+.

⚠️ 첫 실행 주의:
  - 베이스 모델 ~40GB 다운로드(디스크 여유 확인: df -h).
  - diffusers가 구버전이면 `QwenImageEditPlusPipeline` 없음 → `pip install -U diffusers accelerate`.
    (런타임 Chroma가 깨질까 걱정되면 별도 venv 권장. 근데 보통 상위호환 됨.)
  - 46GB VRAM에 20B는 빠듯 → `--offload`(CPU offload)로 맞춤(느려지지만 안전).

사용(테스트 5장):
  python run/qwen_pairgen.py --input input/sfhq_t2i/a_tiny_sample_new \
    --out out/qwen_cartoon_test --n 5 --offload
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, glob, json, torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

MODEL = "Qwen/Qwen-Image-Edit-2509"
LORA = "autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime"
PROMPT = ("transform this portrait into a 2D cartoon anime illustration, cel shaded, "
          "clean bold outlines, flat colors, keep the exact same face pose, gaze and expression, "
          "same framing and composition")
NEG = ("photorealistic, realistic, oil painting, painterly, 3d render, deformed, "
       "extra person, multiple people, blurry, low quality, text, watermark")
EXTS = (".png", ".jpg", ".jpeg", ".webp")


def prep(img, size):
    img = img.convert("RGB"); w, h = img.size; s = size / max(w, h)
    nw = max(64, int(round(w * s)) // 8 * 8); nh = max(64, int(round(h * s)) // 8 * 8)
    return img.resize((nw, nh), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="out/pairs_cartoon")
    ap.add_argument("--n", type=int, default=0, help="처리 장수(0=전부)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=4.0, help="true_cfg_scale")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora", default=LORA)
    ap.add_argument("--lora-weight", default=None, dest="lora_weight", help="LoRA safetensors 파일명(필요시)")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--neg", default=NEG)
    ap.add_argument("--offload", action="store_true", help="CPU offload(46GB 빠듯하면 권장)")
    args = ap.parse_args()

    imgs = sorted(p for p in glob.glob(os.path.join(args.input, "*")) if p.lower().endswith(EXTS))
    if args.n > 0:
        imgs = imgs[:args.n]
    if not imgs:
        raise SystemExit(f"입력 없음: {args.input}")

    din = os.path.join(args.out, "input"); dtg = os.path.join(args.out, "target")
    os.makedirs(din, exist_ok=True); os.makedirs(dtg, exist_ok=True)

    print(f"[load] {MODEL} (bf16) — 첫 실행 시 ~40GB 다운로드")
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    try:
        if args.lora_weight:
            pipe.load_lora_weights(args.lora, weight_name=args.lora_weight)
        else:
            pipe.load_lora_weights(args.lora)
        print(f"[lora] {args.lora} 로드 완료")
    except Exception as e:
        print(f"[경고] LoRA 로드 실패({e}) → 베이스만으로 진행(프롬프트로 카툰 유도)")
    if args.offload:
        pipe.enable_model_cpu_offload()
        print("[mem] CPU offload 활성")
    else:
        pipe.to("cuda")
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass

    man = open(os.path.join(args.out, "manifest.jsonl"), "w")
    for i, p in enumerate(imgs):
        init = prep(Image.open(p), args.size)
        gen = torch.Generator("cpu").manual_seed(args.seed + i)
        out = pipe(image=[init], prompt=args.prompt, negative_prompt=args.neg,
                   true_cfg_scale=args.cfg, num_inference_steps=args.steps, generator=gen).images[0]
        name = f"pair_{i:05d}.png"
        init.save(os.path.join(din, name)); out.save(os.path.join(dtg, name))
        man.write(json.dumps({"name": name, "src": os.path.basename(p)}) + "\n")
        print(f"[{i+1}/{len(imgs)}] {name}")
    man.close()
    print(f"\n완료 → {args.out}  (input/=실사, target/=카툰)")


if __name__ == "__main__":
    main()
