#!/usr/bin/env python3
"""[③ 페어 생성] LoRA 입힌 Chroma로 실사 얼굴(SFHQ) → 2.5D 애니 변환 → (실사, 애니) 페어.
ai-toolkit LoRA(BFL/ComfyUI 포맷)를 diffusers 포맷으로 '내장 변환'해서 로드(융합 qkv 분리 포함).
  # 테스트 3장:
  python run/chroma_pairgen.py \
    --input input/sfhq_t2i/a_tiny_sample_new \
    --lora ai-toolkit/output/chroma_style_lora/chroma_style_lora.safetensors \
    --out out/pairs_test --n 3 --strength 0.65 --lora-scale 1.0
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, glob, json, torch
from PIL import Image
from safetensors.torch import load_file
from diffusers import ChromaImg2ImgPipeline, ChromaTransformer2DModel, GGUFQuantizationConfig

MODEL = "lodestones/Chroma1-HD"
GGUF = "https://huggingface.co/silveroxides/Chroma1-HD-GGUF/blob/main/Chroma1-HD-Q6_K.gguf"
HID = 3072   # Chroma hidden dim (qkv lora_B=9216=3*3072 로 확인)

PROMPT = ("s2anime, semi-realistic 2.5D anime portrait, single person close-up face, "
          "soft painterly anime shading, keep the same pose and facial expression, plain simple background")
NEG = ("3D render, Pixar style, Disney style, photorealistic, real photo, flat 2D, chibi, "
       "multiple people, two people, extra person, full body, duplicate face, merged faces, "
       "deformed, disfigured, melted, smeared, extra fingers, bad hands, blurry, low quality, watermark, text")
EXTS = (".png", ".jpg", ".jpeg", ".webp")


def convert_bfl_to_diffusers(sd):
    """ai-toolkit BFL LoRA(diffusion_model.double/single_blocks, 융합 qkv) → diffusers 포맷.
    융합 qkv/linear1: lora_A 공유, lora_B를 q/k/v(/mlp)로 슬라이스."""
    new = {}
    def put(mod, A, B):
        new[f"transformer.{mod}.lora_A.weight"] = A.clone().contiguous()
        new[f"transformer.{mod}.lora_B.weight"] = B.clone().contiguous()
    bases = {k[:-len(".lora_A.weight")] for k in sd if k.endswith(".lora_A.weight")}
    for base in sorted(bases):
        A = sd[base + ".lora_A.weight"]; B = sd[base + ".lora_B.weight"]
        b = base.replace("diffusion_model.", "", 1)
        p = b.split(".")
        if p[0] == "double_blocks":
            i, sub = p[1], ".".join(p[2:]); pre = f"transformer_blocks.{i}"
            if sub == "img_attn.qkv":
                for nm, s in [("attn.to_q", 0), ("attn.to_k", 1), ("attn.to_v", 2)]:
                    put(f"{pre}.{nm}", A, B[s*HID:(s+1)*HID])
            elif sub == "img_attn.proj":  put(f"{pre}.attn.to_out.0", A, B)
            elif sub == "txt_attn.qkv":
                for nm, s in [("attn.add_q_proj", 0), ("attn.add_k_proj", 1), ("attn.add_v_proj", 2)]:
                    put(f"{pre}.{nm}", A, B[s*HID:(s+1)*HID])
            elif sub == "txt_attn.proj": put(f"{pre}.attn.to_add_out", A, B)
            elif sub == "img_mlp.0":     put(f"{pre}.ff.net.0.proj", A, B)
            elif sub == "img_mlp.2":     put(f"{pre}.ff.net.2", A, B)
            elif sub == "txt_mlp.0":     put(f"{pre}.ff_context.net.0.proj", A, B)
            elif sub == "txt_mlp.2":     put(f"{pre}.ff_context.net.2", A, B)
            else: print("  skip(double):", sub)
        elif p[0] == "single_blocks":
            i, sub = p[1], ".".join(p[2:]); pre = f"single_transformer_blocks.{i}"
            if sub == "linear1":
                for nm, s in [("attn.to_q", 0), ("attn.to_k", 1), ("attn.to_v", 2)]:
                    put(f"{pre}.{nm}", A, B[s*HID:(s+1)*HID])
                put(f"{pre}.proj_mlp", A, B[3*HID:])   # 나머지 = mlp(12288)
            elif sub == "linear2": put(f"{pre}.proj_out", A, B)
            else: print("  skip(single):", sub)
        else: print("  skip(top):", b)
    return new


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
    conv = convert_bfl_to_diffusers(load_file(lora))   # BFL → diffusers 변환
    print(f"LoRA 변환: {len(conv)} keys")
    pipe.load_lora_weights(conv, adapter_name="style")
    pipe.set_adapters(["style"], adapter_weights=[lora_scale])
    print(f"LoRA 적용 완료 (scale={lora_scale})")
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", default="out/pairs")
    ap.add_argument("--n", type=int, default=0, help="처리 장수(0=전부)")
    ap.add_argument("--strength", type=float, default=0.7, help="변환 강도(↑=스타일강/원본약)")
    ap.add_argument("--lora-scale", type=float, default=1.0, dest="lora_scale", help="LoRA 세기")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gguf", default=GGUF)
    ap.add_argument("--prompt", default=PROMPT, help="화풍 프롬프트(카툰 테스트용 오버라이드)")
    ap.add_argument("--neg", default=NEG, help="네거티브 프롬프트 오버라이드")
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
        out = pipe(prompt=args.prompt, negative_prompt=args.neg, image=init, strength=args.strength,
                   guidance_scale=args.guidance, num_inference_steps=args.steps, generator=gen).images[0]
        name = f"pair_{i:05d}.png"
        init.save(os.path.join(din, name)); out.save(os.path.join(dtg, name))
        man.write(json.dumps({"name": name, "src": os.path.basename(p),
                              "strength": args.strength, "lora_scale": args.lora_scale}) + "\n")
        print(f"[{i+1}/{len(imgs)}] {name}  <- {os.path.basename(p)}")
    man.close()
    print(f"\n완료 → {args.out}  (input/=실사, target/=애니)")

if __name__ == "__main__":
    main()
