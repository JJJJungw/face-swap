#!/usr/bin/env python3
"""[③-b 카툰 페어 생성 — Qwen 2511 + Anime LoRA] (실사→flat 카툰) 페어 생성.

기존 `qwen_pairgen.py`(2509 + autoweeb LoRA)를 대체하지 않는다. 두 teacher를 각각
독립적으로 돌려 A/B 하기 위한 별도 스크립트다.

라이선스 (전 구간 Apache 2.0 — 하드 요구사항 통과):
  - 베이스   Qwen/Qwen-Image-Edit-2511                      Apache 2.0
  - 화풍     prithivMLmods/Qwen-Image-Edit-2511-Anime       Apache 2.0
  - 속도     lightx2v/Qwen-Image-Edit-2511-Lightning        Apache 2.0

2509 대비 기대 이점:
  1) drift 완화 / character consistency 향상 → 페어 정합↑ → paired L1의 blur↓
  2) 화풍("flat cel shading")이 프롬프트 유도가 아니라 LoRA 가중치에 학습됨 → 코퍼스 분산↓
  3) Lightning 4-step → 장당 ~110s(28step) 대비 대폭 단축

⚠️ 4-step은 Lightning LoRA를 함께 스택해야 성립한다(화풍 LoRA 단독 아님).
⚠️ Lightning(step-distilled) 사용 시 true_cfg_scale=1.0이 정석 → **negative_prompt가 무시된다.**
   2509에서 쓰던 NEG("extra person, deformed...") 가드가 사라지므로 큐레이션을 더 봐야 한다.
   NEG가 꼭 필요하면 --no-fast 로 Lightning을 빼고 --cfg 4.0 --steps 28 로 돌린다.

사용:
  # 스모크 테스트 5장 (먼저 이것부터)
  python3 run/qwen2511_pairgen.py --input input/sfhq_t2i/a_tiny_sample_new \
      --out out/qwen2511_smoke --n 5

  # 풀 코퍼스 (--resume: 이미 만든 페어는 건너뜀)
  python3 run/qwen2511_pairgen.py --input input/<코퍼스> \
      --out out/pairs_2511 --n 500 --resume
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, glob, json, time, torch
from PIL import Image
from diffusers import GGUFQuantizationConfig
try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError:
    QwenImageEditPlusPipeline = None
from diffusers import DiffusionPipeline
try:
    from diffusers import QwenImageTransformer2DModel
except ImportError:
    QwenImageTransformer2DModel = None

MODEL = "Qwen/Qwen-Image-Edit-2511"

STYLE_REPO = "prithivMLmods/Qwen-Image-Edit-2511-Anime"
STYLE_FILE = "Qwen-Image-Edit-2511-Anime-2000.safetensors"

FAST_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
FAST_FILE = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

# GGUF 양자화 transformer — 20B bf16(~40GB)은 L40S 46GB에 text encoder까지 못 얹음.
# unsloth/Qwen-Image-Edit-2511-GGUF (Apache 2.0). ※ QuantStack엔 2511 GGUF 없음(404).
#   Q4_K_M 13.2GB / Q5_K_M 15GB / Q6_K / Q8_0 21.8GB / BF16 40.9GB
# teacher는 오프라인 1회 실행이고 그 출력이 학생의 정답(ground truth)이 된다.
# 양자화 손실이 곧 학습 타겟의 손실이므로, VRAM이 남으면 아끼지 말 것.
# L40S 46GB: Q8_0(21.8) + text encoder(~16) ≈ 38GB → 여유 있음 → Q8_0 기본값.
GGUF_REPO = "unsloth/Qwen-Image-Edit-2511-GGUF"
GGUF_FILE = "qwen-image-edit-2511-Q8_0.gguf"

# 화풍 LoRA의 trigger. 2509처럼 긴 프롬프트로 스타일을 "유도"하지 않는다 —
# 화풍은 가중치에 있고, 긴 프롬프트는 오히려 LoRA prior와 싸워서 분산을 키운다.
PROMPT = "Transform into anime."
NEG = ("photorealistic, realistic, oil painting, painterly, 3d render, deformed, "
       "extra person, multiple people, blurry, low quality, text, watermark")
EXTS = (".png", ".jpg", ".jpeg", ".webp")


def prep(img, size):
    """긴 변을 size로 맞추고 8의 배수로 정렬(VAE 요구)."""
    img = img.convert("RGB")
    w, h = img.size
    s = size / max(w, h)
    nw = max(64, int(round(w * s)) // 8 * 8)
    nh = max(64, int(round(h * s)) // 8 * 8)
    return img.resize((nw, nh), Image.LANCZOS)


def build_pipe(args):
    if args.gguf_file:
        if QwenImageTransformer2DModel is None:
            raise SystemExit("QwenImageTransformer2DModel 없음 → pip install -U diffusers")
        url = f"https://huggingface.co/{args.gguf_repo}/blob/main/{args.gguf_file}"
        print(f"[load] GGUF transformer: {args.gguf_file}")
        t = QwenImageTransformer2DModel.from_single_file(
            url, config=MODEL, subfolder="transformer",   # GGUF엔 config 없음 → 원본 config 지정
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16)
        kw = dict(transformer=t)
    else:
        print("[load] bf16 transformer (VRAM 여유 필요)")
        kw = {}

    cls = QwenImageEditPlusPipeline or DiffusionPipeline
    try:
        pipe = cls.from_pretrained(MODEL, torch_dtype=torch.bfloat16, **kw)
    except Exception as e:
        print(f"[경고] {cls.__name__} 실패({e}) → DiffusionPipeline 자동 해석으로 재시도")
        pipe = DiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, **kw)
    return pipe


def load_loras(pipe, args):
    """화풍 LoRA + (옵션)Lightning LoRA 스택. 실패해도 죽지 않고 상태를 리턴."""
    names, weights = [], []

    try:
        pipe.load_lora_weights(args.style_repo, weight_name=args.style_file, adapter_name="anime")
        names.append("anime"); weights.append(args.style_scale)
        print(f"[lora] 화풍 {args.style_repo} (scale={args.style_scale})")
    except Exception as e:
        raise SystemExit(f"[치명] 화풍 LoRA 로드 실패 — 이게 없으면 의미가 없음: {e}")

    if args.fast:
        try:
            pipe.load_lora_weights(args.fast_repo, weight_name=args.fast_file, adapter_name="fast")
            names.append("fast"); weights.append(args.fast_scale)
            print(f"[lora] 속도 {args.fast_repo} (scale={args.fast_scale})")
        except Exception as e:
            print(f"[경고] Lightning LoRA 로드 실패({e}) → 4-step 불가.")
            print("       → steps/cfg를 non-fast 기본값으로 되돌림(느리지만 정상 동작).")
            args.fast = False

    try:
        pipe.set_adapters(names, adapter_weights=weights)
        print(f"[lora] 스택 활성: {names} @ {weights}")
    except Exception as e:
        print(f"[경고] set_adapters 실패({e}) → 마지막 로드된 LoRA만 적용될 수 있음")
    return args.fast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="out/pairs_2511")
    ap.add_argument("--n", type=int, default=0, help="처리 장수(0=전부)")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--neg", default=NEG)
    ap.add_argument("--resume", action="store_true", help="target/에 이미 있는 페어는 건너뜀")

    ap.add_argument("--steps", type=int, default=None, help="미지정 시 fast=4 / non-fast=28")
    ap.add_argument("--cfg", type=float, default=None, help="true_cfg_scale. 미지정 시 fast=1.0 / non-fast=4.0")

    ap.add_argument("--style-repo", default=STYLE_REPO, dest="style_repo")
    ap.add_argument("--style-file", default=STYLE_FILE, dest="style_file")
    ap.add_argument("--style-scale", type=float, default=1.0, dest="style_scale")

    ap.add_argument("--fast", dest="fast", action="store_true", default=True,
                    help="Lightning 4-step (기본 켜짐)")
    ap.add_argument("--no-fast", dest="fast", action="store_false",
                    help="Lightning 빼고 정석 CFG로 (negative_prompt 살아남)")
    ap.add_argument("--fast-repo", default=FAST_REPO, dest="fast_repo")
    ap.add_argument("--fast-file", default=FAST_FILE, dest="fast_file")
    ap.add_argument("--fast-scale", type=float, default=1.0, dest="fast_scale")

    ap.add_argument("--gguf-repo", default=GGUF_REPO, dest="gguf_repo")
    ap.add_argument("--gguf-file", default=GGUF_FILE, dest="gguf_file",
                    help="예: qwen-image-edit-2511-Q8_0.gguf / -Q5_K_M / -Q4_K_M. "
                         "빈 문자열이면 bf16 풀 로드(VRAM 초과 주의)")
    args = ap.parse_args()

    imgs = sorted(p for p in glob.glob(os.path.join(args.input, "*")) if p.lower().endswith(EXTS))
    if args.n > 0:
        imgs = imgs[:args.n]
    if not imgs:
        raise SystemExit(f"입력 없음: {args.input}")

    din = os.path.join(args.out, "input")
    dtg = os.path.join(args.out, "target")
    os.makedirs(din, exist_ok=True)
    os.makedirs(dtg, exist_ok=True)

    pipe = build_pipe(args)
    fast = load_loras(pipe, args)

    steps = args.steps if args.steps is not None else (4 if fast else 28)
    cfg = args.cfg if args.cfg is not None else (1.0 if fast else 4.0)
    if cfg <= 1.0:
        print("[주의] true_cfg_scale<=1.0 → negative_prompt 무시됨(Lightning 정석). 큐레이션 필요.")

    pipe.to("cuda")
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass

    # 2509 스크립트의 버그 수정: "w"(버퍼링) → "a"+flush. 중단해도 manifest 꼬리가 안 잘림.
    man = open(os.path.join(args.out, "manifest.jsonl"), "a", buffering=1)
    print(f"\n[run] steps={steps} cfg={cfg} size={args.size} fast={fast} n={len(imgs)}\n")

    done = skipped = 0
    t_all = time.time()
    for i, p in enumerate(imgs):
        name = f"pair_{i:05d}.png"
        if args.resume and os.path.exists(os.path.join(dtg, name)) \
                       and os.path.exists(os.path.join(din, name)):
            skipped += 1
            continue

        t0 = time.time()
        init = prep(Image.open(p), args.size)
        gen = torch.Generator("cpu").manual_seed(args.seed + i)
        out = pipe(image=[init], prompt=args.prompt, negative_prompt=args.neg,
                   true_cfg_scale=cfg, num_inference_steps=steps, generator=gen).images[0]

        # target 먼저 쓰면 중단 시 input 없는 고아가 생김 → input 먼저, target 나중.
        init.save(os.path.join(din, name))
        out.save(os.path.join(dtg, name))
        man.write(json.dumps({"name": name, "src": os.path.basename(p),
                              "steps": steps, "cfg": cfg}) + "\n")
        done += 1
        dt = time.time() - t0
        eta = (len(imgs) - i - 1) * dt / 60
        print(f"[{i+1}/{len(imgs)}] {name}  {dt:.1f}s  (ETA {eta:.0f}분)")

    man.close()
    print(f"\n완료 → {args.out}   생성 {done}장 / 건너뜀 {skipped}장 / "
          f"총 {(time.time()-t_all)/60:.1f}분")
    print("  input/=실사, target/=카툰  →  다음: run/compare_grid.py 로 육안 확인")


if __name__ == "__main__":
    main()
