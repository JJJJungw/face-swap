#!/usr/bin/env python3
"""
FLUX.1-schnell 영상 카툰화 테스트 (per-frame img2img)
입력 영상 → 프레임 분해 → per-frame 스타일화 → 재결합(+오디오)

주의: per-frame diffusion이라 약간의 깜빡임(flicker)은 정상.
      1차 "영상이 되는지" 검증용. 이후 optical flow/AnimateDiff로 일관성 강화 예정.

사용법:
  python run/flux_video_test.py --video clip.mp4
  python run/flux_video_test.py --video clip.mp4 --fps 8 --size 512 --strength 0.5 --smooth 0.3
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, subprocess, glob, shutil, torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline

PROMPT = ("semi-realistic anime illustration, soft painterly shading, "
          "clean subtle lineart, detailed expressive eyes, smooth skin rendering, "
          "keep the same subject and expression, cinematic soft lighting, high quality")

def run(cmd):
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="입력 영상(mp4 등)")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--fps", type=int, default=8, help="처리 fps (낮출수록 빠름, 8~12 권장)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--strength", type=float, default=0.5, help="스타일 강도(낮을수록 원본 유지)")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="0~0.6 이전 프레임과 블렌딩(깜빡임 저감, 움직임엔 잔상 주의)")
    ap.add_argument("--out", default="video_out")
    args = ap.parse_args()

    fin = os.path.join(args.out, "frames_in")
    fout = os.path.join(args.out, "frames_out")
    for d in (fin, fout):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d, exist_ok=True)

    # 1) 프레임 분해 + 오디오 추출
    run(["ffmpeg", "-y", "-i", args.video, "-vf", f"fps={args.fps}",
         os.path.join(fin, "f_%05d.png")])
    audio = os.path.join(args.out, "audio.m4a")
    has_audio = subprocess.run(
        ["ffmpeg", "-y", "-i", args.video, "-vn", "-c:a", "aac", audio],
        capture_output=True).returncode == 0

    frames = sorted(glob.glob(os.path.join(fin, "f_*.png")))
    print(f"{len(frames)} frames @ {args.fps}fps  (오디오: {'있음' if has_audio else '없음'})")

    # 2) 모델 로드 (프롬프트 임베딩 1회 계산해 프레임마다 텍스트 인코딩 생략 → 속도↑)
    pipe = FluxImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()  # L4 24GB 대응 (영상 다량 처리는 4bit 양자화 권장)
    with torch.no_grad():
        pe, ppe, _ = pipe.encode_prompt(prompt=args.prompt, prompt_2=args.prompt,
                                        device=pipe._execution_device, num_images_per_prompt=1)

    steps = max(args.steps, int(args.steps / args.strength) + 1)
    prev = None
    for i, fp in enumerate(frames):
        img = Image.open(fp).convert("RGB").resize((args.size, args.size))
        gen = torch.Generator("cpu").manual_seed(0)   # 고정 시드 = 깜빡임 저감
        out = pipe(prompt_embeds=pe, pooled_prompt_embeds=ppe,
                   image=img, strength=args.strength, guidance_scale=0.0,
                   num_inference_steps=steps, generator=gen).images[0]
        if args.smooth > 0 and prev is not None:
            out = Image.blend(out, prev, args.smooth)  # 이전 결과와 블렌딩
        prev = out
        out.save(os.path.join(fout, f"f_{i+1:05d}.png"))
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(frames)}")

    # 3) 재결합 (+오디오)
    result = os.path.join(args.out, "result.mp4")
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps),
           "-i", os.path.join(fout, "f_%05d.png")]
    if has_audio:
        cmd += ["-i", audio, "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", result]
    run(cmd)
    print("DONE ->", result)

if __name__ == "__main__":
    main()
