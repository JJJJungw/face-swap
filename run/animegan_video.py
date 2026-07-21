#!/usr/bin/env python3
"""animegan2 영상 스타일화 (전체 프레임 MVP)
프레임 분해 → 배치+FP16 스타일화 → 재결합(+오디오) + 속도(1:2) 측정
- 코드/가중치 자동 다운로드 (MIT)
사용법:
  python run/animegan_video.py --video input/clip.mp4
  python run/animegan_video.py --video input/clip.mp4 --fps 24 --size 512 --batch 8
"""
import os, sys, argparse, subprocess, glob, shutil, ssl, urllib.request, time, torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

CKPT_DIR = "gan_ckpt"
BASE = "https://raw.githubusercontent.com/bryandlee/animegan2-pytorch"
FILES = {"model.py": f"{BASE}/master/model.py", "w.pt": f"{BASE}/main/weights/face_paint_512_v2.pt"}

def ensure_ckpt():
    os.makedirs(CKPT_DIR, exist_ok=True)
    ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
    for n, u in FILES.items():
        d = os.path.join(CKPT_DIR, n)
        if not os.path.exists(d):
            with urllib.request.urlopen(u, context=ctx, timeout=90) as r, open(d, "wb") as f:
                f.write(r.read())

def run(cmd): subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--size", type=int, default=512, help="긴 변 기준 처리 해상도")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="video_out")
    args = ap.parse_args()

    fin = os.path.join(args.out, "in"); fout = os.path.join(args.out, "out")
    for d in (fin, fout): shutil.rmtree(d, ignore_errors=True); os.makedirs(d, exist_ok=True)

    # 1) 프레임 분해 + 오디오
    run(["ffmpeg", "-y", "-i", args.video, "-vf", f"fps={args.fps}", os.path.join(fin, "f_%05d.png")])
    audio = os.path.join(args.out, "audio.m4a")
    has_audio = subprocess.run(["ffmpeg", "-y", "-i", args.video, "-vn", "-c:a", "aac", audio],
                               capture_output=True).returncode == 0
    frames = sorted(glob.glob(os.path.join(fin, "f_*.png")))
    print(f"{len(frames)} frames @ {args.fps}fps, audio={has_audio}")

    # 2) 모델
    ensure_ckpt(); sys.path.insert(0, CKPT_DIR)
    from model import Generator
    torch.backends.cudnn.benchmark = True
    dev = "cuda"; m = Generator().to(dev).eval().to(memory_format=torch.channels_last)

    im0 = Image.open(frames[0]); w, h = im0.size; sc = args.size / max(w, h)
    nw = max(64, int(round(w*sc/8))*8); nh = max(64, int(round(h*sc/8))*8)
    print("proc size:", (nw, nh))

    torch.cuda.synchronize(); t0 = time.perf_counter(); done = 0
    for i in range(0, len(frames), args.batch):
        chunk = frames[i:i+args.batch]
        batch = torch.stack([to_tensor(Image.open(p).convert("RGB").resize((nw, nh), Image.LANCZOS))*2-1
                             for p in chunk]).to(dev).to(memory_format=torch.channels_last)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            y = m(batch)
        y = (y.float()*0.5+0.5).clip(0, 1).cpu()
        for j, p in enumerate(chunk):
            to_pil_image(y[j]).save(os.path.join(fout, os.path.basename(p)))
        done += len(chunk); print(f"  {done}/{len(frames)}", end="\r")
    torch.cuda.synchronize(); dt = time.perf_counter() - t0; print()

    # 3) 재결합 + 오디오
    result = os.path.join(args.out, "result.mp4")
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", os.path.join(fout, "f_%05d.png")]
    if has_audio: cmd += ["-i", audio, "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", result]
    run(cmd)

    dur = len(frames)/args.fps
    print(f"\nDONE -> {result}")
    print(f"[SPEED] 스타일화 {dt:.1f}s / 영상 {dur:.1f}s = {dt/dur:.2f}x realtime  "
          f"({'✅ 1:2 OK' if dt <= 2*dur else '❌ over 2x'})")

if __name__ == "__main__":
    main()
