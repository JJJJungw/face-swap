#!/usr/bin/env python3
"""[실험] 트랙 히스테리시스로 블러↔카툰 튐 제거.
deid_cartoon.py(검출·GAN·색감·블러) + track_probe.py(IoU 트래커) 재활용.
트랙별로 카툰/블러 모드를 '스티키'하게 유지(히스테리시스+크기 스무딩) → 경계 깜빡임 제거.
별도 파일: 별로면 삭제, 괜찮으면 deid_cartoon에 컴포넌트로 흡수.

사용:
  bash run/run_track.sh --video input/x.mp4 --trt --gan-backend onnx --debug
"""
import os, sys, argparse, subprocess, time, collections
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deid_cartoon import Detector, AnimeGANONNX, AnimeGAN, color_transfer, blur_crop
from track_probe import IoUTracker

class ModeState:
    """트랙별 카툰/블러 모드를 히스테리시스로 유지. 순간 크기 대신 최근 median으로 판정."""
    def __init__(self, hi=165, lo=135, smooth=5):
        self.hi = hi; self.lo = lo; self.smooth = smooth
        self.hist = {}   # tid -> deque(sizes)
        self.mode = {}   # tid -> "cartoon"|"blur"

    def decide(self, tid, size):
        h = self.hist.setdefault(tid, collections.deque(maxlen=self.smooth))
        h.append(size)
        med = sorted(h)[len(h)//2]
        m = self.mode.get(tid)
        if m is None:                                   # 첫 등장: 밴드 중앙 기준
            m = "cartoon" if med >= (self.hi + self.lo)//2 else "blur"
        elif m == "blur" and med >= self.hi:            # 충분히 커져야 카툰 진입
            m = "cartoon"
        elif m == "cartoon" and med < self.lo:          # 충분히 작아져야 블러 강등
            m = "blur"
        self.mode[tid] = m
        return m, med

def render_face(frame, box, mode, styl, blur_mode, expand, color_match, W, H):
    x1, y1, x2, y2 = box; bw, bh = x2-x1, y2-y1
    cx1 = int(max(0, x1-bw*expand)); cy1 = int(max(0, y1-bh*expand))
    cx2 = int(min(W, x2+bw*expand)); cy2 = int(min(H, y2+bh*expand))
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0: return
    if mode == "cartoon":
        s = cv2.resize(styl.stylize(crop), (cx2-cx1, cy2-cy1), interpolation=cv2.INTER_LANCZOS4)
        proc = color_transfer(s, crop, color_match)
    else:
        proc = blur_crop(crop, blur_mode)
    mask = np.zeros((cy2-cy1, cx2-cx1), np.uint8)
    ecx, ecy = (cx2-cx1)//2, (cy2-cy1)//2
    cv2.ellipse(mask, (ecx, ecy), (max(1, ecx-2), max(1, ecy-2)), 0, 0, 360, 255, -1)
    fk = min(31, max(5, (min(cx2-cx1, cy2-cy1)//8) | 1))
    m = cv2.GaussianBlur(mask, (fk, fk), 0).astype(np.float32)/255.0
    frame[cy1:cy2, cx1:cx2] = (crop*(1-m[:, :, None]) + proc*m[:, :, None]).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="models/base_v2f2_1280_fp16.onnx")
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--trt", action="store_true")
    ap.add_argument("--gan-backend", default="onnx", choices=["torch", "onnx"], dest="gan_backend")
    ap.add_argument("--gan-onnx", default="gan_ckpt/animegan_512.onnx", dest="gan_onnx")
    ap.add_argument("--gan-onnx-size", type=int, default=512, dest="gan_onnx_size")
    ap.add_argument("--encoder", default="nvenc", choices=["nvenc", "x264"])
    ap.add_argument("--blur-mode", default="pixelate", choices=["pixelate", "gaussian", "box"], dest="blur_mode")
    ap.add_argument("--color-match", type=float, default=0.0, dest="color_match")
    ap.add_argument("--expand", type=float, default=0.15)
    # 히스테리시스 밴드 (lo 밑=블러, hi 위=카툰, 사이=직전 유지)
    ap.add_argument("--hi", type=int, default=165, help="블러→카툰 진입 임계")
    ap.add_argument("--lo", type=int, default=135, help="카툰→블러 강등 임계")
    ap.add_argument("--smooth", type=int, default=5, help="크기 median 프레임 수")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--max-gap", type=int, default=5, dest="max_gap")
    ap.add_argument("--scene-cut", type=float, default=55.0, dest="scene_cut", help="장면전환 임계(0=끔)")
    ap.add_argument("--debug", action="store_true", help="모드/ID/크기 오버레이")
    ap.add_argument("--max-frames", type=int, default=0, dest="max_frames")
    ap.add_argument("--out", default="out/deid_track.mp4")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    det = Detector(args.model, args.size, use_trt=args.trt)
    if args.gan_backend == "onnx":
        styl = AnimeGANONNX(args.gan_onnx, size=args.gan_onnx_size, use_trt=args.trt)
    else:
        styl = AnimeGAN(gan_size=512)
    trk = IoUTracker(args.iou, args.max_gap)
    ms = ModeState(args.hi, args.lo, args.smooth)

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 30.0; total = int(cap.get(7))
    print(f"{W}x{H} @ {fps:.0f}fps, {total} frames | hi={args.hi} lo={args.lo} smooth={args.smooth} scene_cut={args.scene_cut}")

    enc = (["-c:v","h264_nvenc","-preset","p4","-cq","23"] if args.encoder=="nvenc" else ["-c:v","libx264","-crf","23"])
    cmd = ["ffmpeg","-y","-loglevel","error","-nostats","-f","rawvideo","-pix_fmt","bgr24",
           "-s",f"{W}x{H}","-r",f"{fps}","-i","-","-i",args.video,"-map","0:v","-map","1:a?"] + enc + \
          ["-pix_fmt","yuv420p","-c:a","aac","-shortest", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    idx = ncut = nc = nb = nflip = 0; prev_g = None; last_mode = {}; t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok: break
        idx += 1
        if args.scene_cut > 0:
            g = cv2.cvtColor(cv2.resize(frame, (64,36)), cv2.COLOR_BGR2GRAY)
            if prev_g is not None and np.abs(g.astype(np.int16)-prev_g.astype(np.int16)).mean() > args.scene_cut:
                trk.reset(); ncut += 1
            prev_g = g
        for tid, box, size in trk.step(det.detect(frame, W, H), idx):
            mode, med = ms.decide(tid, size)
            if last_mode.get(tid) not in (None, mode): nflip += 1     # 실제 전환 횟수(튐 지표)
            last_mode[tid] = mode
            if mode == "cartoon": nc += 1
            else: nb += 1
            render_face(frame, box, mode, styl, args.blur_mode, args.expand, args.color_match, W, H)
            if args.debug:
                x1, y1, x2, y2 = [int(v) for v in box]
                col = (0,220,0) if mode == "cartoon" else (0,140,255)
                cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
                cv2.putText(frame, f"ID{tid} {med}px {mode}", (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        proc.stdin.write(frame.tobytes())
        if idx % 30 == 0: print(f"  {idx}/{total}  {idx/(time.perf_counter()-t0):.1f}fps", end="\r")
        if args.max_frames and idx >= args.max_frames: break
    cap.release(); proc.stdin.close(); proc.wait()
    dt = time.perf_counter()-t0; vid = idx/fps if fps else 0; print()
    print(f"DONE {dt:.1f}s / video {vid:.1f}s = {dt/vid:.2f}x realtime  카툰{nc}/블러{nb}  컷{ncut}  모드전환{nflip}회 -> {args.out}")
    print("(모드전환 횟수가 낮을수록 튐이 적음 — 밴드 hi/lo 넓히면 더 줄어듦)")

if __name__ == "__main__":
    main()
