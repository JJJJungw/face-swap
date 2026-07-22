#!/usr/bin/env python3
"""[실험] 카툰 출력 선명화 후처리 비교.
--sharpen {none,unsharp,detail,bilateral,realesrgan} 로 얼굴 카툰 출력 후처리를 바꿔가며 비교.
deid_track의 트래커+히스테리시스는 그대로, '512 출력→박스 확대' 단계만 각 방식으로 교체.
제일 좋은 하나 고른 뒤 deid_cartoon에 흡수(별로면 이 파일만 삭제).

  bash run/run_sharpen.sh --video input/swap4.mp4 --trt --gan-backend onnx --sharpen realesrgan
"""
import os, sys, argparse, subprocess, time, ssl, urllib.request
import numpy as np, cv2
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deid_cartoon import Detector, AnimeGANONNX, AnimeGAN, color_transfer, blur_crop
from track_probe import IoUTracker
from deid_track import ModeState

# ===== Real-ESRGAN anime (RRDBNet 6B, BSD-3) — basicsr/torchvision 없이 순수 torch =====
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1); self.conv2 = nn.Conv2d(nf+gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf+2*gc, gc, 3, 1, 1); self.conv4 = nn.Conv2d(nf+3*gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf+4*gc, nf, 3, 1, 1); self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x)); x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc); self.rdb2 = ResidualDenseBlock(nf, gc); self.rdb3 = ResidualDenseBlock(nf, gc)
    def forward(self, x):
        out = self.rdb1(x); out = self.rdb2(out); out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))

class RealESRGANAnime:
    URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    def __init__(self, ckpt="gan_ckpt/realesrgan_anime_6b.pth"):
        os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
        if not os.path.exists(ckpt):
            ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
            print("[realesrgan] 가중치 다운로드...", flush=True)
            with urllib.request.urlopen(self.URL, context=ctx, timeout=180) as r, open(ckpt, "wb") as f: f.write(r.read())
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("params_ema", sd.get("params", sd)) if isinstance(sd, dict) else sd
        self.m = RRDBNet().cuda().eval().half()
        self.m.load_state_dict(sd, strict=True)
        self.t = 0.0; self.n = 0
        print("[realesrgan] anime 6B 로드 완료(BSD-3)")

    @torch.no_grad()
    def enhance(self, bgr):                          # x4 업스케일(512→2048), 애니 라인 선명화
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda().half()
        y = self.m(x).clamp(0, 1)[0]
        out = (y.permute(1, 2, 0).float().cpu().numpy()*255).round().astype(np.uint8)
        self.t += time.perf_counter()-t0; self.n += 1
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

# ===== 후처리: 512 카툰 출력 → 박스 크기 =====
def to_box(c, w, h, mode, sr):
    if mode == "realesrgan" and sr is not None:
        c = sr.enhance(c)                            # 512→2048 선명 업스케일 후 박스로
    out = cv2.resize(c, (w, h), interpolation=cv2.INTER_LANCZOS4)
    if mode == "unsharp":
        b = cv2.GaussianBlur(out, (0, 0), 2.0); out = cv2.addWeighted(out, 1.5, b, -0.5, 0)
    elif mode == "detail":
        out = cv2.detailEnhance(out, sigma_s=8, sigma_r=0.15)
    elif mode == "bilateral":
        out = cv2.bilateralFilter(out, 7, 50, 50)
        b = cv2.GaussianBlur(out, (0, 0), 1.5); out = cv2.addWeighted(out, 1.4, b, -0.4, 0)
    return out

def render_face(frame, box, cmode, styl, sharpen, sr, blur_mode, expand, color_match, W, H):
    x1, y1, x2, y2 = box; bw, bh = x2-x1, y2-y1
    cx1 = int(max(0, x1-bw*expand)); cy1 = int(max(0, y1-bh*expand))
    cx2 = int(min(W, x2+bw*expand)); cy2 = int(min(H, y2+bh*expand))
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0: return
    if cmode == "cartoon":
        c = styl.stylize(crop)                       # 정사각 스타일 출력(512)
        proc = color_transfer(to_box(c, cx2-cx1, cy2-cy1, sharpen, sr), crop, color_match)
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
    ap.add_argument("--sharpen", default="none", choices=["none", "unsharp", "detail", "bilateral", "realesrgan"])
    ap.add_argument("--encoder", default="nvenc", choices=["nvenc", "x264"])
    ap.add_argument("--blur-mode", default="pixelate", choices=["pixelate", "gaussian", "box"], dest="blur_mode")
    ap.add_argument("--color-match", type=float, default=0.0, dest="color_match")
    ap.add_argument("--expand", type=float, default=0.15)
    ap.add_argument("--cartoon-min", type=int, default=150, dest="cartoon_min", help="히스테리시스 없이 단순 임계(참고)")
    ap.add_argument("--hi", type=int, default=165); ap.add_argument("--lo", type=int, default=135)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--iou", type=float, default=0.3); ap.add_argument("--max-gap", type=int, default=5, dest="max_gap")
    ap.add_argument("--scene-cut", type=float, default=55.0, dest="scene_cut")
    ap.add_argument("--max-frames", type=int, default=0, dest="max_frames")
    ap.add_argument("--out", default="out/deid_sharpen.mp4")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    det = Detector(args.model, args.size, use_trt=args.trt)
    styl = AnimeGANONNX(args.gan_onnx, size=args.gan_onnx_size, use_trt=args.trt) if args.gan_backend == "onnx" else AnimeGAN(gan_size=512)
    sr = RealESRGANAnime() if args.sharpen == "realesrgan" else None
    trk = IoUTracker(args.iou, args.max_gap); ms = ModeState(args.hi, args.lo, args.smooth)

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 30.0; total = int(cap.get(7))
    print(f"{W}x{H} @ {fps:.0f}fps, {total} frames | sharpen={args.sharpen}")

    enc = (["-c:v","h264_nvenc","-preset","p4","-cq","20"] if args.encoder=="nvenc" else ["-c:v","libx264","-crf","18"])
    cmd = ["ffmpeg","-y","-loglevel","error","-nostats","-f","rawvideo","-pix_fmt","bgr24",
           "-s",f"{W}x{H}","-r",f"{fps}","-i","-","-i",args.video,"-map","0:v","-map","1:a?"] + enc + \
          ["-pix_fmt","yuv420p","-c:a","aac","-shortest", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    idx = nc = nb = 0; prev_g = None; t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok: break
        idx += 1
        if args.scene_cut > 0:
            g = cv2.cvtColor(cv2.resize(frame, (64,36)), cv2.COLOR_BGR2GRAY)
            if prev_g is not None and np.abs(g.astype(np.int16)-prev_g.astype(np.int16)).mean() > args.scene_cut:
                trk.reset()
            prev_g = g
        for tid, box, size in trk.step(det.detect(frame, W, H), idx):
            mode, _ = ms.decide(tid, size)
            if mode == "cartoon": nc += 1
            else: nb += 1
            render_face(frame, box, mode, styl, args.sharpen, sr, args.blur_mode, args.expand, args.color_match, W, H)
        proc.stdin.write(frame.tobytes())
        if idx % 30 == 0: print(f"  {idx}/{total}  {idx/(time.perf_counter()-t0):.1f}fps", end="\r")
        if args.max_frames and idx >= args.max_frames: break
    cap.release(); proc.stdin.close(); proc.wait()
    dt = time.perf_counter()-t0; vid = idx/fps if fps else 0; print()
    extra = f" | SR {1000*sr.t/max(sr.n,1):.1f}ms/face" if sr else ""
    print(f"DONE {dt:.1f}s / video {vid:.1f}s = {dt/vid:.2f}x realtime  카툰{nc}/블러{nb}  sharpen={args.sharpen}{extra} -> {args.out}")

if __name__ == "__main__":
    main()
