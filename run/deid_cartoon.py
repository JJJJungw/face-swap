#!/usr/bin/env python3
"""얼굴 검출(YOLOX ONNX) → 크기 임계값 이상 얼굴만 카툰화 → 타원 페더 합성 → 영상
- 검출: face-deidentification의 detector.py + policy.py 로직을 독립 재현 (onnxruntime, 그쪽 레포 의존 X)
- 카툰: 교체 가능한 스타일러 슬롯. 지금은 animegan2(파이프라인 메커니즘 검증용 placeholder)
        → 나중에 Flux 증류로 만든 3D카툰 학생모델(ONNX)로 교체.
- 합성: blur.py의 타원 페더 마스크 방식 재현.
사용법:
  python run/deid_cartoon.py --video input/swap1.mp4 --min-face 60
"""
import os, sys, argparse, subprocess, shutil, ssl, urllib.request, time
import numpy as np, cv2

# ============ 검출 파라미터 (face-deid presets.py "default" 재현) ============
DET_LOW = 0.20; NMS = 0.45; MIN_SIZE = 19
MAX_FRAC = 0.90; BIG_FRAC = 0.45; BIG_CONF = 0.5

def preproc(img, size, pad=114):
    """레터박스 전처리 (face-deid detector.preproc 동일)."""
    h, w = img.shape[:2]; r = min(size/h, size/w)
    nh, nw = int(round(h*r)), int(round(w*r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad, dtype=np.uint8); canvas[:nh, :nw] = resized
    return canvas.transpose(2, 0, 1)[None].astype(np.float32), r

class Detector:
    def __init__(self, model, size=1280):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name; self.size = size
        warm = np.zeros((1, 3, size, size), dtype=np.float32)
        for _ in range(2): self.sess.run(None, {self.inp: warm})
        print("detector providers:", self.sess.get_providers())

    def infer(self, frame):
        blob, r = preproc(frame, self.size)
        out = self.sess.run(None, {self.inp: blob})[0][0]     # [N, 4+1+cls]
        sc = out[:, 4] * out[:, 5]                             # obj × face_cls
        cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        x1 = (cx-bw/2)/r; y1 = (cy-bh/2)/r; x2 = (cx+bw/2)/r; y2 = (cy+bh/2)/r
        return np.stack([x1, y1, x2, y2, sc], axis=1)

    def detect(self, frame, W, H):
        cand = self.infer(frame)
        x1, y1, x2, y2, sc = cand.T
        keep = sc > DET_LOW
        if not keep.any(): return []
        x1, y1, x2, y2, sc = x1[keep], y1[keep], x2[keep], y2[keep], sc[keep]
        xywh = np.stack([x1, y1, x2-x1, y2-y1], axis=1)
        idxs = cv2.dnn.NMSBoxes(xywh.tolist(), sc.tolist(), DET_LOW, NMS)
        res = []
        for i in (np.array(idxs).flatten() if len(idxs) else []):
            w_, h_ = float(x2[i]-x1[i]), float(y2[i]-y1[i]); size = max(w_, h_)
            if size < MIN_SIZE: continue
            if w_ > MAX_FRAC*W and h_ > MAX_FRAC*H: continue
            if size > BIG_FRAC*min(W, H) and float(sc[i]) < BIG_CONF: continue
            res.append([float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(sc[i])])
        return res

# ============ 카툰 스타일러 (교체 가능 슬롯) — 지금은 animegan2 placeholder ============
def ensure_animegan(ckpt="gan_ckpt"):
    os.makedirs(ckpt, exist_ok=True)
    ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
    base = "https://raw.githubusercontent.com/bryandlee/animegan2-pytorch"
    for name, url in {"model.py": f"{base}/master/model.py",
                      "face_paint_512_v2.pt": f"{base}/main/weights/face_paint_512_v2.pt"}.items():
        d = os.path.join(ckpt, name)
        if not os.path.exists(d):
            with urllib.request.urlopen(url, context=ctx, timeout=90) as r, open(d, "wb") as f: f.write(r.read())

class AnimeGAN:
    def __init__(self, ckpt="gan_ckpt"):
        ensure_animegan(ckpt); sys.path.insert(0, ckpt)
        import torch; from model import Generator
        self.torch = torch
        self.m = Generator().to("cuda").eval()
        self.m.load_state_dict(torch.load(os.path.join(ckpt, "face_paint_512_v2.pt"), map_location="cuda"))

    def stylize(self, face_bgr):
        torch = self.torch
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).mul(2).sub(1).unsqueeze(0).to("cuda")
        with torch.no_grad():
            y = (self.m(t)[0]*0.5+0.5).clamp(0, 1)            # batch=1 (cu130 배치버그 회피)
        out = (y.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

# ============ 색감 매칭 (Reinhard LAB color transfer) ============
def color_transfer(src, ref, strength=1.0):
    """src(스타일화 결과)의 색 통계를 ref(원본 크롭)에 맞춤 → 머리색·피부톤 정렬.
    strength: 0=끔, 1=완전일치, 그 사이는 부분."""
    if strength <= 0:
        return src
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    for i in range(3):
        sm, ss = s[:, :, i].mean(), s[:, :, i].std() + 1e-6
        rm, rs = r[:, :, i].mean(), r[:, :, i].std() + 1e-6
        s[:, :, i] = (s[:, :, i] - sm) / ss * rs + rm
    matched = cv2.cvtColor(np.clip(s, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return matched if strength >= 1.0 else cv2.addWeighted(matched, strength, src, 1 - strength, 0)

# ============ 크롭 → 스타일화 → 타원 페더 합성 (blur.py 방식) ============
def composite(frame, boxes, stylizer, min_face, expand=0.15, color_match=0.0):
    H, W = frame.shape[:2]
    for x1, y1, x2, y2, sc in boxes:
        bw, bh = x2-x1, y2-y1
        if max(bw, bh) < min_face:            # 크기 임계값 이상만
            continue
        cx1 = int(max(0, x1-bw*expand)); cy1 = int(max(0, y1-bh*expand))
        cx2 = int(min(W, x2+bw*expand)); cy2 = int(min(H, y2+bh*expand))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0: continue
        styl = cv2.resize(stylizer.stylize(crop), (cx2-cx1, cy2-cy1))
        styl = color_transfer(styl, crop, color_match)   # 원본 색감에 맞춤
        mask = np.zeros((cy2-cy1, cx2-cx1), dtype=np.uint8)
        ecx, ecy = (cx2-cx1)//2, (cy2-cy1)//2
        cv2.ellipse(mask, (ecx, ecy), (max(1, ecx-2), max(1, ecy-2)), 0, 0, 360, 255, -1)
        fk = min(31, max(5, (min(cx2-cx1, cy2-cy1)//8) | 1))
        m = cv2.GaussianBlur(mask, (fk, fk), 0).astype(np.float32)/255.0
        frame[cy1:cy2, cx1:cx2] = (crop*(1-m[:, :, None]) + styl*m[:, :, None]).astype(np.uint8)
    return frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="models/base_v2f2_1280_fp16.onnx")
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--min-face", type=int, default=60, dest="min_face", help="이 픽셀 이상 얼굴만 카툰화")
    ap.add_argument("--color-match", type=float, default=0.0, dest="color_match",
                    help="원본 색감에 맞추기 0~1 (0=끔, 0.5=절반, 1=완전일치)")
    ap.add_argument("--out", default="out/deid_cartoon.mp4")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    det = Detector(args.model, args.size)
    styl = AnimeGAN()

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 30.0; total = int(cap.get(7))
    print(f"{W}x{H} @ {fps:.0f}fps, {total} frames")
    tmp = "out/_frames_deid"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)

    i = 0; t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok: break
        i += 1
        boxes = det.detect(frame, W, H)
        composite(frame, boxes, styl, args.min_face, color_match=args.color_match)
        cv2.imwrite(f"{tmp}/f_{i:05d}.png", frame)
        if i % 10 == 0: print(f"  {i}/{total}", end="\r")
    cap.release(); dt = time.perf_counter()-t0; print()

    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", f"{tmp}/f_%05d.png",
                    "-i", args.video, "-map", "0:v", "-map", "1:a?",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", args.out], check=True)
    vid = i/fps if fps else 0
    print(f"DONE {dt:.1f}s / video {vid:.1f}s = {dt/vid:.2f}x realtime -> {args.out}")

if __name__ == "__main__":
    main()
