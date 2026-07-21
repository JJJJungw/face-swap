#!/usr/bin/env python3
"""얼굴 검출(YOLOX ONNX) → 크면 카툰 / 작으면 블러 → 타원 페더 합성 → 영상
- 검출: face-deid detector.py+policy.py 독립 재현. TensorRT(--trt) 또는 CUDA.
- 카툰: animegan2(MIT) 슬롯 + 색감 매칭(--color-match).
- 합성: 타원 페더, 크기분기(--cartoon-min).
- 인코딩: ffmpeg 직결 파이프(NVENC 옵션) — PNG 중간파일 없음.
사용법:
  python run/deid_cartoon.py --video input/swap2.mp4 --trt --encoder nvenc --color-match 0.5
"""
import os, sys, argparse, subprocess, ssl, urllib.request, time
import numpy as np, cv2

# ============ 검출 파라미터 (face-deid presets "default") ============
DET_LOW=0.20; NMS=0.45; MIN_SIZE=19; MAX_FRAC=0.90; BIG_FRAC=0.45; BIG_CONF=0.5

def preproc(img, size, pad=114):
    h, w = img.shape[:2]; r = min(size/h, size/w)
    nh, nw = int(round(h*r)), int(round(w*r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad, dtype=np.uint8); canvas[:nh, :nw] = resized
    return canvas.transpose(2, 0, 1)[None].astype(np.float32), r

def _preload_trt_libs():
    """TensorRT .so를 RTLD_GLOBAL로 미리 로드 → ORT TRT EP가 libnvinfer 찾게.
    tensorrt_libs / tensorrt_cu13_libs 등 모듈명이 달라도 site-packages에서 검색."""
    import glob, ctypes, site
    dirs = []
    for mod in ("tensorrt_libs", "tensorrt_cu13_libs", "tensorrt_cu12_libs"):
        try:
            m = __import__(mod); dirs.append(os.path.dirname(m.__file__)); break
        except Exception: pass
    if not dirs:                              # 모듈 import 실패 시 파일 검색 폴백
        roots = list(site.getsitepackages()) + [site.getusersitepackages()]
        for sp in roots:
            for so in glob.glob(os.path.join(sp, "**", "libnvinfer*.so*"), recursive=True):
                dirs.append(os.path.dirname(so)); break
            if dirs: break
    for d in dirs:
        for _ in range(2):
            for so in sorted(glob.glob(os.path.join(d, "*.so*"))):
                try: ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError: pass

def build_providers(model_path, use_trt):
    if not use_trt:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    _preload_trt_libs()
    cache = os.path.join(os.path.dirname(os.path.abspath(model_path)) or ".", "trt_cache")
    os.makedirs(cache, exist_ok=True)
    trt = ("TensorrtExecutionProvider", {
        "trt_fp16_enable": True, "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache, "trt_timing_cache_enable": True})
    return [trt, "CUDAExecutionProvider", "CPUExecutionProvider"]

class Detector:
    def __init__(self, model, size=1280, use_trt=False):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(model, providers=build_providers(model, use_trt))
        if not use_trt and self.sess.get_providers() == ["CPUExecutionProvider"]:
            self.sess = ort.InferenceSession(model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name; self.size = size
        print("detector providers:", self.sess.get_providers())
        warm = np.zeros((1, 3, size, size), dtype=np.float32)
        for _ in range(3): self.sess.run(None, {self.inp: warm})   # TRT면 첫 실행에 엔진 빌드(느림)

    def infer(self, frame):
        blob, r = preproc(frame, self.size)
        out = self.sess.run(None, {self.inp: blob})[0][0]
        sc = out[:, 4] * out[:, 5]
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

# ============ 카툰 스타일러 (교체 슬롯) — animegan2 placeholder ============
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
    def __init__(self, ckpt="gan_ckpt", half=False, gan_size=0):
        ensure_animegan(ckpt); sys.path.insert(0, ckpt)
        import torch; from model import Generator
        self.torch = torch; self.half = half; self.gan_size = gan_size
        self.m = Generator().to("cuda").eval()
        self.m.load_state_dict(torch.load(os.path.join(ckpt, "face_paint_512_v2.pt"), map_location="cuda"))
        if half: self.m.half()
        self.t = 0.0; self.n = 0                    # 계측: 누적 시간/호출수

    def stylize(self, face_bgr):
        torch = self.torch; t0 = time.perf_counter()
        img = cv2.resize(face_bgr, (self.gan_size, self.gan_size), interpolation=cv2.INTER_AREA) \
              if self.gan_size else face_bgr        # GAN 입력 고정크기(작을수록 빠름)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).mul(2).sub(1).unsqueeze(0).to("cuda")
        if self.half: x = x.half()
        with torch.no_grad():
            y = (self.m(x)[0].float()*0.5+0.5).clamp(0, 1)
        out = (y.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
        self.t += time.perf_counter()-t0; self.n += 1
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

# ============ 색감 매칭 + 블러 + 합성 ============
def color_transfer(src, ref, strength=1.0):
    if strength <= 0: return src
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    for i in range(3):
        sm, ss = s[:, :, i].mean(), s[:, :, i].std()+1e-6
        rm, rs = r[:, :, i].mean(), r[:, :, i].std()+1e-6
        s[:, :, i] = (s[:, :, i]-sm)/ss*rs+rm
    m = cv2.cvtColor(np.clip(s, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return m if strength >= 1.0 else cv2.addWeighted(m, strength, src, 1-strength, 0)

def blur_crop(crop, mode="pixelate", cap=12):
    h, w = crop.shape[:2]
    if mode == "box": return np.zeros_like(crop)
    if mode == "gaussian":
        sw, sh = max(1, min(w//16, cap)), max(1, min(h//16, cap))
        return cv2.resize(cv2.resize(crop, (sw, sh)), (w, h))
    blocks = max(1, min(min(w, h)//10, cap))
    return cv2.resize(cv2.resize(crop, (blocks, blocks)), (w, h), interpolation=cv2.INTER_NEAREST)

def composite(frame, boxes, stylizer, cartoon_min, blur_mode="pixelate", expand=0.15, color_match=0.0):
    H, W = frame.shape[:2]; nc = nb = 0
    for x1, y1, x2, y2, sc in boxes:
        bw, bh = x2-x1, y2-y1; size = max(bw, bh)
        cx1 = int(max(0, x1-bw*expand)); cy1 = int(max(0, y1-bh*expand))
        cx2 = int(min(W, x2+bw*expand)); cy2 = int(min(H, y2+bh*expand))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0: continue
        if size >= cartoon_min:
            proc = color_transfer(cv2.resize(stylizer.stylize(crop), (cx2-cx1, cy2-cy1)), crop, color_match); nc += 1
        else:
            proc = blur_crop(crop, blur_mode); nb += 1
        mask = np.zeros((cy2-cy1, cx2-cx1), dtype=np.uint8)
        ecx, ecy = (cx2-cx1)//2, (cy2-cy1)//2
        cv2.ellipse(mask, (ecx, ecy), (max(1, ecx-2), max(1, ecy-2)), 0, 0, 360, 255, -1)
        fk = min(31, max(5, (min(cx2-cx1, cy2-cy1)//8) | 1))
        m = cv2.GaussianBlur(mask, (fk, fk), 0).astype(np.float32)/255.0
        frame[cy1:cy2, cx1:cx2] = (crop*(1-m[:, :, None]) + proc*m[:, :, None]).astype(np.uint8)
    return nc, nb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="models/base_v2f2_1280_fp16.onnx")
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--cartoon-min", type=int, default=150, dest="cartoon_min", help="이 픽셀 이상 → 카툰, 미만 → 블러")
    ap.add_argument("--blur-mode", default="pixelate", choices=["pixelate", "gaussian", "box"], dest="blur_mode")
    ap.add_argument("--color-match", type=float, default=0.0, dest="color_match", help="원본 색감 매칭 0~1")
    ap.add_argument("--trt", action="store_true", help="TensorRT 검출(첫 실행은 엔진 빌드로 느림)")
    ap.add_argument("--encoder", default="nvenc", choices=["nvenc", "x264"], help="영상 인코더")
    ap.add_argument("--half", action="store_true", help="카툰 GAN fp16(속도↑)")
    ap.add_argument("--gan-size", type=int, default=0, dest="gan_size", help="GAN 입력 고정크기(예 384). 0=크롭 원본")
    ap.add_argument("--max-frames", type=int, default=0, dest="max_frames", help="N프레임만 처리(측정용). 0=전체")
    ap.add_argument("--out", default="out/deid_cartoon.mp4")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    det = Detector(args.model, args.size, use_trt=args.trt)
    styl = AnimeGAN(half=args.half, gan_size=args.gan_size)

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 30.0; total = int(cap.get(7))
    print(f"{W}x{H} @ {fps:.0f}fps, {total} frames | enc={args.encoder} trt={args.trt} half={args.half} gan_size={args.gan_size}")

    # ffmpeg 직결 파이프. -loglevel error/-nostats: 인코더 진행률 노이즈 억제(우리 fps만 출력).
    enc = (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"] if args.encoder == "nvenc"
           else ["-c:v", "libx264", "-crf", "23"])
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-nostats",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}",
           "-i", "-", "-i", args.video, "-map", "0:v", "-map", "1:a?"] + enc + \
          ["-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    i = 0; tot_c = tot_b = 0; t_det = t_comp = t_write = 0.0; t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok: break
        i += 1
        a = time.perf_counter()
        boxes = det.detect(frame, W, H)
        b = time.perf_counter()
        nc, nb = composite(frame, boxes, styl, args.cartoon_min, args.blur_mode, color_match=args.color_match)
        c = time.perf_counter()
        proc.stdin.write(frame.tobytes())
        d = time.perf_counter()
        t_det += b-a; t_comp += c-b; t_write += d-c; tot_c += nc; tot_b += nb
        if i % 20 == 0: print(f"  {i}/{total}  카툰{tot_c}/블러{tot_b}  {i/(time.perf_counter()-t0):.1f}fps", end="\r")
        if args.max_frames and i >= args.max_frames: break
    cap.release(); proc.stdin.close(); proc.wait()
    dt = time.perf_counter()-t0; vid = i/fps if fps else 0; print()
    print(f"DONE {dt:.1f}s / video {vid:.1f}s = {dt/vid:.2f}x realtime  (카툰{tot_c}/블러{tot_b}) -> {args.out}")
    # ── 단계별 평균(프레임당 ms). 스타일화는 GAN 자체 누적시간(styl.t)에서 분리 ──
    ms = lambda s: 1000*s/max(i, 1)
    t_styl = styl.t; t_comp_cpu = t_comp - t_styl
    print(f"프레임당 ms: 검출 {ms(t_det):.1f} | 합성 {ms(t_comp):.1f}"
          f"(그중 GAN {ms(t_styl):.1f}, CPU합성 {ms(t_comp_cpu):.1f}) | 인코딩write {ms(t_write):.1f}")
    print(f"GAN 호출 {styl.n}회, 호출당 {1000*styl.t/max(styl.n,1):.1f}ms")

if __name__ == "__main__":
    main()
