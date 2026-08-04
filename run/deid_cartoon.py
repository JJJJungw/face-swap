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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_utils import occupancy_crop_bounds

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
    """torch eager 백엔드. --compile 시 torch.compile(고정 입력크기 권장)."""
    def __init__(self, ckpt="gan_ckpt", half=False, gan_size=0, compile=False):
        ensure_animegan(ckpt); sys.path.insert(0, ckpt)
        import torch; from model import Generator
        self.torch = torch; self.half = half; self.gan_size = gan_size
        self.m = Generator().to("cuda").eval()
        self.m.load_state_dict(torch.load(os.path.join(ckpt, "face_paint_512_v2.pt"), map_location="cuda"))
        if half: self.m.half()
        if compile:
            self.m = torch.compile(self.m, mode="max-autotune")   # 첫 호출 컴파일로 느림→이후 빠름
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

# ============ GAN ONNX → TensorRT 백엔드 (가중치 동일 = 512 화질 그대로, 연산만 가속) ============
def export_animegan_onnx(onnx_path, ckpt="gan_ckpt", size=512):
    """AnimeGAN2 제너레이터를 고정 size로 ONNX export(정적 shape → TRT 정적엔진 최적)."""
    ensure_animegan(ckpt)
    if ckpt not in sys.path: sys.path.insert(0, ckpt)
    import torch; from model import Generator
    m = Generator().eval()
    m.load_state_dict(torch.load(os.path.join(ckpt, "face_paint_512_v2.pt"), map_location="cpu"))
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    dummy = torch.randn(1, 3, size, size)
    # torch 2.13 기본은 dynamo 익스포터(onnxscript 필요). legacy(TorchScript)가 TRT 친화적 + 의존성 없음.
    try:
        torch.onnx.export(m, dummy, onnx_path, input_names=["x"], output_names=["y"],
                          opset_version=17, dynamo=False)
    except TypeError:                       # 구버전 torch: dynamo 인자 없음
        torch.onnx.export(m, dummy, onnx_path, input_names=["x"], output_names=["y"], opset_version=17)
    print(f"[export] {onnx_path} (size={size})")

class AnimeGANONNX:
    """onnxruntime(+TensorRT EP) 백엔드. 고정 512 입력 → 검출기와 같은 TRT 파이프라인 재사용.
    가중치 동일하므로 출력은 torch@512와 사실상 같음(fp16 미세 오차뿐)."""
    def __init__(self, onnx_path="gan_ckpt/animegan_512.onnx", size=512, use_trt=True):
        self.size = size
        if not os.path.exists(onnx_path):
            export_animegan_onnx(onnx_path, size=size)
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=build_providers(onnx_path, use_trt))
        self.inp = self.sess.get_inputs()[0].name
        print("GAN providers:", self.sess.get_providers())
        warm = np.zeros((1, 3, size, size), dtype=np.float32)
        for _ in range(3): self.sess.run(None, {self.inp: warm})   # TRT면 첫 실행에 엔진 빌드(느림)
        self.t = 0.0; self.n = 0

    def stylize(self, face_bgr):
        t0 = time.perf_counter()
        img = cv2.resize(face_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255.0) * 2 - 1).transpose(2, 0, 1)[None]
        y = self.sess.run(None, {self.inp: x})[0][0]
        out = (np.clip(y * 0.5 + 0.5, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
        self.t += time.perf_counter() - t0; self.n += 1
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

def lowpass(image, sigma):
    """축소 → 작은 블러 → 확대. 큰 sigma 를 직접 돌리면 프레임당 수십~수백 ms 가 든다.

    저주파만 필요하므로 다운샘플해서 계산해도 결과가 사실상 같다.
    (측정: 534px 크롭에 sigma 53 직접 블러 = 약 80ms/회 → 이 방식 1ms 미만)
    """
    if sigma < 1.0:
        return image.copy()
    k = max(1, int(sigma / 4.0))
    if k <= 1:
        return cv2.GaussianBlur(image, (0, 0), sigma)
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(2, w // k), max(2, h // k)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / k)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def color_transfer_lowfreq(src, ref, strength=1.0, sigma_ratio=0.10, luma=0.5,
                           vivid_chroma=1.0, vivid_luma=1.0):
    """저주파(조명·색조)는 원본에서, 고주파(선·화풍)는 스타일 결과에서 가져온다.

    ■ 왜 전역 전이로는 안 되는가
      color_transfer 는 크롭 전체의 평균/표준편차만 맞춘다. 그런데 실제 장면에서는
      조명이 얼굴을 가로질러 변한다(한쪽만 분홍 조명, 역광 윤곽, 그림자).
      전역 보정은 그 기울기를 못 따라가서 얼굴 색이 주변과 따로 노는 이질감이 남는다.
      --color-match 를 1.0 까지 올려도 해소되지 않는 이유다.

      저주파를 통째로 갈아끼우면 조명·색 캐스트가 **공간적으로** 따라온다.

    ■ luma 를 왜 나눠 쓰는가
      밝기(L)의 저주파까지 원본에서 다 가져오면 사진의 명암 그라데이션이 되돌아와
      평평한 색면이 깨진다(= 화풍 손실). 색(a,b)은 전부 가져오되 밝기는 일부만 가져온다.
      luma=0 이면 화풍의 명암을 완전히 지키고, 1 이면 원본 조명을 그대로 따른다.
    """
    if strength <= 0:
        return src
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    sigma = max(3.0, min(src.shape[:2]) * sigma_ratio)
    s_lo = lowpass(s, sigma)
    r_lo = lowpass(r, sigma)
    weights = (float(luma), 1.0, 1.0)          # L, a, b
    out = s + np.stack([(r_lo[:, :, i] - s_lo[:, :, i]) * weights[i] for i in range(3)], 2)

    # [쨍함] 저주파 주위의 국소 진폭을 키운다.
    #   "둥둥 뜬다"(저주파 색조 불일치)와 "쨍하다"(국소 진폭)는 서로 다른 성분이라
    #   저주파를 원본에 맞추면서 진폭만 따로 올릴 수 있다. 저주파 기준으로 늘리므로
    #   색조는 그대로 유지되고 대비·채도만 올라간다. 1.0 이면 변화 없음.
    if vivid_chroma != 1.0 or vivid_luma != 1.0:
        base = lowpass(out, sigma)
        gains = (float(vivid_luma), float(vivid_chroma), float(vivid_chroma))
        out = base + np.stack([(out[:, :, i] - base[:, :, i]) * gains[i] for i in range(3)], 2)

    m = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return m if strength >= 1.0 else cv2.addWeighted(m, strength, src, 1 - strength, 0)


def denoise_crop(crop, strength):
    """스타일화 **전에** 입력의 압축 잡음·센서 노이즈를 지운다.

    학생은 h.264 블록과 피부 잡음을 '그려야 할 선'으로 오인해 붓질 자국을 만든다
    (영상 크롭에서 선 밀도가 teacher 대비 1.30배로 측정됨).
    입력에서 미리 지우면 증강 재학습과 같은 이득을 얻으면서 **출력 대비는 안 올라가므로**
    흔들림이 새로 드러나지 않는다. edge-preserving 이라 눈·입 윤곽은 남는다.
    """
    if strength <= 0:
        return crop
    # bilateral 은 커널이 커지면 급격히 느려진다. 축소해서 걸고 되돌린다.
    h, w = crop.shape[:2]
    k = max(1, min(h, w) // 256)
    small = cv2.resize(crop, (w // k, h // k), interpolation=cv2.INTER_AREA) if k > 1 else crop
    small = cv2.bilateralFilter(small, 5, 25.0 * strength, 25.0 * strength)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR) if k > 1 else small


def flatten_skin(image, strength, edge_keep=0.35):
    """스타일화 **후에** 저대비 영역(피부)만 고르게 편다. 선은 남긴다.

    눈 밑 주근깨 점, 뺨 얼룩처럼 작고 대비 높은 요소가 프레임마다 나타났다 사라지는 것이
    체감 불안정의 큰 부분이다. 고주파를 **줄이는** 방향이라 --sharpen 과 반대로
    흔들림을 키우지 않는다.
    """
    if strength <= 0:
        return image
    h, w = image.shape[:2]
    k = max(1, min(h, w) // 256)
    if k > 1:
        small = cv2.resize(image, (w // k, h // k), interpolation=cv2.INTER_AREA)
        smooth = cv2.resize(cv2.bilateralFilter(small, 7, 45.0, 45.0), (w, h),
                            interpolation=cv2.INTER_LINEAR)
    else:
        smooth = cv2.bilateralFilter(image, 7, 45.0, 45.0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                         cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    keep = np.clip(grad / (grad.max() * edge_keep + 1e-6), 0, 1)[:, :, None]
    blend = strength * (1.0 - keep)
    return (image * (1 - blend) + smooth * blend).astype(np.uint8)


def blur_crop(crop, mode="pixelate", cap=12):
    h, w = crop.shape[:2]
    if mode == "box": return np.zeros_like(crop)
    if mode == "gaussian":
        sw, sh = max(1, min(w//16, cap)), max(1, min(h//16, cap))
        return cv2.resize(cv2.resize(crop, (sw, sh)), (w, h))
    blocks = max(1, min(min(w, h)//10, cap))
    return cv2.resize(cv2.resize(crop, (blocks, blocks)), (w, h), interpolation=cv2.INTER_NEAREST)

class BoxSmoother:
    """검출 박스를 프레임 간 EMA 로 안정화한다.

    프레임마다 독립 검출하면 박스가 수 px 씩 떨리고, 크롭 좌표가 떨리면
    GAN 입력 샘플링이 달라져 출력 화풍·색이 프레임마다 튄다(번쩍임).
    IoU 로 직전 프레임의 같은 얼굴을 찾아 좌표를 지수평활한다.
    alpha=0 이면 평활 없음(원래 동작), 클수록 직전 프레임을 더 따른다.
    """

    def __init__(self, alpha=0.0, iou_threshold=0.3, max_missing=5):
        self.alpha = float(alpha)
        self.iou_threshold = float(iou_threshold)
        self.max_missing = int(max_missing)
        self.tracks = []            # [[x1,y1,x2,y2,score], missing]

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def __call__(self, boxes):
        if self.alpha <= 0:
            return boxes
        used, out = set(), []
        for box in boxes:
            best, best_iou = -1, self.iou_threshold
            for idx, (prev, _) in enumerate(self.tracks):
                if idx in used:
                    continue
                iou = self._iou(box, prev)
                if iou > best_iou:
                    best, best_iou = idx, iou
            if best >= 0:
                prev = self.tracks[best][0]
                a = self.alpha
                smoothed = [a * prev[k] + (1.0 - a) * box[k] for k in range(4)] + [box[4]]
                self.tracks[best] = [smoothed, 0]
                used.add(best)
                out.append(tuple(smoothed))
            else:
                self.tracks.append([list(box), 0])
                used.add(len(self.tracks) - 1)
                out.append(tuple(box))
        kept = []
        for idx, (prev, missing) in enumerate(self.tracks):
            if idx in used:
                kept.append([prev, 0])
            elif missing + 1 <= self.max_missing:
                kept.append([prev, missing + 1])
        self.tracks = kept
        return out


def sharpen_crop(image, amount, radius_ratio=0.006):
    """언샤프 마스크. 스타일화 결과의 선을 세운다.

    512 로 스타일화한 뒤 크롭 크기로 되돌리는 과정에서 선이 뭉개진다.
    radius 는 크롭 크기에 비례해야 얼굴 크기가 달라져도 같은 세기로 보인다.
    amount 가 크면 아티팩트·노이즈도 같이 증폭되므로 0.3~0.8 이 실용 범위다.
    """
    if amount <= 0:
        return image
    sigma = max(0.8, min(image.shape[:2]) * radius_ratio)
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def composite(frame, boxes, stylizer, cartoon_min, blur_mode="pixelate", expand=0.15,
              color_match=0.0, square_crop=False, mask_mode="crop-ellipse",
              mask_scale_x=0.92, mask_scale_y=1.0, mask_feather=0.04, occupancy=0.0,
              sharpen=0.0, temporal=0.0, prev_frame=None,
              color_mode="global", color_sigma=0.10, color_luma=0.5,
              vivid_chroma=1.0, vivid_luma=1.0,
              input_denoise=0.0, input_temporal=0.0, prev_raw=None, flatten=0.0):
    H, W = frame.shape[:2]; nc = nb = 0
    for x1, y1, x2, y2, sc in boxes:
        bw, bh = x2-x1, y2-y1; size = max(bw, bh)
        if square_crop:
            # 학습 크롭과 같은 규칙. occupancy 를 주면 얼굴 면적비로 정한다.
            if occupancy > 0:
                ox1, oy1, ox2, oy2 = occupancy_crop_bounds((x1, y1, x2, y2), occupancy)
                side = float(ox2 - ox1)
            else:
                side = size * (1.0 + 2.0 * expand)
            center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            cx1 = int(np.floor(center_x - side * 0.5))
            cy1 = int(np.floor(center_y - side * 0.5))
            cx2 = int(np.ceil(center_x + side * 0.5))
            cy2 = int(np.ceil(center_y + side * 0.5))
            pad_left, pad_top = max(0, -cx1), max(0, -cy1)
            pad_right, pad_bottom = max(0, cx2-W), max(0, cy2-H)
            padded = cv2.copyMakeBorder(
                frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101,
            )
            crop = padded[
                cy1+pad_top:cy2+pad_top,
                cx1+pad_left:cx2+pad_left,
            ]
            px1, py1, px2, py2 = max(0, cx1), max(0, cy1), min(W, cx2), min(H, cy2)
            ox1, oy1 = px1-cx1, py1-cy1
            ox2, oy2 = ox1+(px2-px1), oy1+(py2-py1)
        else:
            cx1 = int(max(0, x1-bw*expand)); cy1 = int(max(0, y1-bh*expand))
            cx2 = int(min(W, x2+bw*expand)); cy2 = int(min(H, y2+bh*expand))
            crop = frame[cy1:cy2, cx1:cx2]
            px1, py1, px2, py2 = cx1, cy1, cx2, cy2
            ox1 = oy1 = 0
            ox2, oy2 = cx2-cx1, cy2-cy1
        if crop.size == 0: continue
        if size >= cartoon_min:
            gan_in = crop
            if input_temporal > 0 and prev_raw is not None:
                # 입력끼리 섞는다. 출력 블렌딩과 달리 사진끼리라 잔상이 아니라
                # 자연스러운 모션블러로 보이고, GAN 은 안정된 입력 하나만 본다.
                prev_crop = prev_raw[py1:py2, px1:px2]
                if prev_crop.shape == crop[oy1:oy2, ox1:ox2].shape:
                    gan_in = crop.copy()
                    gan_in[oy1:oy2, ox1:ox2] = cv2.addWeighted(
                        prev_crop, input_temporal,
                        crop[oy1:oy2, ox1:ox2], 1.0 - input_temporal, 0)
            gan_in = denoise_crop(gan_in, input_denoise)
            styl = cv2.resize(stylizer.stylize(gan_in), (cx2-cx1, cy2-cy1), interpolation=cv2.INTER_LANCZOS4)
            styl = sharpen_crop(styl, sharpen)
            if color_mode == "lowfreq":
                proc = color_transfer_lowfreq(styl, crop, color_match,
                                              color_sigma, color_luma,
                                              vivid_chroma, vivid_luma)
            else:
                proc = color_transfer(styl, crop, color_match)
            proc = flatten_skin(proc, flatten)
            nc += 1
        else:
            proc = blur_crop(crop, blur_mode); nb += 1
        crop_h, crop_w = crop.shape[:2]
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        if mask_mode == "face-ellipse":
            ecx = int(round((x1 + x2) * 0.5 - cx1))
            ecy = int(round((y1 + y2) * 0.5 - cy1))
            eax = max(1, int(round(bw * 0.5 * mask_scale_x)))
            eay = max(1, int(round(bh * 0.5 * mask_scale_y)))
            cv2.ellipse(mask, (ecx, ecy), (eax, eay), 0, 0, 360, 255, -1)
            fk = max(3, int(round(min(crop_w, crop_h) * mask_feather)) | 1)
            fk = min(151, fk)
        else:
            ecx, ecy = crop_w//2, crop_h//2
            # 타원을 크롭보다 조금 작게 잡아 페더가 크롭 안에서 끝나게 한다.
            # 경계에 닿으면 블러가 잘려 직선 이음매(머리카락을 가로지르는 선)가 보인다.
            eax = max(1, int(round(ecx * 0.90)))
            eay = max(1, int(round(ecy * 0.90)))
            cv2.ellipse(mask, (ecx, ecy), (eax, eay), 0, 0, 360, 255, -1)
            # 페더는 고정 상한이 아니라 크롭 크기에 비례해야 한다.
            # 상한 31px 은 큰 얼굴에서 하드 엣지가 된다.
            fk = max(5, int(round(min(crop_w, crop_h) * mask_feather)) | 1)
            fk = min(151, fk)
        # 페더도 큰 커널이면 비싸다. 저역통과 헬퍼로 동일 효과를 싸게 낸다.
        m = lowpass(mask.astype(np.float32), max(1.0, fk / 3.0)) / 255.0
        original = frame[py1:py2, px1:px2]
        proc_roi = proc[oy1:oy2, ox1:ox2].astype(np.float32)
        mask_roi = m[oy1:oy2, ox1:ox2, None]
        if temporal > 0 and prev_frame is not None:
            # 스타일러는 실제 움직임을 1.3배 증폭한다(선이 프레임마다 튄다).
            # 같은 화면 좌표의 직전 출력과 섞어 그 진동을 눌러준다.
            # 얼굴이 빠르게 움직이면 잔상이 생기므로 0.3~0.5 가 실용 범위다.
            proc_roi = temporal * prev_frame[py1:py2, px1:px2].astype(np.float32) \
                       + (1.0 - temporal) * proc_roi
        frame[py1:py2, px1:px2] = (
            original*(1-mask_roi) + proc_roi*mask_roi
        ).astype(np.uint8)
    return nc, nb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="models/base_v2f2_1280_fp16.onnx")
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--cartoon-min", type=int, default=150, dest="cartoon_min", help="이 픽셀 이상 → 카툰, 미만 → 블러")
    ap.add_argument("--blur-mode", default="pixelate", choices=["pixelate", "gaussian", "box"], dest="blur_mode")
    ap.add_argument("--face-occupancy", type=float, default=0.0, dest="face_occupancy",
                    help="얼굴 면적 / 크롭 면적. 0=--face-expand 사용. 학습 매니페스트와 같은 값을 쓴다")
    ap.add_argument("--color-match", type=float, default=0.0, dest="color_match", help="원본 색감 매칭 0~1")
    ap.add_argument("--face-expand", type=float, default=0.15, dest="face_expand",
                    help="검출 얼굴 박스 바깥으로 모델 입력 crop을 확장할 비율")
    ap.add_argument("--square-crop", action="store_true", dest="square_crop",
                    help="얼굴 중심의 정사각 crop 사용(얼굴 전용 학습 입력과 일치)")
    ap.add_argument("--mask-mode", default="crop-ellipse", choices=["crop-ellipse", "face-ellipse"],
                    dest="mask_mode", help="crop 전체 또는 원래 얼굴 박스 기준 합성 타원")
    ap.add_argument("--mask-scale-x", type=float, default=0.92, dest="mask_scale_x")
    ap.add_argument("--mask-scale-y", type=float, default=1.0, dest="mask_scale_y")
    ap.add_argument("--mask-feather", type=float, default=0.04, dest="mask_feather")
    ap.add_argument("--color-mode", default="global", choices=["global", "lowfreq"],
                    dest="color_mode",
                    help="global=전역 평균/표준편차, lowfreq=저주파 교체(공간적으로 변하는 조명 대응)")
    ap.add_argument("--color-sigma", type=float, default=0.10, dest="color_sigma",
                    help="lowfreq 의 저주파 경계. 크롭 크기 대비 비율. 작을수록 사진 명암이 더 돌아온다")
    ap.add_argument("--color-luma", type=float, default=0.5, dest="color_luma",
                    help="밝기(L) 저주파를 원본에서 가져오는 비율. 0=화풍 명암 유지, 1=원본 조명 그대로")
    ap.add_argument("--vivid-chroma", type=float, default=1.0, dest="vivid_chroma",
                    help="lowfreq 모드에서 채도 국소 진폭 배율. 1.2~1.5 로 올리면 색이 쨍해진다")
    ap.add_argument("--vivid-luma", type=float, default=1.0, dest="vivid_luma",
                    help="lowfreq 모드에서 명암 국소 진폭 배율. 올리면 대비가 강해지나 노이즈도 커진다")
    ap.add_argument("--input-denoise", type=float, default=0.0, dest="input_denoise",
                    help="스타일화 전 입력 잡음 제거(0=끔, 0.5~1.5). 붓질 자국을 줄인다")
    ap.add_argument("--input-temporal", type=float, default=0.0, dest="input_temporal",
                    help="스타일화 전 이전 프레임 입력과 혼합(0=끔, 0.2~0.4). 흔들림의 원인을 줄인다")
    ap.add_argument("--flatten", type=float, default=0.0,
                    help="스타일화 후 피부 평탄화(0=끔, 0.3~0.7). 점·얼룩을 지우고 선은 남긴다")
    ap.add_argument("--temporal", type=float, default=0.0,
                    help="직전 출력 프레임과의 블렌딩 비율(0=끔, 0.3~0.5 권장). 선 튐을 줄이나 잔상이 생긴다")
    ap.add_argument("--box-smooth", type=float, default=0.0, dest="box_smooth",
                    help="검출 박스 EMA 계수(0=끔, 0.5~0.8 권장). 프레임 간 번쩍임을 줄인다")
    ap.add_argument("--sharpen", type=float, default=0.0,
                    help="스타일화 결과 언샤프 마스크 강도. 0=끔, 0.3~0.8 권장")
    ap.add_argument("--trt", action="store_true", help="TensorRT 검출(첫 실행은 엔진 빌드로 느림)")
    ap.add_argument("--encoder", default="nvenc", choices=["nvenc", "x264"], help="영상 인코더")
    ap.add_argument("--half", action="store_true", help="카툰 GAN fp16(torch 백엔드)")
    ap.add_argument("--gan-size", type=int, default=0, dest="gan_size", help="GAN 입력 고정크기(예 384). 0=크롭 원본")
    ap.add_argument("--gan-backend", default="torch", choices=["torch", "onnx"], dest="gan_backend",
                    help="torch=eager(+--compile/--half) | onnx=onnxruntime(+--trt로 TensorRT)")
    ap.add_argument("--compile", action="store_true", help="torch.compile(torch 백엔드, 첫 호출 느림)")
    ap.add_argument("--gan-onnx", default="gan_ckpt/animegan_512.onnx", dest="gan_onnx", help="ONNX GAN 경로(없으면 자동 export)")
    ap.add_argument("--gan-onnx-size", type=int, default=512, dest="gan_onnx_size", help="ONNX GAN 고정 입력크기")
    ap.add_argument("--max-frames", type=int, default=0, dest="max_frames", help="N프레임만 처리(측정용). 0=전체")
    ap.add_argument("--out", default="out/deid_cartoon.mp4")
    args = ap.parse_args()
    if args.face_expand < 0:
        ap.error("--face-expand must be non-negative")
    if args.mask_scale_x <= 0 or args.mask_scale_y <= 0:
        ap.error("mask scales must be positive")
    if args.mask_feather < 0:
        ap.error("--mask-feather must be non-negative")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    det = Detector(args.model, args.size, use_trt=args.trt)
    if args.gan_backend == "onnx":
        styl = AnimeGANONNX(args.gan_onnx, size=args.gan_onnx_size, use_trt=args.trt)  # 512 고정, TRT 가속
    else:
        gs = args.gan_size or (512 if args.compile else 0)   # compile은 고정크기 필요→기본 512
        styl = AnimeGAN(half=args.half, gan_size=gs, compile=args.compile)

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 30.0; total = int(cap.get(7))
    print(f"{W}x{H} @ {fps:.0f}fps, {total} frames | enc={args.encoder} trt={args.trt} "
          f"gan={args.gan_backend} compile={args.compile} half={args.half} gan_size={args.gan_size}")

    # ffmpeg 직결 파이프. -loglevel error/-nostats: 인코더 진행률 노이즈 억제(우리 fps만 출력).
    enc = (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"] if args.encoder == "nvenc"
           else ["-c:v", "libx264", "-crf", "23"])
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-nostats",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}",
           "-i", "-", "-i", args.video, "-map", "0:v", "-map", "1:a?"] + enc + \
          ["-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    smoother = BoxSmoother(args.box_smooth)
    prev_out = None
    prev_raw = None
    i = 0; tot_c = tot_b = 0; t_det = t_comp = t_write = 0.0; t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok: break
        i += 1
        a = time.perf_counter()
        raw = frame.copy() if args.input_temporal > 0 else None
        boxes = smoother(det.detect(frame, W, H))
        b = time.perf_counter()
        nc, nb = composite(
            frame, boxes, styl, args.cartoon_min, args.blur_mode,
            expand=args.face_expand, color_match=args.color_match,
            square_crop=args.square_crop, mask_mode=args.mask_mode, occupancy=args.face_occupancy,
            sharpen=args.sharpen, temporal=args.temporal, prev_frame=prev_out,
            color_mode=args.color_mode, color_sigma=args.color_sigma, color_luma=args.color_luma,
            vivid_chroma=args.vivid_chroma, vivid_luma=args.vivid_luma,
            input_denoise=args.input_denoise, input_temporal=args.input_temporal,
            prev_raw=prev_raw, flatten=args.flatten,
            mask_scale_x=args.mask_scale_x, mask_scale_y=args.mask_scale_y,
            mask_feather=args.mask_feather,
        )
        c = time.perf_counter()
        if args.temporal > 0:
            prev_out = frame.copy()      # 다음 프레임의 시간 블렌딩 기준
        prev_raw = raw               # 입력 시간 평활 기준(스타일화 전 원본)
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
