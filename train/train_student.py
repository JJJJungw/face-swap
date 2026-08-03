#!/usr/bin/env python3
"""
④ 학생 학습 — PAIRED 지도학습(pix2pix식): (실사 input → 2.5D 애니 target) 페어를 직접 학습.

■ 왜 paired로 다시 썼나 (2026-07-28)
  unpaired + gram(style) 방식은 gram이 '텍스처 통계'를 맞추다 보니 **붓터치=유화(oil painting)** 질감이 나온다.
  (게다가 gram 가중치를 40까지 올려 유화가 심화됨.) → 결과가 2.5D가 아니라 유화.
  우리는 정렬된 페어(out/pairs_dataset/input↔target)가 있으므로, gram을 버리고
  **L1 + perceptual 로 target(깔끔한 2.5D)을 직접 재현** → 유화 텍스처 없이 코퍼스 화풍 그대로.
    - 화풍/구조/표정 = target을 L1로 그대로 따라감 (target이 이미 원본 포즈·표정 유지).
    - 디테일·선명 = 다층 VGG perceptual + 경량 adversarial(LSGAN).
    - 신원 제거 = id-loss(옵션, fake를 input 신원에서 멀어지게. 기본 0).

■ Generator = animegan2 구조 유지 → 런타임(deid_cartoon.py) ONNX 슬롯과 그대로 호환.

라이선스: 본 파일=우리 소유. Generator 구조=animegan2-pytorch(MIT) 재구현. VGG/facenet=학습 전용(런타임 미포함).

사용:
  python train/train_student.py --smoke
  python train/train_student.py --data out/pairs_dataset --out train/student_paired \
    --size 256 --batch 8 --init-steps 1500 --steps 40000 --id-loss 0
"""
import os, argparse, sys, random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, get_worker_info
from torch.nn.utils import spectral_norm
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run"))
from pair_utils import discover_pairs

# ============ Generator (animegan2-pytorch 구조 재구현, MIT 귀속 — 런타임 호환) ============
class ConvNormLReLU(nn.Sequential):
    def __init__(self, i, o, k=3, s=1, p=1):
        super().__init__(
            nn.ReflectionPad2d(p),
            nn.Conv2d(i, o, k, s, 0, bias=False),
            nn.GroupNorm(1, o, affine=True),
            nn.LeakyReLU(0.2, inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, i, o, e=2):
        super().__init__()
        self.res = (i == o)
        m = round(i * e)
        L = []
        if e != 1:
            L.append(ConvNormLReLU(i, m, k=1, p=0))
        L += [nn.ReflectionPad2d(1),
              nn.Conv2d(m, m, 3, 1, 0, groups=m, bias=False),
              nn.GroupNorm(1, m, affine=True),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Conv2d(m, o, 1, 1, 0, bias=False),
              nn.GroupNorm(1, o, affine=True)]
        self.body = nn.Sequential(*L)

    def forward(self, x):
        y = self.body(x)
        return y + x if self.res else y


class UpPixelShuffle(nn.Module):
    """2x 업샘플: conv → PixelShuffle (bilinear보다 선명, 학습형 업샘플 → 유화 원인 제거)."""
    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.Conv2d(i, o * 4, 3, 1, 1, bias=False)
        self.ps = nn.PixelShuffle(2)
        self.norm = nn.GroupNorm(1, o, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.ps(self.conv(x))))


class Generator(nn.Module):
    """animegan2 인코더-디코더 + U-Net skip + PixelShuffle 업샘플.
    skip이 인코더의 고해상도 선/경계를 디코더로 넘겨 '부드럽지만 깔끔한' 2.5D 유지(유화 방지).
    입력 크기는 4의 배수(256/512 OK). ONNX/런타임 슬롯 시그니처(x[-1,1]→y[-1,1]) 동일."""
    def __init__(self, ch=32):
        super().__init__()
        c1, c2, c3 = ch, ch * 2, ch * 4                     # 32,64,128
        self.in_conv = ConvNormLReLU(3, c1, k=7, p=3)       # /1, c1   → skip1
        self.down1 = nn.Sequential(ConvNormLReLU(c1, c2, s=2), ConvNormLReLU(c2, c2))   # /2, c2 → skip2
        self.down2 = nn.Sequential(ConvNormLReLU(c2, c3, s=2), ConvNormLReLU(c3, c3))   # /4, c3
        self.mid = nn.Sequential(ConvNormLReLU(c3, c3),
                                 InvertedResidual(c3, c3), InvertedResidual(c3, c3),
                                 InvertedResidual(c3, c3), InvertedResidual(c3, c3),
                                 ConvNormLReLU(c3, c3))      # /4, c3 (병목)
        self.up1 = UpPixelShuffle(c3, c2)                   # /4→/2, c2
        self.dec1 = nn.Sequential(ConvNormLReLU(c2 * 2, c2), ConvNormLReLU(c2, c2))     # +skip2
        self.up2 = UpPixelShuffle(c2, c1)                   # /2→/1, c1
        self.dec2 = nn.Sequential(ConvNormLReLU(c1 * 2, c1), ConvNormLReLU(c1, c1))     # +skip1
        self.out = nn.Sequential(nn.Conv2d(c1, 3, 1, 1, 0, bias=False), nn.Tanh())

    def forward(self, x):
        s1 = self.in_conv(x)                                # /1
        s2 = self.down1(s1)                                 # /2
        h = self.down2(s2)                                  # /4
        h = self.mid(h)
        h = self.up1(h)                                     # /2
        h = self.dec1(torch.cat([h, s2], 1))               # skip2
        h = self.up2(h)                                     # /1
        h = self.dec2(torch.cat([h, s1], 1))               # skip1
        return self.out(h)


# ============ Discriminator (PatchGAN + spectral norm) ============
class Discriminator(nn.Module):
    def __init__(self, ch=48, n=3):
        super().__init__()
        L = [spectral_norm(nn.Conv2d(3, ch, 3, 1, 1)), nn.LeakyReLU(0.2, True)]
        c = ch
        for _ in range(n):
            L += [spectral_norm(nn.Conv2d(c, c * 2, 3, 2, 1)), nn.LeakyReLU(0.2, True),
                  spectral_norm(nn.Conv2d(c * 2, c * 2, 3, 1, 1)),
                  nn.GroupNorm(1, c * 2, affine=True), nn.LeakyReLU(0.2, True)]
            c *= 2
        L += [spectral_norm(nn.Conv2d(c, 1, 3, 1, 1))]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


# ============ VGG19 다층 perceptual (relu1_2,2_2,3_4,4_4) — gram 아님! 특징 L1 ============
class VGGPerceptual(nn.Module):
    LAYERS = (3, 8, 17, 26)

    def __init__(self, allow_random=False):
        super().__init__()
        try:
            v = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        except Exception as e:
            if not allow_random:
                raise RuntimeError(f"VGG19 pretrained weights are required: {e}") from e
            print(f"[vgg] 가중치 다운로드 실패 → 랜덤init(스모크용): {e}")
            v = models.vgg19(weights=None)
        self.v = v.features[:27].eval()
        for p in self.v.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def feats(self, x):
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        out = []
        for i, layer in enumerate(self.v):
            x = layer(x)
            if i in self.LAYERS:
                out.append(x)
        return out

    def forward(self, fake, tgt):          # 다층 특징 L1(=perceptual). target을 '직접' 따라감 → 유화 X
        ff, ft = self.feats(fake), self.feats(tgt)
        return sum(F.l1_loss(a, b) for a, b in zip(ff, ft)) / len(ff)


# ============ 신원 임베더(옵션, 학습 전용) ============
def load_id(device):
    try:
        from facenet_pytorch import InceptionResnetV1
        m = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        for p in m.parameters():
            p.requires_grad_(False)
        print("[id] facenet(vggface2) 로드 — 학습전용")
        return m
    except Exception as e:
        print(f"[id] 임베더 로드 실패 → 신원손실 비활성: {e}")
        return None


def id_embed(m, x):
    x = F.interpolate(x, 160, mode="bilinear", align_corners=False)
    return F.normalize(m(x), dim=1)


# ============ 도메인 갭 증강 (인공 열화) ============
# 학습 코퍼스(SFHQ-T2I)는 Flux 생성 인물 = 깨끗하고 크고 대체로 정면.
# 런타임 입력은 deid_cartoon.py 가 영상에서 잘라낸 얼굴이다:
#   YOLOX 박스 → expand 0.15 크롭 → 종횡비 무시하고 (512,512) INTER_AREA → GAN
# 즉 작은 얼굴일수록 크게 확대돼 들어온다.
#
# ★ 설계 의도 (2026-07-29 변경)
#   기존 cartoon_min=150 은 사전학습 face_paint_512_v2 가 작은 얼굴에서 무너져서
#   막아둔 **우회책**이었다. 자체 학생을 학습하는 지금은 목표가 다르다 —
#   **작은 얼굴도 잘 카툰화하도록 명시적으로 가르친다.**
#   따라서 저해상도를 "가끔 주는 노이즈"가 아니라 **주력 학습 케이스**로 넣는다.
#   (log-uniform 샘플링으로 작은 쪽에 표본을 몰아준다)
#   성공하면 cartoon_min 을 크게 낮출 수 있고, 블러↔카툰 경계 튐 문제도 함께 줄어든다.
#
# 저작권상 실제 영상 크롭을 코퍼스에 섞을 수 없으므로 Real-ESRGAN/BSRGAN 계열의
# 인공 열화로 대체한다. 원칙:
#   - 기하 변환(반전/크롭/종횡비)은 input·target **양쪽에** → 페어 정합 유지
#   - 열화(블러·축소·노이즈·JPEG·색)는 **input에만** → target은 정답이라 깨끗해야 한다.
#     그래야 "작고 더러운 사진 → 크고 깨끗한 카툰" 매핑을 배운다(복원+스타일화 동시).
def _degrade(img, level, rng, size):
    """input 전용 열화. 실제 촬영 체인 순서: 블러 → 축소 → 노이즈 → 압축 → 재확대."""
    h, w = img.shape[:2]

    # ① 블러 (초점/모션) — 광학 단계라 축소보다 먼저
    if rng.random() < (0.35 if level < 3 else 0.5):
        if level >= 3 and rng.random() < 0.4:                      # 모션 블러(방향성)
            k = int(rng.integers(5, 16)) | 1
            ker = np.zeros((k, k), np.float32)
            ang, c = rng.uniform(0, np.pi), k // 2
            for t in np.linspace(-c, c, k * 2):
                x, y = int(round(c + t * np.cos(ang))), int(round(c + t * np.sin(ang)))
                if 0 <= x < k and 0 <= y < k:
                    ker[y, x] = 1
            if ker.sum() > 0:
                img = cv2.filter2D(img, -1, ker / ker.sum())
        else:                                                       # 초점 흐림
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.5 if level < 3 else 2.5))

    # ② 축소 → 재확대 ★ 핵심. 작은 얼굴을 주력으로 학습시킨다.
    #    log-uniform 이라 작은 쪽에 표본이 몰린다(중앙값 ≈ sqrt(lo)).
    #    level 2: 0.30~1.0 → 154~512px   (중앙값 ≈ 280px)
    #    level 3: 0.12~1.0 → 61~512px    (중앙값 ≈ 177px, 절반이 그 아래)
    lo = {1: 0.60, 2: 0.30}.get(level, 0.12)
    if rng.random() < (0.5 if level == 1 else 0.9):
        sc = float(np.exp(rng.uniform(np.log(lo), 0.0)))
        sw, sh = max(12, int(w * sc)), max(12, int(h * sc))
        img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)

        # ③ 노이즈 — 저해상도 단계에서 얹어야 확대 시 실제처럼 번진다
        if level >= 3 and rng.random() < 0.4:
            img = np.clip(img.astype(np.float32)
                          + rng.normal(0, rng.uniform(1, 8), img.shape), 0, 255).astype(np.uint8)

        # ④ JPEG/H.264 압축 아티팩트 — 압축도 저해상도 단계
        if rng.random() < 0.7:
            q = int(rng.integers(30 if level >= 3 else 50, 92))
            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        # ⑤ 런타임과 동일하게 되돌림 (deid_cartoon.py 는 INTER_AREA 사용)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    # ⑥ 노출·대비·감마·색온도 (역광, 실내조명, 화이트밸런스)
    if rng.random() < 0.6:
        f = img.astype(np.float32) / 255.
        f = np.clip(f ** rng.uniform(0.7, 1.4), 0, 1)
        f = np.clip((f - 0.5) * rng.uniform(0.8, 1.25) + 0.5 + rng.uniform(-0.08, 0.08), 0, 1)
        if level >= 2:
            f = np.clip(f * rng.uniform(0.9, 1.1, size=(1, 1, 3)), 0, 1)     # 채널별 = 색온도
        img = (f * 255).astype(np.uint8)

    return img


# ============ 데이터 (PAIRED: input↔target 같은 stem) ============
def parse_aug_mix(value):
    if not value:
        return None
    mix = []
    for item in value.split(","):
        level_text, weight_text = item.split(":", 1)
        level, weight = int(level_text), float(weight_text)
        if level not in (0, 1, 2, 3) or weight < 0:
            raise ValueError(f"invalid augmentation mix item: {item}")
        mix.append((level, weight))
    total = sum(weight for _, weight in mix)
    if total <= 0:
        raise ValueError("augmentation mix weights must sum to a positive value")
    return [(level, weight / total) for level, weight in mix]


class PairImgs(Dataset):
    def __init__(self, root, size, aug=False, aug_level=0, pairs=None, aug_mix=None, seed=0,
                 load_masks=False):
        din, dtg = os.path.join(root, "input"), os.path.join(root, "target")
        self.pairs = list(pairs) if pairs is not None else discover_pairs(din, dtg)
        if not self.pairs:
            raise SystemExit(f"페어 없음: {din} ∩ {dtg}")
        self.din, self.dtg, self.size = din, dtg, size
        # 구 --aug 플래그 호환: 켜져 있고 --aug-level 미지정이면 1
        self.level = aug_level if aug_level > 0 else (1 if aug else 0)
        self.aug_mix = aug_mix
        self.seed = seed
        self.rng = None
        self.load_masks = load_masks
        self.mask_dir = os.path.join(root, "mask")
        if self.load_masks and not os.path.isdir(self.mask_dir):
            raise SystemExit(f"face mask directory missing: {self.mask_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pair = self.pairs[i]
        a = cv2.cvtColor(np.array(Image.open(pair.input_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        b = cv2.cvtColor(np.array(Image.open(pair.target_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        mask = None
        if self.load_masks:
            mask_path = os.path.join(self.mask_dir, f"{pair.stem}.png")
            if not os.path.isfile(mask_path):
                raise SystemExit(f"face mask missing: {mask_path}")
            mask = np.array(Image.open(mask_path).convert("L"))
        if a.shape[:2] != b.shape[:2]:                   # teacher 출력이 더 큰 경우 정렬
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        if mask is not None and mask.shape[:2] != a.shape[:2]:
            mask = cv2.resize(mask, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)

        if self.rng is None:
            worker = get_worker_info()
            worker_seed = worker.seed if worker is not None else self.seed
            self.rng = np.random.default_rng(worker_seed)
        rng = self.rng
        level = self.level
        if self.aug_mix:
            levels, weights = zip(*self.aug_mix)
            level = int(rng.choice(levels, p=weights))
        if level >= 1:
            # ── 기하: input·target 동일 적용 (정합 유지) ────────────────
            if rng.random() < 0.5:
                a, b = a[:, ::-1], b[:, ::-1]
                if mask is not None:
                    mask = mask[:, ::-1]
            if level >= 2 and rng.random() < 0.7:
                # 런타임은 expand 0.15 박스를 종횡비 무시하고 정사각으로 눌러 넣는다.
                # → 타이트 크롭 + 종횡비 왜곡 재현. 반드시 양쪽에 같이.
                H, W = a.shape[:2]
                ch = rng.uniform(0.72, 1.0)
                cw = min(ch * rng.uniform(0.85, 1.18), 1.0)
                th, tw = int(H * ch), int(W * cw)
                y0 = int(rng.integers(0, H - th + 1)); x0 = int(rng.integers(0, W - tw + 1))
                a = a[y0:y0 + th, x0:x0 + tw]; b = b[y0:y0 + th, x0:x0 + tw]
                if mask is not None:
                    mask = mask[y0:y0 + th, x0:x0 + tw]

        a = cv2.resize(np.ascontiguousarray(a), (self.size, self.size), interpolation=cv2.INTER_AREA)
        b = cv2.resize(np.ascontiguousarray(b), (self.size, self.size), interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(np.ascontiguousarray(mask), (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        if level >= 1:
            a = _degrade(a, level, rng, self.size)             # ★ input에만

        ta = torch.from_numpy(cv2.cvtColor(a, cv2.COLOR_BGR2RGB).copy()).permute(2, 0, 1).float() / 127.5 - 1
        tb = torch.from_numpy(cv2.cvtColor(b, cv2.COLOR_BGR2RGB).copy()).permute(2, 0, 1).float() / 127.5 - 1
        if mask is None:
            return ta, tb
        tm = torch.from_numpy(mask.copy()).unsqueeze(0).float() / 255.0
        return ta, tb, tm


def cycle(dl):
    while True:
        for x in dl:
            yield x


# ============ 학습 ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/pairs_dataset", help="input/ 와 target/ 를 가진 폴더")
    ap.add_argument("--out", default="train/student_paired")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4, help="pix2pix식 2e-4")
    ap.add_argument("--init-steps", type=int, default=1500, dest="init_steps", help="L1+perc만(adv 없이) 워밍업")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--adv-ramp", type=int, default=2000, dest="adv_ramp", help="adv 0→목표 램프")
    ap.add_argument("--w-l1", type=float, default=10.0, dest="w_l1", help="target 직접재현(주력)")
    ap.add_argument("--w-perc", type=float, default=1.0, dest="w_perc", help="다층 perceptual(디테일)")
    ap.add_argument("--w-adv", type=float, default=1.0, dest="w_adv", help="경량 adversarial(선명)")
    ap.add_argument("--w-tv", type=float, default=0.0, dest="w_tv")
    ap.add_argument("--id-loss", type=float, default=0.0, dest="id_loss", help="신원억제(0=off). input 신원에서 멀어지게")
    ap.add_argument("--id-margin", type=float, default=0.3, dest="id_margin")
    ap.add_argument("--face-mask-weight", type=float, default=1.0, dest="face_mask_weight",
                    help="mask/ 내부 L1 가중치. 1=마스크 미사용, 얼굴 국소 파인튜닝은 4~8 권장")
    ap.add_argument("--gen-ch", type=int, default=32, dest="gen_ch", help="제너레이터 기본채널(용량↑=32→48→64, 속도↓)")
    ap.add_argument("--d-ch", type=int, default=48, dest="d_ch")
    ap.add_argument("--d-n", type=int, default=3, dest="d_n")
    ap.add_argument("--aug", action="store_true", help="(구) 약한 색 aug. --aug-level 1과 동등")
    ap.add_argument("--aug-level", type=int, default=0, dest="aug_level", choices=[0, 1, 2, 3],
                    help="도메인 갭 인공 열화 강도 (input에만 적용). "
                         "0=없음 / 1=색·압축·약블러 / 2=+저해상도(154~512px)·타이트크롭 / "
                         "3=+모션블러·노이즈·강한 저해상도(61~512px). "
                         "★ 작은 얼굴까지 카툰화하려면 2 이상 권장")
    ap.add_argument("--sample-every", type=int, default=500, dest="sample_every")
    ap.add_argument("--ckpt-every", type=int, default=5000, dest="ckpt_every")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0, dest="split_seed")
    ap.add_argument("--val-ratio", type=float, default=0.05, dest="val_ratio",
                    help="고정 validation 분할 비율(0=분할 안 함)")
    ap.add_argument("--val-n", type=int, default=64, dest="val_n",
                    help="샘플 저장 시 validation L1을 계산할 최대 페어 수")
    ap.add_argument("--aug-mix", default=None, dest="aug_mix",
                    help="샘플별 증강 혼합. 예: 0:0.7,1:0.2,2:0.1")
    ap.add_argument("--overfit-n", type=int, default=0, dest="overfit_n",
                    help="진단용으로 앞 N개 페어만 사용하고 validation/증강을 끔")
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="full checkpoint에서 재시작. 경로 생략 시 <out>/checkpoint_latest.pt")
    ap.add_argument("--init-ckpt", default=None, dest="init_ckpt",
                    help="새 실험 초기화용 checkpoint. G 가중치만 로드하고 step/optimizer/split은 초기화")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not 0 <= args.val_ratio < 1:
        ap.error("--val-ratio must be in [0, 1)")
    if args.overfit_n < 0:
        ap.error("--overfit-n must be non-negative")
    if args.val_n < 0:
        ap.error("--val-n must be non-negative")
    if args.face_mask_weight < 1:
        ap.error("--face-mask-weight must be at least 1")
    try:
        aug_mix = parse_aug_mix(args.aug_mix)
    except (ValueError, TypeError) as exc:
        ap.error(str(exc))
    if aug_mix and (args.aug or args.aug_level):
        ap.error("--aug-mix cannot be combined with --aug/--aug-level")
    if args.overfit_n and (aug_mix or args.aug or args.aug_level):
        ap.error("--overfit-n requires clean inputs; do not pass augmentation options")
    if args.resume and args.init_ckpt:
        ap.error("--resume and --init-ckpt are mutually exclusive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator(args.gen_ch).to(dev)
    D = Discriminator(args.d_ch, args.d_n).to(dev)
    vgg = VGGPerceptual(allow_random=args.smoke).to(dev)
    idm = load_id(dev) if args.id_loss > 0 else None
    optG = Adam(G.parameters(), args.lr, betas=(0.5, 0.999))
    optD = Adam(D.parameters(), args.lr, betas=(0.5, 0.999))
    start_step = 0
    resume_state = None

    if args.init_ckpt:
        try:
            init_state = torch.load(args.init_ckpt, map_location=dev, weights_only=False)
        except TypeError:
            init_state = torch.load(args.init_ckpt, map_location=dev)
        init_weights = init_state["G"] if isinstance(init_state, dict) and "G" in init_state else init_state
        detected_ch = int(init_weights["in_conv.1.weight"].shape[0])
        if detected_ch != args.gen_ch:
            raise SystemExit(
                f"--gen-ch {args.gen_ch} does not match --init-ckpt channel count {detected_ch}"
            )
        G.load_state_dict(init_weights, strict=True)
        print(f"[init] G only from {args.init_ckpt}; optimizer/step/split reset")

    def checkpoint_state(step):
        numpy_state = np.random.get_state()
        return {
            "G": G.state_dict(),
            "D": D.state_dict(),
            "optG": optG.state_dict(),
            "optD": optD.state_dict(),
            "step": step,
            "args": vars(args),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng": random.getstate(),
            "split": {
                "train": [pair.stem for pair in train_pairs],
                "val": [pair.stem for pair in val_pairs],
            },
            "numpy_rng": (
                numpy_state[0], numpy_state[1].tolist(), numpy_state[2],
                numpy_state[3], numpy_state[4],
            ),
        }

    def save_checkpoint(path, step):
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = path + ".tmp"
        torch.save(checkpoint_state(step), temporary)
        os.replace(temporary, path)

    if args.resume:
        resume_path = (
            os.path.join(args.out, "checkpoint_latest.pt")
            if args.resume == "auto" else args.resume
        )
        try:
            state = torch.load(resume_path, map_location=dev, weights_only=False)
        except TypeError:
            state = torch.load(resume_path, map_location=dev)
        saved_args = state.get("args", {})
        if saved_args.get("gen_ch", args.gen_ch) != args.gen_ch:
            raise SystemExit("--gen-ch does not match resume checkpoint")
        G.load_state_dict(state["G"], strict=True)
        D.load_state_dict(state["D"], strict=True)
        optG.load_state_dict(state["optG"])
        optD.load_state_dict(state["optD"])
        start_step = int(state["step"])
        resume_state = state
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if "python_rng" in state:
            random.setstate(state["python_rng"])
        if "numpy_rng" in state:
            numpy_state = state["numpy_rng"]
            np.random.set_state((
                numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                numpy_state[2], numpy_state[3], numpy_state[4],
            ))
        print(f"[resume] {resume_path} step={start_step}")

    def g_losses(inp, tgt, w_adv_eff, face_mask=None):
        fake = G(inp)
        if face_mask is not None:
            weights = 1.0 + (args.face_mask_weight - 1.0) * face_mask
            l1 = ((fake - tgt).abs() * weights).sum() / (weights.sum() * fake.shape[1])
            # 바깥을 target으로 치환하면 perceptual gradient가 얼굴 주변에 집중된다.
            perceptual_fake = fake * face_mask + tgt.detach() * (1.0 - face_mask)
            perc = vgg(perceptual_fake, tgt)
        else:
            l1 = F.l1_loss(fake, tgt)                    # ★ target 직접 재현 → 유화 없이 2.5D 그대로
            perc = vgg(fake, tgt)                        # 다층 perceptual(디테일·선명)
        g = args.w_l1 * l1 + args.w_perc * perc
        adv = torch.tensor(0.0, device=dev)
        if w_adv_eff > 0:
            df = D(fake)
            adv = F.mse_loss(df, torch.ones_like(df))
            g = g + w_adv_eff * adv
        if args.w_tv > 0:
            tv = (fake[:, :, 1:] - fake[:, :, :-1]).abs().mean() + (fake[:, :, :, 1:] - fake[:, :, :, :-1]).abs().mean()
            g = g + args.w_tv * tv
        idl = torch.tensor(0.0, device=dev)
        if idm is not None:                              # 신원: fake를 input 신원에서 멀어지게
            cos = (id_embed(idm, inp) * id_embed(idm, fake)).sum(1)
            idl = F.relu(cos - args.id_margin).mean()
            g = g + args.id_loss * idl
        return fake, g, dict(l1=l1.item(), perc=perc.item(), adv=float(adv.detach()), idl=float(idl.detach()))

    def d_loss(inp, tgt):
        fake = G(inp).detach()
        dr, df = D(tgt), D(fake)                          # 진짜 애니 target →1, 생성물 →0
        return 0.5 * (F.mse_loss(dr, torch.ones_like(dr)) + F.mse_loss(df, torch.zeros_like(df)))

    if args.smoke:
        print(f"[smoke] dev={dev} size={args.size}")
        inp = torch.rand(2, 3, args.size, args.size, device=dev) * 2 - 1
        tgt = torch.rand(2, 3, args.size, args.size, device=dev) * 2 - 1
        dl = d_loss(inp, tgt); optD.zero_grad(); dl.backward(); optD.step()
        fake, gl, parts = g_losses(inp, tgt, args.w_adv); optG.zero_grad(); gl.backward(); optG.step()
        print(f"[smoke] out={tuple(fake.shape)} D_patch={tuple(D(inp).shape)} d={dl.item():.3f} g={gl.item():.3f} {parts}")
        print(f"[smoke] G params={sum(p.numel() for p in G.parameters())/1e6:.2f}M → 배선 OK")
        return

    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    all_pairs = discover_pairs(os.path.join(args.data, "input"), os.path.join(args.data, "target"))
    pair_by_stem = {pair.stem: pair for pair in all_pairs}
    saved_split = resume_state.get("split") if resume_state else None
    if saved_split:
        missing = [
            stem for stem in saved_split["train"] + saved_split["val"]
            if stem not in pair_by_stem
        ]
        if missing:
            raise SystemExit(
                f"resume split에서 {len(missing)}개 페어를 찾지 못함: {missing[:5]}"
            )
        train_pairs = [pair_by_stem[stem] for stem in saved_split["train"]]
        val_pairs = [pair_by_stem[stem] for stem in saved_split["val"]]
        print("[split] checkpoint에 저장된 train/validation 분할 복구")
    else:
        split_rng = random.Random(args.split_seed)
        split_rng.shuffle(all_pairs)
        if args.overfit_n:
            train_pairs = all_pairs[:args.overfit_n]
            val_pairs = []
            print(f"[overfit] clean diagnostic subset={len(train_pairs)}")
        else:
            val_count = (
                min(len(all_pairs) - 1, max(1, round(len(all_pairs) * args.val_ratio)))
                if args.val_ratio and len(all_pairs) > 1 else 0
            )
            val_pairs = all_pairs[:val_count]
            train_pairs = all_pairs[val_count:]
    if len(train_pairs) < args.batch:
        raise SystemExit(f"학습 페어 {len(train_pairs)}개가 batch {args.batch}보다 적음")
    use_face_masks = args.face_mask_weight > 1
    ds = PairImgs(
        args.data, args.size, args.aug, args.aug_level, train_pairs, aug_mix, args.seed,
        load_masks=use_face_masks,
    )
    val_ds = (
        PairImgs(args.data, args.size, pairs=val_pairs, seed=args.seed, load_masks=use_face_masks)
        if val_pairs else None
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = cycle(DataLoader(
        ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
        pin_memory=(dev == "cuda"), persistent_workers=(args.workers > 0), generator=generator,
    ))
    print(f"[data] train={len(ds)} val={len(val_ds) if val_ds else 0} "
          f"aug_mix={aug_mix} face_mask_weight={args.face_mask_weight:g}")

    for name, pairs in (("train_stems.txt", train_pairs), ("val_stems.txt", val_pairs)):
        path = os.path.join(args.out, name)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write("".join(f"{pair.stem}\n" for pair in pairs))
        os.replace(temporary, path)

    # 고정 평가셋 4쌍(진행 추적)
    eval_ds = val_ds or ds
    ne = min(4, len(eval_ds))
    ev = [eval_ds[i] for i in range(ne)]
    ev_in = torch.stack([item[0] for item in ev]).to(dev)
    ev_tg = torch.stack([item[1] for item in ev]).to(dev)

    def save_eval(step):
        G.eval()
        with torch.no_grad():
            fk = G(ev_in).clamp(-1, 1)
            grid = torch.cat([ev_in, fk, ev_tg], 0) * 0.5 + 0.5   # 입력 / 학생출력 / 정답target
            save_image(grid, os.path.join(args.out, "samples", f"s{step:06d}.png"), nrow=ne)
            if val_ds is not None and args.val_n:
                losses = []
                face_losses = []
                for begin in range(0, min(args.val_n, len(val_ds)), args.batch):
                    batch = [val_ds[i] for i in range(begin, min(begin + args.batch, len(val_ds), args.val_n))]
                    va = torch.stack([item[0] for item in batch]).to(dev)
                    vt = torch.stack([item[1] for item in batch]).to(dev)
                    vf = G(va).clamp(-1, 1)
                    losses.extend(F.l1_loss(vf, vt, reduction="none").mean((1, 2, 3)).cpu().tolist())
                    if len(batch[0]) == 3:
                        vm = torch.stack([item[2] for item in batch]).to(dev)
                        face_error = ((vf - vt).abs() * vm).sum((1, 2, 3))
                        face_norm = vm.sum((1, 2, 3)).clamp_min(1.0) * vf.shape[1]
                        face_losses.extend((face_error / face_norm).cpu().tolist())
                suffix = f" face_l1={np.mean(face_losses):.4f}" if face_losses else ""
                print(f"[val:{step}] n={len(losses)} l1={np.mean(losses):.4f}{suffix}")
        G.train()

    for step in range(start_step + 1, args.steps + 1):
        train_batch = next(loader)
        inp, tgt = train_batch[:2]
        face_mask = train_batch[2] if len(train_batch) == 3 else None
        inp = inp.to(dev, non_blocking=True)
        tgt = tgt.to(dev, non_blocking=True)
        if face_mask is not None:
            face_mask = face_mask.to(dev, non_blocking=True)
        w_adv_eff = 0.0 if step <= args.init_steps else args.w_adv * min(1.0, (step - args.init_steps) / max(1, args.adv_ramp))
        if w_adv_eff > 0:
            dl = d_loss(inp, tgt); optD.zero_grad(); dl.backward(); optD.step()
        else:
            dl = torch.tensor(0.0)
        fake, gl, parts = g_losses(inp, tgt, w_adv_eff, face_mask); optG.zero_grad(); gl.backward(); optG.step()
        if step % 100 == 0:
            print(f"[{step}/{args.steps}] wadv={w_adv_eff:.2f} D={float(dl):.3f} G={gl.item():.3f} "
                  f"l1={parts['l1']:.3f} perc={parts['perc']:.3f} adv={parts['adv']:.2f} id={parts['idl']:.3f}")
        if step % args.sample_every == 0:
            save_eval(step)
        if step % args.ckpt_every == 0:
            save_checkpoint(os.path.join(args.out, f"student_{step:06d}.pt"), step)
            save_checkpoint(os.path.join(args.out, "checkpoint_latest.pt"), step)
    save_checkpoint(os.path.join(args.out, "student_final.pt"), args.steps)
    save_checkpoint(os.path.join(args.out, "checkpoint_latest.pt"), args.steps)
    print(f"완료 → {args.out}/student_final.pt")


if __name__ == "__main__":
    main()
