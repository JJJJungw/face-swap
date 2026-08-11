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
import os, argparse, sys, random, json, time, math
from contextlib import nullcontext
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
from crop_utils import crop_with_edge_padding

# ============ Generator (animegan2-pytorch 구조 재구현, MIT 귀속 — 런타임 호환) ============
class BlurPool2d(nn.Module):
    """저역통과 후 stride 2. stride-2 conv 의 앨리어싱을 없앤다.

    ■ 왜 필요한가 (2026-08-04 측정)
      입력을 1px 옮기면 출력이 1.30배로 변했다(밝기 +2 → 1.01배, JPEG → 0.93배).
      즉 이 네트워크는 **공간 이동에만** 과민하다. 원인은 stride 2 가 두 픽셀 중
      하나를 그냥 버리는 것이라, 얼굴이 1px 움직이면 살아남는 픽셀이 통째로 바뀌기
      때문이다. 가는 선·머리카락이 있다 없다 하고, 그 위에서 계산된 특징이 흔들려
      최종 출력의 선이 프레임마다 튄다(영상 깜빡임).

      줄이기 전에 이항 커널로 흐리면 버려질 고주파가 미리 제거되어 어느 픽셀이
      살아남든 결과가 비슷해진다. 근거: Zhang, ICML 2019 — 증강 없이 학습한 모델의
      shift consistency 88.1% → 98.1% (우리와 같은 --aug-level 0 조건).
      커널은 고정값이라 학습 파라미터가 늘지 않는다.
    """

    def __init__(self, channels, filt_size=3, stride=2):
        super().__init__()
        if filt_size == 3:
            base = torch.tensor([1.0, 2.0, 1.0])
        elif filt_size == 5:
            base = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        else:
            raise ValueError(f"filt_size must be 3 or 5: {filt_size}")
        kernel = torch.outer(base, base)
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel[None, None].repeat(channels, 1, 1, 1))
        self.channels, self.stride = channels, stride
        self.pad = nn.ReflectionPad2d(filt_size // 2)

    def forward(self, x):
        return F.conv2d(self.pad(x), self.kernel, stride=self.stride, groups=self.channels)


class ConvNormLReLU(nn.Sequential):
    def __init__(self, i, o, k=3, s=1, p=1, antialias=0):
        # antialias>0 이고 stride 2 이면 conv 는 stride 1 로 두고 BlurPool 로 줄인다.
        if s == 2 and antialias:
            super().__init__(
                nn.ReflectionPad2d(p),
                nn.Conv2d(i, o, k, 1, 0, bias=False),
                nn.GroupNorm(1, o, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                BlurPool2d(o, filt_size=antialias, stride=2))
        else:
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
    def __init__(self, ch=32, antialias=0):
        super().__init__()
        c1, c2, c3 = ch, ch * 2, ch * 4                     # 32,64,128
        self.in_conv = ConvNormLReLU(3, c1, k=7, p=3)       # /1, c1   → skip1
        self.down1 = nn.Sequential(ConvNormLReLU(c1, c2, s=2, antialias=antialias), ConvNormLReLU(c2, c2))   # /2, c2 → skip2
        self.down2 = nn.Sequential(ConvNormLReLU(c2, c3, s=2, antialias=antialias), ConvNormLReLU(c3, c3))   # /4, c3
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


class GatedSkip(nn.Module):
    """Add a skip connection with a learnable, deliberately weak initial gain."""

    def __init__(self, init_gain=0.1):
        super().__init__()
        init_gain = min(max(float(init_gain), 1e-4), 1.0 - 1e-4)
        self.logit = nn.Parameter(torch.tensor(math.log(init_gain / (1.0 - init_gain))))

    def forward(self, decoded, skip):
        return decoded + torch.sigmoid(self.logit) * skip


class GeneratorDeep8(nn.Module):
    """Redraw-oriented generator with a /8 bottleneck and weak low-resolution skips."""

    def __init__(self, ch=32, skip2_init=0.1, skip4_init=0.1, antialias=0):
        super().__init__()
        c1, c2, c3, c4 = ch, ch * 2, ch * 4, ch * 6
        self.in_conv = ConvNormLReLU(3, c1, k=7, p=3)
        self.down1 = nn.Sequential(ConvNormLReLU(c1, c2, s=2, antialias=antialias), ConvNormLReLU(c2, c2))
        self.down2 = nn.Sequential(ConvNormLReLU(c2, c3, s=2, antialias=antialias), ConvNormLReLU(c3, c3))
        self.down3 = nn.Sequential(ConvNormLReLU(c3, c4, s=2, antialias=antialias), ConvNormLReLU(c4, c4))
        self.mid = nn.Sequential(
            ConvNormLReLU(c4, c4),
            *[InvertedResidual(c4, c4) for _ in range(6)],
            ConvNormLReLU(c4, c4),
        )
        self.up1 = UpPixelShuffle(c4, c3)
        self.skip4 = GatedSkip(skip4_init)
        self.dec1 = nn.Sequential(ConvNormLReLU(c3, c3), ConvNormLReLU(c3, c3))
        self.up2 = UpPixelShuffle(c3, c2)
        self.skip2 = GatedSkip(skip2_init)
        self.dec2 = nn.Sequential(ConvNormLReLU(c2, c2), ConvNormLReLU(c2, c2))
        # No full-resolution skip: the output must redraw facial features instead of copying pixels.
        self.up3 = UpPixelShuffle(c2, c1)
        self.dec3 = nn.Sequential(ConvNormLReLU(c1, c1), ConvNormLReLU(c1, c1))
        self.out = nn.Sequential(nn.Conv2d(c1, 3, 1, 1, 0, bias=False), nn.Tanh())

    def forward(self, x):
        s1 = self.in_conv(x)
        s2 = self.down1(s1)
        s4 = self.down2(s2)
        h = self.mid(self.down3(s4))
        h = self.dec1(self.skip4(self.up1(h), s4))
        h = self.dec2(self.skip2(self.up2(h), s2))
        h = self.dec3(self.up3(h))
        return self.out(h)


def random_affine_theta(n, device, shift_px, scale, rot_deg, size):
    """합성 모션용 미세 아핀 파라미터. 프레임 간 얼굴 흔들림을 흉내낸다."""
    ang = (torch.rand(n, device=device) * 2 - 1) * (rot_deg * math.pi / 180.0)
    sc = 1.0 + (torch.rand(n, device=device) * 2 - 1) * scale
    tx = (torch.rand(n, device=device) * 2 - 1) * (2.0 * shift_px / max(size, 1))
    ty = (torch.rand(n, device=device) * 2 - 1) * (2.0 * shift_px / max(size, 1))
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    return theta


def warp_affine(x, theta):
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                         align_corners=False)


def build_generator(ch=32, arch="legacy", skip2_init=0.1, skip4_init=0.1, antialias=0):
    if arch == "legacy":
        return Generator(ch, antialias=antialias)
    if arch == "deep8":
        return GeneratorDeep8(ch, skip2_init=skip2_init, skip4_init=skip4_init,
                              antialias=antialias)
    raise ValueError(f"unknown generator architecture: {arch}")


def checkpoint_generator_arch(checkpoint, weights=None):
    if isinstance(checkpoint, dict):
        saved_args = checkpoint.get("args") or {}
        if saved_args.get("gen_arch"):
            return saved_args["gen_arch"]
    if weights is None:
        weights = checkpoint.get("G") if isinstance(checkpoint, dict) else checkpoint
    if isinstance(weights, dict) and any(key.startswith("down3.") for key in weights):
        return "deep8"
    return "legacy"


def checkpoint_generator_kwargs(checkpoint, weights=None):
    """체크포인트만 보고 생성기를 그대로 복원하기 위한 인자를 뽑는다.

    BlurPool 을 켜면 state_dict 에 고정 커널 버퍼(.kernel)가 생기므로,
    구조를 모르고 만들면 strict 로딩이 실패한다. 저장된 args 를 우선 쓰고,
    없으면 가중치에서 역추론한다(구 체크포인트 호환).
    """
    if weights is None:
        weights = checkpoint.get("G") if isinstance(checkpoint, dict) else checkpoint
    saved = (checkpoint.get("args") or {}) if isinstance(checkpoint, dict) else {}
    antialias = saved.get("antialias")
    if antialias is None:
        kernels = [v for k, v in weights.items() if k.endswith(".kernel") and v.dim() == 4]
        antialias = int(kernels[0].shape[-1]) if kernels else 0
    return dict(
        ch=int(weights["in_conv.1.weight"].shape[0]),
        arch=checkpoint_generator_arch(checkpoint, weights),
        antialias=int(antialias),
        skip2_init=float(saved.get("skip2_init", 0.1)),
        skip4_init=float(saved.get("skip4_init", 0.1)),
    )


# ============ Discriminator (PatchGAN + spectral norm) ============
class Discriminator(nn.Module):
    def __init__(self, ch=48, n=3, in_channels=3):
        super().__init__()
        L = [spectral_norm(nn.Conv2d(in_channels, ch, 3, 1, 1)), nn.LeakyReLU(0.2, True)]
        c = ch
        for _ in range(n):
            L += [spectral_norm(nn.Conv2d(c, c * 2, 3, 2, 1)), nn.LeakyReLU(0.2, True),
                  spectral_norm(nn.Conv2d(c * 2, c * 2, 3, 1, 1)),
                  nn.GroupNorm(1, c * 2, affine=True), nn.LeakyReLU(0.2, True)]
            c *= 2
        L += [spectral_norm(nn.Conv2d(c, 1, 3, 1, 1))]
        self.net = nn.Sequential(*L)

    def forward(self, x, return_features=False):
        features = []
        for layer in self.net:
            x = layer(x)
            if return_features and isinstance(layer, nn.LeakyReLU):
                features.append(x)
        if return_features:
            return x, features
        return x


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
_GRID = {}


def _dir_grid(h, w, ang):
    """각도 ang 방향의 -0.7~0.7 그라디언트. 좌표 그리드는 크기별로 캐시한다."""
    key = (h, w)
    if key not in _GRID:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        _GRID[key] = (xx / max(w, 1) - 0.5, yy / max(h, 1) - 0.5)
    gx, gy = _GRID[key]
    return math.cos(ang) * gx + math.sin(ang) * gy


def _radial_grid(h, w, cx, cy):
    """중심 (cx, cy)(정규화 -0.5~0.5 좌표)로부터의 거리 제곱. 그리드는 _GRID 캐시를 공유한다."""
    key = (h, w)
    if key not in _GRID:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        _GRID[key] = (xx / max(w, 1) - 0.5, yy / max(h, 1) - 0.5)
    gx, gy = _GRID[key]
    return (gx - cx) ** 2 + (gy - cy) ** 2


# 광원 색온도 (BGR 게인). 실제 촬영 광원은 무채색이 아니다.
_LIGHT_TINTS = (
    np.array([0.88, 0.97, 1.12], np.float32),   # 텅스텐 · 가로등 · 석양 (주황)
    np.array([1.14, 1.01, 0.88], np.float32),   # 블루아워 · 달빛 · 그늘 (청)
    np.array([0.94, 1.10, 0.94], np.float32),   # 형광 · 사무실 (녹)
    np.array([0.96, 0.92, 1.14], np.float32),   # 네온 · 무대 (자홍)
)


def _stage_light(img, rng, strength_max=0.85):
    """[input 전용] 촬영 조명을 흉내 낸다 — 광원 형태 · 차폐 패턴 · 색온도 · 저조도 · 리림.

    ■ 왜 필요한가 (2026-08-10)
      드라마 영상(swap13)에서 얼굴이 회백색으로 뜨고 이목구비가 무너졌다.
      원인은 **코퍼스에 방향성 조명이 아예 없다**는 것이다.
      SFHQ 는 전부 균일한 스튜디오 조명이고, 기존 증강의 노출 범위는
      감마 0.7~1.4 · 밝기 ±0.08 로 너무 좁다.

      그리고 감마를 낮추는 것으로는 부족하다. 그건 얼굴을 **균일하게** 어둡게 만들 뿐이다.
      실제 촬영 조명의 특징은 어두움이 아니라 **방향성**이다 —
      얼굴 절반에 그림자가 지고, 역광이 윤곽을 긋고, 창문 그림자가 하드한 경계를 만든다.

      포즈와 달리 조명은 **합성으로 흉내 낼 수 있다.** 그래서 teacher 굽기 예산(10.5시간)은
      포즈에 쓰고 조명은 여기서 만든다.

    ■ 1차(08-10)가 너무 단순했다 (08-11 확장)
      1차는 **무채색 평행광 1개 + sigmoid 경계 1개**가 전부였다. swap12/13 에서 개선은
      확인됐으나 실제 촬영은 이보다 훨씬 다양하다. 네 가지를 더한다.

      광원 형태   평행광만으로는 얼굴 위 **국소 하이라이트**가 안 생긴다.
                  스탠드·촛불·가로등은 점광원이고 역제곱으로 감쇠한다.
      필라이트    1차는 그림자 쪽이 그냥 죽었다. 실제로는 반사판·환경광이 바닥을 만든다.
                  floor 를 두고, 낮으면 하드 / 높으면 소프트가 되게 한다.
      차폐 패턴   블라인드 줄무늬 · 나뭇잎 얼룩. 경계가 하나뿐인 그림자는 실내 촬영의 일부일 뿐이다.
      자동 노출   위가 전부 곱셈이라 겹치면 화면이 새까매진다. 실제 카메라는 얼굴에 노출을 맞춘다.
                  시트로 확인했을 때 1차 구현은 표본의 절반이 판독 불가였다. ⑦에서 바닥을 든다.
      색온도      **1차의 가장 큰 누락.** 조명이 전부 무채색이었다. 실내 촬영의 기본형은
                  텅스텐 키(주황) + 창문 그림자(청) 처럼 **밝은 쪽과 그림자 쪽의 색이 다르다.**
                  input 에만 걸리므로 학생은 "주황빛 얼굴 → 정상 색 카툰"을 배운다.
    """
    f = img.astype(np.float32) / 255.0
    h, w = f.shape[:2]

    # ① 광원 형태 — 평행광 또는 점광원.
    #    n 은 [-1,1] 로 정규화한 "빛이 닿는 정도". +1=정면으로 받음, -1=완전 그림자.
    if rng.random() < 0.30:
        r2 = _radial_grid(h, w, rng.uniform(-0.55, 0.55), rng.uniform(-0.55, 0.55))
        n = 2.0 / (1.0 + r2 / rng.uniform(0.02, 0.25)) - 1.0
    else:
        n = _dir_grid(h, w, rng.uniform(0, 2 * math.pi)) / 0.7

    # ② 키라이트 + 필라이트. floor 가 필라이트다 — 낮으면 하드, 높으면 소프트.
    ramp = 1.0 + rng.uniform(0.25, strength_max) * n
    ramp = np.maximum(ramp, rng.uniform(0.20, 0.60))

    # ③ 차폐 패턴 — 창틀 경계 하나 / 블라인드 줄무늬 / 나뭇잎 얼룩
    r = rng.random()
    if r < 0.25:
        d2 = _dir_grid(h, w, rng.uniform(0, 2 * math.pi))
        m = 1.0 / (1.0 + np.exp(-(d2 - rng.uniform(-0.25, 0.25)) / rng.uniform(0.02, 0.18)))
        ramp = ramp * (1.0 - rng.uniform(0.30, 0.75) * (1.0 - m))
    elif r < 0.40:
        d2 = _dir_grid(h, w, rng.uniform(0, 2 * math.pi))
        s = np.sin(d2 * (2 * math.pi / rng.uniform(0.10, 0.35)) + rng.uniform(0, 2 * math.pi))
        m = 1.0 / (1.0 + np.exp(-s / rng.uniform(0.12, 0.60)))
        ramp = ramp * (1.0 - rng.uniform(0.20, 0.55) * (1.0 - m))
    elif r < 0.52:
        k = int(rng.integers(4, 11))
        g = cv2.resize(rng.normal(0.0, 1.0, (k, k)).astype(np.float32), (w, h),
                       interpolation=cv2.INTER_CUBIC)
        m = 1.0 / (1.0 + np.exp(-(g - rng.uniform(-0.3, 0.6)) / rng.uniform(0.08, 0.50)))
        ramp = ramp * (1.0 - rng.uniform(0.25, 0.60) * (1.0 - m))

    f *= ramp[:, :, None]

    # ④ 색온도 — 밝은 쪽과 그림자 쪽에 서로 반대 방향의 색이 실린다
    if rng.random() < 0.70:
        base = _LIGHT_TINTS[int(rng.integers(0, len(_LIGHT_TINTS)))]
        amt = rng.uniform(0.25, 1.0)
        lit = 1.0 + (base - 1.0) * amt
        sha = 1.0 + (1.0 / base - 1.0) * amt * rng.uniform(0.0, 0.8)
        t = np.clip((n + 1.0) * 0.5, 0.0, 1.0)[:, :, None]
        f *= sha + (lit - sha) * t

    # ⑤ 저조도 + 블랙 리프트(필름·야간 장면의 들린 검정)
    if rng.random() < 0.5:
        f *= rng.uniform(0.30, 0.80)
        f += rng.uniform(0.0, 0.06)

    # ⑥ 리림라이트 — 가장자리만 밝게, 색조 포함
    if rng.random() < 0.30:
        rim = np.clip((np.abs(n) - 0.65) / 0.35, 0, 1)[:, :, None]
        tint = rng.uniform(0.9, 1.5, size=(1, 1, 3)).astype(np.float32)
        f = f + rim * rng.uniform(0.15, 0.5) * tint

    # ⑦ 자동 노출 — 실제 카메라는 얼굴에 노출을 맞춘다.
    #    ①~⑥ 이 곱셈이라 그냥 두면 셋이 겹쳤을 때 화면이 새까매진다.
    #    검정 입력에서 얼굴을 그리게 하면 과제가 환각으로 변질된다 —
    #    --aug-level 3 단독 학습이 실패한 것과 같은 기전이다.
    #    어두운 분위기는 남기되 바닥은 들어올린다.
    mu = float(f.mean())
    lo = rng.uniform(0.10, 0.22)
    if mu < lo:
        f *= lo / max(mu, 1e-3)

    return np.clip(f * 255.0, 0, 255).astype(np.uint8)


def _degrade(img, level, rng, size, beauty_p=None, light_p=None):
    """input 전용 열화. 실제 촬영·방송 체인 순서를 따른다.

        플래시/하이라이트 → 뷰티 필터 → 블러 → 축소 → 노이즈 → 압축 → 재확대
        → 노출·색온도·무대조명 → 서브픽셀 지터

    ■ 각 항목이 어떤 실패에서 나왔는가 (2026-08-06)
      뷰티 필터   swap10(중국 예능)에서 얼굴이 밀랍 인형처럼 나왔다. 소스에 이미
                  피부 스무딩이 걸려 있어 **학생이 선으로 바꿀 명암 변화가 없었다.**
                  가우시안 블러로는 흉내 못 낸다 — 그건 선까지 뭉갠다.
                  bilateral 은 대비 큰 화소를 보존하므로 **질감만 지우고 선은 남긴다.**
      플래시      시상식·레드카펫. 국소 과노출로 얼굴 일부가 흰색으로 클리핑된다.
      무대조명    강한 단색 조명. 기존 ±10% 채널 스케일로는 재현이 안 된다.
      모션블러    액션·빠른 움직임. 기존에는 level 3 에서만, 확률 0.4 로 약했다.
      서브픽셀    검출 박스가 프레임마다 흔들려 리샘플링 위상이 달라진다.
                  비등변 잔차가 0.473 로 남아 있어 이 축이 살아 있다.

    ■ beauty_p (2026-08-06 2차)
      1차 augmix 에서 뷰티필터 실적용률이 **7.5%** 였다
      (mix 0:0.7/1:0.2/2:0.1 × p 0.20/0.35). swap10 턱선이 '생기다 만' 수준에 그쳤다.
      --aug-mix 를 올려 용량을 키우면 축소·모션블러까지 같이 세져 단일 변수가 깨지고,
      level 2 축소(154px)가 늘어 과제가 초해상도로 변질된다(아래 참고).
      그래서 **이 항목의 확률만** 따로 연다. beauty_p=0.8 → 실적용률 24%.

    ■ 강도를 균일하게 올리지 말 것
      과거 --aug-level 3 단독 학습이 실패했다. 입력을 62px 까지 뭉개서 512 로 늘리므로
      과제가 *초해상도 + 환각 + 스타일화 동시 수행* 이 되고, L1 최적해가 흐릿한 평균이 된다.
      **--aug-mix 로 대부분을 깨끗하게 두고 일부만 열화시킨다.**
    """
    h, w = img.shape[:2]

    # ⓪-a 촬영 조명 — 광학 단계의 맨 앞. 방향성 키라이트·하드 그림자·저조도·리림
    p_light = 0.0 if light_p is None else float(light_p)
    if level >= 1 and p_light > 0 and rng.random() < p_light:
        img = _stage_light(img, rng)

    # ⓪-a2 필름 그레인 — 휘도 위주(채널 공통)라 컬러 노이즈와 다르다.
    #      드라마·필름 룩 전반에 깔려 있고, 어두운 장면일수록 두드러진다.
    #      기존 노이즈는 level>=3 에서만, 채널 독립이라 이 상황을 재현하지 못했다.
    if level >= 1 and rng.random() < 0.35:
        g = rng.normal(0.0, rng.uniform(2.0, 9.0), (h, w, 1)).astype(np.float32)
        img = np.clip(img.astype(np.float32) + g, 0, 255).astype(np.uint8)

    # ⓪ 플래시 / 하이라이트 클리핑 — 광학 단계라 가장 먼저
    if level >= 2 and rng.random() < 0.15:
        f = img.astype(np.float32)
        if rng.random() < 0.5:                                    # 전역 과노출
            f *= rng.uniform(1.15, 1.6)
        else:                                                     # 국소 스팟
            cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            r = rng.uniform(0.25, 0.6) * max(h, w)
            spot = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))
            f *= (1.0 + rng.uniform(0.3, 1.0) * spot)[:, :, None]
        img = np.clip(f, 0, 255).astype(np.uint8)

    # ⓪-b 뷰티 필터 (피부 스무딩) — 방송 후처리 단계. 블러보다 먼저 온다.
    p_beauty = (0.20 if level == 1 else 0.35) if beauty_p is None else float(beauty_p)
    if level >= 1 and rng.random() < p_beauty:
        d = int(rng.integers(5, 13)) | 1
        img = cv2.bilateralFilter(img, d, rng.uniform(40, 110), rng.uniform(40, 110))

    # ① 블러 (초점/모션) — 광학 단계라 축소보다 먼저
    if rng.random() < (0.35 if level < 3 else 0.5):
        if level >= 2 and rng.random() < (0.45 if level == 2 else 0.6):   # 모션 블러(방향성)
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

    # ⑦ 무대 단색 조명 — 기존 ±10% 로는 재현 안 되는 강한 색 캐스트
    if level >= 2 and rng.random() < 0.20:
        tint = rng.uniform(0.55, 1.45, size=(1, 1, 3)).astype(np.float32)
        strength = rng.uniform(0.3, 0.9)
        blend = 1.0 + (tint - 1.0) * strength
        img = np.clip(img.astype(np.float32) * blend, 0, 255).astype(np.uint8)

    # ⑧ 서브픽셀 지터 — 검출 박스 흔들림으로 리샘플링 위상이 프레임마다 달라지는 것
    #    input 에만 걸어 "같은 얼굴이 조금 다르게 샘플링돼도 같은 그림" 을 학습시킨다.
    if level >= 1 and rng.random() < 0.5:
        dx, dy = rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)

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


def load_localize_manifest(path):
    records = {}
    if not path:
        return records
    with open(path, encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid localize manifest at line {line_number}: {exc}") from exc
            stem = record.get("stem")
            if not stem or stem in records:
                raise SystemExit(f"invalid or duplicate localize manifest stem at line {line_number}: {stem}")
            if "box" not in record or "crop_bounds" not in record:
                raise SystemExit(f"localize manifest line {line_number} lacks box/crop_bounds")
            records[stem] = record
    return records



def manifest_face_mask(record, output_size):
    left, top, right, bottom = [float(value) for value in record["crop_bounds"]]
    crop_w, crop_h = right - left, bottom - top
    if crop_w <= 0 or crop_h <= 0:
        raise SystemExit(f"invalid crop bounds for {record.get('stem')}: {record['crop_bounds']}")
    sx, sy = output_size / crop_w, output_size / crop_h
    x1, y1, x2, y2 = [float(value) for value in record["box"][:4]]
    scale_x, scale_y = record.get("mask_scale", (0.92, 1.0))
    cx = ((x1 + x2) * 0.5 - left) * sx
    cy = ((y1 + y2) * 0.5 - top) * sy
    ax = max(1, int((x2 - x1) * 0.5 * float(scale_x) * sx))
    ay = max(1, int((y2 - y1) * 0.5 * float(scale_y) * sy))
    mask = np.zeros((output_size, output_size), dtype=np.uint8)
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (ax, ay), 0, 0, 360, 255, -1)
    feather = float(record.get("feather", 0.04))
    if feather > 0:
        kernel = max(3, int(round(output_size * feather)) | 1)
        mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    return mask


class PairImgs(Dataset):
    def __init__(self, root, size, aug=False, aug_level=0, pairs=None, aug_mix=None, seed=0,
                 beauty_p=None, light_p=None, part_dir=None,
                 load_masks=False, localize_manifest=None):
        din, dtg = os.path.join(root, "input"), os.path.join(root, "target")
        self.pairs = list(pairs) if pairs is not None else discover_pairs(din, dtg)
        if not self.pairs:
            raise SystemExit(f"페어 없음: {din} ∩ {dtg}")
        self.din, self.dtg, self.size = din, dtg, size
        # 구 --aug 플래그 호환: 켜져 있고 --aug-level 미지정이면 1
        self.level = aug_level if aug_level > 0 else (1 if aug else 0)
        self.aug_mix = aug_mix
        self.beauty_p = beauty_p
        self.light_p = light_p
        self.part_dir = part_dir
        self.seed = seed
        self.rng = None
        self.load_masks = load_masks
        self.localize_records = load_localize_manifest(localize_manifest)
        if localize_manifest:
            # 검출 실패·패딩 초과 페어는 build_localface_pairs 가 의도적으로 제외한다.
            # 매니페스트에 있는 것만 학습에 쓰고, 빠진 수는 기록만 남긴다.
            kept = [pair for pair in self.pairs if pair.stem in self.localize_records]
            dropped = len(self.pairs) - len(kept)
            if not kept:
                raise SystemExit(f"localize manifest matches no selected pair: {localize_manifest}")
            if dropped:
                print(f"[data] 매니페스트에 없는 {dropped}쌍 제외 → {len(kept)}쌍 사용")
            self.pairs = kept
        self.mask_dir = os.path.join(root, "mask")
        if self.load_masks and not self.localize_records and not os.path.isdir(self.mask_dir):
            raise SystemExit(f"face mask directory missing: {self.mask_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pair = self.pairs[i]
        a = cv2.cvtColor(np.array(Image.open(pair.input_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        b = cv2.cvtColor(np.array(Image.open(pair.target_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        mask = None
        if self.localize_records:
            record = self.localize_records[pair.stem]
            source_size = record.get("source_size")
            if source_size and list(source_size) != [a.shape[1], a.shape[0]]:
                raise SystemExit(
                    f"source size changed after indexing for {pair.stem}: "
                    f"manifest={source_size} actual={[a.shape[1], a.shape[0]]}"
                )
            if a.shape[:2] != b.shape[:2]:
                b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
            bounds = record["crop_bounds"]
            a = cv2.resize(
                crop_with_edge_padding(a, bounds), (self.size, self.size), interpolation=cv2.INTER_AREA
            )
            b = cv2.resize(
                crop_with_edge_padding(b, bounds), (self.size, self.size), interpolation=cv2.INTER_AREA
            )
            mask = manifest_face_mask(record, self.size)
            # 블렌딩 여부는 매니페스트가 정한다. blend=False 면 정답 = teacher crop 그대로이고
            # 타원 합성은 런타임(deid_cartoon.composite)에서만 한다.
            if record.get("blend", True):
                alpha = mask.astype(np.float32)[:, :, None] / 255.0
                b = np.clip(
                    b.astype(np.float32) * alpha + a.astype(np.float32) * (1.0 - alpha), 0, 255
                ).astype(np.uint8)
            if not self.load_masks:
                mask = None
        elif self.load_masks:
            mask_path = os.path.join(self.mask_dir, f"{pair.stem}.png")
            if not os.path.isfile(mask_path):
                raise SystemExit(f"face mask missing: {mask_path}")
            mask = np.array(Image.open(mask_path).convert("L"))
        if a.shape[:2] != b.shape[:2]:                   # teacher 출력이 더 큰 경우 정렬
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        if mask is not None and mask.shape[:2] != a.shape[:2]:
            mask = cv2.resize(mask, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)

        pmask = None
        if self.part_dir:
            ppath = os.path.join(self.part_dir, f"{pair.stem}.png")
            if not os.path.isfile(ppath):
                raise SystemExit(f"part mask missing: {ppath}  (run/build_part_masks.py 먼저)")
            pmask = cv2.imread(ppath, cv2.IMREAD_GRAYSCALE)
            if pmask is None:
                raise SystemExit(f"part mask unreadable: {ppath}")
            if pmask.shape[:2] != a.shape[:2]:
                pmask = cv2.resize(pmask, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)

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
                if pmask is not None:
                    pmask = pmask[:, ::-1]
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
                if pmask is not None:
                    pmask = pmask[y0:y0 + th, x0:x0 + tw]

        a = cv2.resize(np.ascontiguousarray(a), (self.size, self.size), interpolation=cv2.INTER_AREA)
        b = cv2.resize(np.ascontiguousarray(b), (self.size, self.size), interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(np.ascontiguousarray(mask), (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if pmask is not None:
            pmask = cv2.resize(np.ascontiguousarray(pmask), (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        if level >= 1:
            a = _degrade(a, level, rng, self.size, self.beauty_p, self.light_p)   # ★ input에만

        ta = torch.from_numpy(cv2.cvtColor(a, cv2.COLOR_BGR2RGB).copy()).permute(2, 0, 1).float() / 127.5 - 1
        tb = torch.from_numpy(cv2.cvtColor(b, cv2.COLOR_BGR2RGB).copy()).permute(2, 0, 1).float() / 127.5 - 1
        tm = (torch.from_numpy(mask.copy()).unsqueeze(0).float() / 255.0
              if mask is not None else None)
        if pmask is not None:
            tp = torch.from_numpy(pmask.copy()).unsqueeze(0).float() / 255.0
            # face_mask 자리가 비면 1로 채운다. face_mask_weight=1.0 이면 가중 L1 이
            # 평균 L1 과 수치적으로 동일하므로 기존 동작이 바뀌지 않는다.
            return ta, tb, (tm if tm is not None else torch.ones_like(tp)), tp
        if tm is None:
            return ta, tb
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
    ap.add_argument("--w-edge", type=float, default=0.0, dest="w_edge",
                    help="target 윤곽 gradient L1. 애니 눈/코/턱선 재현은 2~4 권장")
    ap.add_argument("--edge-mode", choices=("diff", "sobel-ms"), default="diff", dest="edge_mode",
                    help="윤곽 손실 방식. sobel-ms는 3개 해상도에서 선 구조를 비교")
    ap.add_argument("--conditional-gan", action="store_true", dest="conditional_gan",
                    help="PatchGAN이 input+output 쌍을 판별하여 입력과 맞는 선명한 변환을 강제")
    ap.add_argument("--w-fm", type=float, default=0.0, dest="w_fm",
                    help="discriminator feature matching. conditional GAN 안정화는 1~5 권장")
    ap.add_argument("--w-tv", type=float, default=0.0, dest="w_tv")
    ap.add_argument("--w-flat", type=float, default=0.0, dest="w_flat",
                    help="타겟이 평탄한 곳에서만 출력 gradient 를 벌한다. 잡선 억제. 1~5 권장")
    ap.add_argument("--flat-thresh", type=float, default=0.05, dest="flat_thresh",
                    help="[-1,1] 스케일 타겟 gradient 기준. 이보다 크면 '선'으로 보고 벌하지 않는다")
    ap.add_argument("--id-loss", type=float, default=0.0, dest="id_loss", help="신원억제(0=off). input 신원에서 멀어지게")
    ap.add_argument("--id-margin", type=float, default=0.3, dest="id_margin")
    ap.add_argument("--face-mask-weight", type=float, default=1.0, dest="face_mask_weight",
                    help="mask/ 내부 L1 가중치. 1=마스크 미사용, 얼굴 국소 파인튜닝은 4~8 권장")
    ap.add_argument("--localize-manifest", default=None, dest="localize_manifest",
                    help="원본 페어를 즉석 얼굴 crop/합성할 build_localface_pairs manifest.jsonl")
    ap.add_argument("--amp", choices=("off", "bf16"), default="off",
                    help="CUDA mixed precision. L40S/Ampere 이상은 bf16 권장")
    ap.add_argument("--perc-size", type=int, default=0, dest="perc_size",
                    help="VGG perceptual 계산 해상도. 0=학습 크기, 256=빠른 512 학습 권장")
    ap.add_argument("--gen-ch", type=int, default=32, dest="gen_ch", help="제너레이터 기본채널(용량↑=32→48→64, 속도↓)")
    ap.add_argument("--antialias", type=int, default=0, choices=[0, 3, 5],
                    help="stride-2 다운샘플을 BlurPool 로 교체(0=끔, 3=[1,2,1], 5=[1,4,6,4,1]). 이동 민감도를 낮춘다")
    ap.add_argument("--w-equiv", type=float, default=0.0, dest="w_equiv",
                    help="equivariance 손실 가중치. L1(G(warp x), warp G(x)). 0=끔, 시작값 20")
    ap.add_argument("--equiv-shift", type=float, default=4.0, dest="equiv_shift",
                    help="합성 모션 최대 이동(px)")
    ap.add_argument("--equiv-scale", type=float, default=0.02, dest="equiv_scale",
                    help="합성 모션 최대 스케일 변화(비율)")
    ap.add_argument("--equiv-rot", type=float, default=2.0, dest="equiv_rot",
                    help="합성 모션 최대 회전(도)")
    ap.add_argument("--equiv-border", type=int, default=16, dest="equiv_border",
                    help="warp 경계 제외 픽셀")
    ap.add_argument("--skip2-init", type=float, default=0.1, dest="skip2_init",
                    help="deep8 /2 게이트 초기 게인. 올리면 중주파 윤곽이 통과해 선명해지고 기하 자유도는 준다")
    ap.add_argument("--skip4-init", type=float, default=0.1, dest="skip4_init",
                    help="deep8 /4 게이트 초기 게인. 해상도가 낮아 위치 구속이 약하다")
    ap.add_argument("--gen-arch", choices=("legacy", "deep8"), default="legacy", dest="gen_arch",
                    help="legacy=/4 U-Net, deep8=/8 병목+약한 skip의 얼굴 redraw 구조")
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
    ap.add_argument("--w-adv-soft", type=float, default=0.0, dest="w_adv_soft",
                    help="소프트 램프 네거티브 가중치. 블러한 타겟을 판별자에 '가짜'로 추가한다. "
                         "경계 폭만 다른 두 클래스라 판별자가 폭 판정기를 만든다. 시작값 0.5")
    ap.add_argument("--soft-sigma", type=float, default=2.0, dest="soft_sigma",
                    help="네거티브를 만들 가우시안 sigma(px @512). 3을 넘기지 말 것 — "
                         "'진짜' 클래스에 halo 가 섞여 생성자가 오버슈트를 배운다")
    ap.add_argument("--part-mask-dir", default=None, dest="part_mask_dir",
                    help="run/build_part_masks.py 로 구운 부위 경계 마스크 폴더")
    ap.add_argument("--w-edge-part", type=float, default=0.0, dest="w_edge_part",
                    help="부위 경계(코·턱선·눈·눈썹·입술)에서만 걸리는 edge 손실. 시작값 3~6")
    ap.add_argument("--part-gate-thresh", type=float, default=0.06, dest="part_gate_thresh",
                    help="타겟 gradient 게이트([-1,1] 스케일). teacher가 안 그린 자리는 부위 손실 0. "
                         "0=게이트 끔(가려진 입·눈에 선을 그리라고 시키게 되므로 권장 안 함)")
    ap.add_argument("--light-p", type=float, default=None, dest="light_p",
                    help="촬영 조명 열화 확률(방향성 키라이트·하드 그림자·저조도·리림). "
                         "코퍼스는 전부 균일한 스튜디오 조명이라 이게 없으면 야간 장면에서 무너진다. "
                         "시작값 0.6")
    ap.add_argument("--beauty-p", type=float, default=None, dest="beauty_p",
                    help="뷰티필터(bilateral) 열화 확률만 따로 지정. 미지정=level별 기본(0.20/0.35). "
                         "실적용률 = P(level>=1) x beauty_p")
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
    if args.flat_thresh <= 0:
        ap.error("--flat-thresh must be positive")
    if args.perc_size < 0:
        ap.error("--perc-size must be non-negative")
    if args.w_fm < 0:
        ap.error("--w-fm must be non-negative")
    if args.w_fm > 0 and args.w_adv <= 0:
        ap.error("--w-fm requires --w-adv > 0")
    if args.gen_arch == "deep8" and args.size % 8:
        ap.error("--gen-arch deep8 requires --size divisible by 8")
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
    if args.amp != "off" and dev != "cuda":
        ap.error("--amp requires CUDA")
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_enabled = args.amp == "bf16"

    def amp_context():
        if not amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    G = build_generator(args.gen_ch, args.gen_arch,
                        skip2_init=args.skip2_init, skip4_init=args.skip4_init,
                        antialias=args.antialias).to(dev)
    D = Discriminator(args.d_ch, args.d_n, 6 if args.conditional_gan else 3).to(dev)
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
        detected_arch = checkpoint_generator_arch(init_state, init_weights)
        if detected_ch != args.gen_ch:
            raise SystemExit(
                f"--gen-ch {args.gen_ch} does not match --init-ckpt channel count {detected_ch}"
            )
        if detected_arch != args.gen_arch:
            raise SystemExit(
                f"--gen-arch {args.gen_arch} does not match --init-ckpt architecture {detected_arch}"
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
        if checkpoint_generator_arch(state, state.get("G")) != args.gen_arch:
            raise SystemExit("--gen-arch does not match resume checkpoint")
        G.load_state_dict(state["G"], strict=True)
        D.load_state_dict(state["D"], strict=True)
        optG.load_state_dict(state["optG"])
        optD.load_state_dict(state["optD"])
        start_step = int(state["step"])
        resume_state = state
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"].detach().cpu().to(torch.uint8))
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            cuda_rng = state["cuda_rng"]
            if torch.is_tensor(cuda_rng):
                cuda_rng = [cuda_rng]
            cuda_rng = [
                rng.detach().cpu().to(torch.uint8) if torch.is_tensor(rng)
                else torch.as_tensor(rng, dtype=torch.uint8)
                for rng in cuda_rng
            ]
            torch.cuda.set_rng_state_all(cuda_rng)
        if "python_rng" in state:
            random.setstate(state["python_rng"])
        if "numpy_rng" in state:
            numpy_state = state["numpy_rng"]
            np.random.set_state((
                numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                numpy_state[2], numpy_state[3], numpy_state[4],
            ))
        print(f"[resume] {resume_path} step={start_step}")

    def perceptual_loss(fake, tgt):
        if args.perc_size and fake.shape[-1] != args.perc_size:
            fake = F.interpolate(
                fake, (args.perc_size, args.perc_size), mode="bilinear", align_corners=False
            )
            tgt = F.interpolate(
                tgt, (args.perc_size, args.perc_size), mode="bilinear", align_corners=False
            )
        return vgg(fake, tgt)

    def difference_edge_loss(fake, tgt):
        fake_dx = fake[:, :, :, 1:] - fake[:, :, :, :-1]
        tgt_dx = tgt[:, :, :, 1:] - tgt[:, :, :, :-1]
        fake_dy = fake[:, :, 1:, :] - fake[:, :, :-1, :]
        tgt_dy = tgt[:, :, 1:, :] - tgt[:, :, :-1, :]
        return 0.5 * (F.l1_loss(fake_dx, tgt_dx) + F.l1_loss(fake_dy, tgt_dy))

    def sobel_edges(x):
        channels = x.shape[1]
        kx = x.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))).view(1, 1, 3, 3) / 8
        ky = kx.transpose(2, 3)
        kx = kx.repeat(channels, 1, 1, 1)
        ky = ky.repeat(channels, 1, 1, 1)
        padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return (
            F.conv2d(padded, kx, groups=channels),
            F.conv2d(padded, ky, groups=channels),
        )

    def multiscale_sobel_loss(fake, tgt):
        losses = []
        for scale in (1, 2, 4):
            if scale > 1:
                f = F.avg_pool2d(fake, scale, scale)
                t = F.avg_pool2d(tgt, scale, scale)
            else:
                f, t = fake, tgt
            fdx, fdy = sobel_edges(f)
            tdx, tdy = sobel_edges(t)
            losses.append(0.5 * (F.l1_loss(fdx, tdx) + F.l1_loss(fdy, tdy)))
        return sum(losses) / len(losses)

    def edge_loss(fake, tgt):
        if args.edge_mode == "sobel-ms":
            return multiscale_sobel_loss(fake, tgt)
        return difference_edge_loss(fake, tgt)

    def part_edge_loss(fake, tgt, pm):
        """**부위 경계 위에서만** 걸리는 sobel L1 (코·턱선·눈·눈썹·입술).

        ■ 왜 w_edge 를 올리는 대신 이것인가 (2026-08-07)
          증상은 "흐리다"가 아니라 **누락**이다 — 코가 통째로 사라진다.
          edge_density 가 teacher 의 0.85 인데 그 부족분이 균등하게 빠지지 않고
          **위치가 가장 불확실한 특징부터** 빠진다. 정면 얼굴에서 코가 정확히 그것이다.
          L1 은 위치가 애매한 선을 **안 그리는 쪽**이 최적이다.

          기존 Sobel 손실은 화면 전체를 균등하게 본다. 화소 수는 작은 gradient 쪽이
          압도적이라 가중치를 올리면 굵은 선이 아니라 **잔선**이 는다
          (w_edge 3.0→5.0: 밀도 10.74%→11.36%, 대비 181.0→179.2, 육안 "지저분해").

          부위 경계로 국한하면 그 이득 경로가 막힌다. 잔선을 늘려서는 손실이 줄지 않고
          **사라지는 특징을 그려야만** 줄어든다.

        ■ ★ 타겟 게이트 — 없는 것을 그리라고 시키지 않는다 (--part-gate-thresh)
          랜드마크는 **가려져도** 얼굴 형태를 추정해서 찍는다. 미리보기에서 확인했다:
            · 수술용 마스크 위에 입술 윤곽이 찍힘
            · 선글라스 위에 눈 윤곽이 찍힘
          그런데 teacher 타겟은 그 자리에 마스크·선글라스를 그렸지 입술·눈을 그리지 않았다.
          그대로 걸면 **마스크 위에 입술 선을 그리라고 시키는 것**이 되어 아티팩트가 된다.

          그래서 타겟 gradient 로 게이트한다. teacher 가 그 자리에 아무것도 안 그렸으면
          가중치가 0 으로 떨어진다. 우리가 고치려는 것은
          "teacher 는 코를 그렸는데 학생이 안 그렸다" 이지
          "teacher 가 안 그린 것을 그려라" 가 아니므로 방향도 맞다.

          이웃 5x5 최대값을 쓴다. 선이 1~2px 어긋나 있어도 게이트가 열려야 하기 때문이다.

        ■ 정규화
          마스크 합으로 나눈다. 그래야 마스크 면적이 변해도 가중치의 의미가 유지된다.
        """
        if args.part_gate_thresh > 0:
            with torch.no_grad():
                gdx, gdy = sobel_edges(tgt)
                gmag = (gdx.abs() + gdy.abs()).mean(1, keepdim=True)
                gmag = F.max_pool2d(gmag, 5, 1, 2)            # 이웃 허용
                pm = pm * (gmag / args.part_gate_thresh).clamp(0.0, 1.0)
        losses = []
        for scale in (1, 2, 4):
            if scale > 1:
                f = F.avg_pool2d(fake, scale, scale)
                t = F.avg_pool2d(tgt, scale, scale)
                w = F.avg_pool2d(pm, scale, scale)
            else:
                f, t, w = fake, tgt, pm
            fdx, fdy = sobel_edges(f)
            tdx, tdy = sobel_edges(t)
            denom = w.sum() * f.shape[1] + 1e-6
            losses.append(0.5 * (((fdx - tdx).abs() * w).sum()
                                 + ((fdy - tdy).abs() * w).sum()) / denom)
        return sum(losses) / len(losses)

    def flatness_loss(fake, tgt):
        """타겟이 평탄한 곳에서만 출력의 gradient 를 벌한다.

        ■ 왜 필요한가 (2026-08-06)
          목적함수에 "굵은 선을 그려라"(w_edge)는 있었지만 "평평한 면은 비워라"는 없었다.
          그래서 w_edge 를 3.0 → 5.0 으로 올리자 선이 굵어진 게 아니라 **없던 약한 선이 늘었다**
          (엣지 밀도 +5.8%, 엣지 대비 −1.0%). 긁힌 듯한 잡선이 생겨 "지저분하다"가 됐다.

          애니 화풍의 정의가 **평평한 면 + 강한 선**인데 손실에는 후자만 있었던 것이다.
          White-box Cartoonization 의 surface/texture 분해, DCT-Net 의 TV 항이 모두 전자를 담당한다.

        ■ 왜 그냥 TV(--w-tv) 를 쓰면 안 되는가
          TV 는 출력의 **모든** gradient 를 벌하므로 우리가 원하는 굵은 선까지 같이 누른다.
          타겟 gradient 로 마스크를 만들어 **평탄한 곳에만** 걸어야 한다.

        ■ flat_thresh
          [-1,1] 스케일의 타겟 gradient 기준값. 이보다 크면 '선'으로 보고 벌하지 않는다.
          0.05 ≈ 6/255. 너무 높이면 진짜 선까지 눌러 전체가 밋밋해진다.
        """
        fdx = fake[:, :, :, 1:] - fake[:, :, :, :-1]
        fdy = fake[:, :, 1:, :] - fake[:, :, :-1, :]
        with torch.no_grad():
            tdx = (tgt[:, :, :, 1:] - tgt[:, :, :, :-1]).abs().mean(1, keepdim=True)
            tdy = (tgt[:, :, 1:, :] - tgt[:, :, :-1, :]).abs().mean(1, keepdim=True)
            wx = 1.0 - (tdx / args.flat_thresh).clamp(0.0, 1.0)
            wy = 1.0 - (tdy / args.flat_thresh).clamp(0.0, 1.0)
        return 0.5 * ((fdx.abs().mean(1, keepdim=True) * wx).mean()
                      + (fdy.abs().mean(1, keepdim=True) * wy).mean())

    def set_requires_grad(model, enabled):
        for parameter in model.parameters():
            parameter.requires_grad_(enabled)

    def g_losses(inp, tgt, w_adv_eff, face_mask=None, part_mask=None):
        with amp_context():
            fake = G(inp)
            if face_mask is not None:
                weights = 1.0 + (args.face_mask_weight - 1.0) * face_mask
                l1 = ((fake - tgt).abs() * weights).sum() / (weights.sum() * fake.shape[1])
                # 바깥을 target으로 치환하면 perceptual gradient가 얼굴 주변에 집중된다.
                perceptual_fake = fake * face_mask + tgt.detach() * (1.0 - face_mask)
                perc = perceptual_loss(perceptual_fake, tgt)
            else:
                l1 = F.l1_loss(fake, tgt)                # target 직접 재현
                perc = perceptual_loss(fake, tgt)        # 다층 perceptual
            edge = edge_loss(perceptual_fake if face_mask is not None else fake, tgt)
            g = args.w_l1 * l1 + args.w_perc * perc + args.w_edge * edge
            edge_part = torch.tensor(0.0, device=dev)
            if part_mask is not None and args.w_edge_part > 0:
                edge_part = part_edge_loss(
                    perceptual_fake if face_mask is not None else fake, tgt, part_mask)
                g = g + args.w_edge_part * edge_part
            adv = torch.tensor(0.0, device=dev)
            fm = torch.tensor(0.0, device=dev)
            if w_adv_eff > 0:
                fake_d_input = torch.cat([inp, fake], 1) if args.conditional_gan else fake
                real_d_input = torch.cat([inp, tgt], 1) if args.conditional_gan else tgt
                df, fake_features = D(fake_d_input, return_features=True)
                adv = F.mse_loss(df, torch.ones_like(df))
                g = g + w_adv_eff * adv
                if args.w_fm > 0:
                    with torch.no_grad():
                        _, real_features = D(real_d_input, return_features=True)
                    fm = sum(
                        F.l1_loss(fake_feature, real_feature)
                        for fake_feature, real_feature in zip(fake_features, real_features)
                    ) / len(fake_features)
                    # Introduce feature matching on the same ramp as adversarial training.
                    fm_ramp = min(1.0, w_adv_eff / max(args.w_adv, 1e-12))
                    g = g + args.w_fm * fm_ramp * fm
            equiv = torch.tensor(0.0, device=dev)
            if args.w_equiv > 0:
                # [시간적 안정성] "옮기고 그리기" == "그리고 옮기기" 를 강제한다.
                # 정지 이미지만으로 프레임 간 안정성을 학습시키는 방법
                # (StableLLVE, CVPR 2021, MIT). 실사 영상이 필요 없어 저작권 제약을 우회한다.
                # 추론 비용은 0이고 학습 forward 만 1회 늘어난다.
                theta = random_affine_theta(inp.shape[0], dev, args.equiv_shift,
                                            args.equiv_scale, args.equiv_rot, inp.shape[-1])
                fake_of_warped = G(warp_affine(inp, theta))
                warped_fake = warp_affine(fake, theta)
                b = max(1, int(args.equiv_border))
                # 경계는 warp 로 생긴 인공 픽셀이라 손실에서 제외한다.
                equiv = F.l1_loss(fake_of_warped[:, :, b:-b, b:-b],
                                  warped_fake[:, :, b:-b, b:-b])
                g = g + args.w_equiv * equiv
            flat = torch.tensor(0.0, device=dev)
            if args.w_flat > 0:
                flat = flatness_loss(fake, tgt)
                g = g + args.w_flat * flat
            if args.w_tv > 0:
                tv = ((fake[:, :, 1:] - fake[:, :, :-1]).abs().mean()
                      + (fake[:, :, :, 1:] - fake[:, :, :, :-1]).abs().mean())
                g = g + args.w_tv * tv
            idl = torch.tensor(0.0, device=dev)
            if idm is not None:                          # 신원: fake를 input 신원에서 멀어지게
                cos = (id_embed(idm, inp) * id_embed(idm, fake)).sum(1)
                idl = F.relu(cos - args.id_margin).mean()
                g = g + args.id_loss * idl
        return fake, g, dict(l1=l1.item(), perc=perc.item(), edge=edge.item(),
                             edge_part=float(edge_part.detach()),
                             adv=float(adv.detach()), fm=float(fm.detach()),
                             equiv=float(equiv.detach()), flat=float(flat.detach()),
                             idl=float(idl.detach()))

    def soft_ramp(x, sigma):
        """타겟의 **명암 경계만** 완만하게 만든 사본. 가우시안 블러.

        색·구도·선 개수·신원은 그대로고 경계 폭만 달라진다. 그게 핵심이다.
        """
        radius = max(1, int(round(3 * sigma)))
        k = 2 * radius + 1
        t = torch.arange(k, device=x.device, dtype=x.dtype) - radius
        g = torch.exp(-(t ** 2) / (2 * sigma * sigma))
        g = (g / g.sum()).view(1, 1, 1, k)
        c = x.shape[1]
        y = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"),
                     g.repeat(c, 1, 1, 1), groups=c)
        y = F.conv2d(F.pad(y, (0, 0, radius, radius), mode="reflect"),
                     g.transpose(2, 3).repeat(c, 1, 1, 1), groups=c)
        return y

    def d_loss(inp, tgt):
        """LSGAN 판별자 손실. --w-adv-soft 가 켜지면 **소프트 램프 네거티브**가 추가된다.

        ■ 왜 필요한가 (2026-08-07)
          요구는 "명암 경계가 계단처럼 딱 끊겼으면"인데, 학생 계단 비율이 3.7% 이고
          teacher 는 9.2% 다(run/transition_width.py). **천장까지 2.5배 여유가 있다.**

          그런데 학생 쪽 손실을 손으로 설계한 시도가 세 번 모두 실패했다.
            w_edge 3→5      ❌ 잔선만 늘었다
            w_flat 2.0      ❌ 전 지표 악화. 없는 문제를 고치려 했다
            부위 게이트 edge ❌ 계단 3.7%→2.0%. 선은 더 그렸는데 더 부드럽게 그렸다

          마지막 것이 원인을 알려줬다. **Sobel L1 은 폭에 눈이 멀었다.**
          12px 완만한 경사와 1px 계단의 총 gradient 크기가 같으면 손실이 같다.
          위치를 좁혀도(부위 게이트) 폭에 눈먼 손실은 여전히 폭에 눈이 멀었다.
          지표에서 이미 발견한 맹점을 손실에서 그대로 반복한 것이다.

        ■ 그래서 기준을 손으로 쓰지 않고 판별자에게 배우게 한다
          CartoonGAN(CVPR 2018)의 edge-promoting 을 **선 대신 명암 경계로** 옮긴다.
          원 논문은 윤곽선을 뭉갠 만화 이미지를 판별자에 '가짜'로 넣어 선을 얻었다.

              L_D = (D(y)-1)^2 + D(G(x))^2 + w_soft * D(blur(y))^2
                                             └── 진짜인데 가짜로 라벨링

          `y` 와 `blur(y)` 는 팔레트·구도·선 개수·신원이 **전부 같고 경계 폭만 다르다.**
          판별자가 볼 수 있는 단서가 그것뿐이므로 **폭 판정기를 만들 수밖에 없고**,
          생성자 기울기가 정확히 그 축을 가리킨다.
          생성자 손실은 건드리지 않는다 — 판별자가 보는 대상만 좁히는 것이다.

        ■ --w-adv 를 올리는 것과 무엇이 다른가
          그것은 **모든** GAN 압력을 키운다. 구조보다 아티팩트가 먼저 온다.
          이쪽은 압력의 크기가 아니라 **방향**을 바꾼다.

        ■ 주의
          sigma 를 키우면 '진짜' 클래스에 오버슈트·halo 가 포함돼 생성자가 그것을 배운다.
          sigma <= 3, w_soft <= 1.0 을 지킬 것.
        """
        with amp_context():
            fake = G(inp).detach()
            if args.conditional_gan:
                dr, df = D(torch.cat([inp, tgt], 1)), D(torch.cat([inp, fake], 1))
            else:
                dr, df = D(tgt), D(fake)                  # 진짜 애니 target →1, 생성물 →0
            loss = 0.5 * (F.mse_loss(dr, torch.ones_like(dr))
                          + F.mse_loss(df, torch.zeros_like(df)))
            if args.w_adv_soft > 0:
                soft = soft_ramp(tgt, args.soft_sigma)
                ds = D(torch.cat([inp, soft], 1)) if args.conditional_gan else D(soft)
                loss = loss + args.w_adv_soft * F.mse_loss(ds, torch.zeros_like(ds))
            return loss

    if args.smoke:
        print(f"[smoke] dev={dev} size={args.size}")
        inp = torch.rand(2, 3, args.size, args.size, device=dev) * 2 - 1
        tgt = torch.rand(2, 3, args.size, args.size, device=dev) * 2 - 1
        set_requires_grad(D, True)
        dl = d_loss(inp, tgt); optD.zero_grad(); dl.backward(); optD.step()
        set_requires_grad(D, False)
        fake, gl, parts = g_losses(inp, tgt, args.w_adv); optG.zero_grad(); gl.backward(); optG.step()
        set_requires_grad(D, True)
        d_input = torch.cat([inp, fake], 1) if args.conditional_gan else fake
        print(f"[smoke] out={tuple(fake.shape)} D_patch={tuple(D(d_input).shape)} d={dl.item():.3f} g={gl.item():.3f} {parts}")
        print(f"[smoke] G params={sum(p.numel() for p in G.parameters())/1e6:.2f}M → 배선 OK")
        return

    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    all_pairs = discover_pairs(os.path.join(args.data, "input"), os.path.join(args.data, "target"))
    if args.localize_manifest:
        localized_stems = set(load_localize_manifest(args.localize_manifest))
        before = len(all_pairs)
        all_pairs = [pair for pair in all_pairs if pair.stem in localized_stems]
        print(
            f"[localize] indexed={len(all_pairs)} excluded_without_face={before - len(all_pairs)}"
        )
        if not all_pairs:
            raise SystemExit("localize manifest has no stems matching the paired corpus")
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
        args.beauty_p, args.light_p, args.part_mask_dir,
        load_masks=use_face_masks, localize_manifest=args.localize_manifest,
    )
    val_ds = (
        PairImgs(
            args.data, args.size, pairs=val_pairs, seed=args.seed,
            load_masks=use_face_masks, localize_manifest=args.localize_manifest,
        )
        if val_pairs else None
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = cycle(DataLoader(
        ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
        pin_memory=(dev == "cuda"), persistent_workers=(args.workers > 0), generator=generator,
    ))
    print(f"[data] train={len(ds)} val={len(val_ds) if val_ds else 0} "
          f"aug_mix={aug_mix} beauty_p={args.beauty_p} light_p={args.light_p} part_masks={args.part_mask_dir or 'none'} "
          f"face_mask_weight={args.face_mask_weight:g} "
          f"localize_manifest={args.localize_manifest or 'none'}")
    # ★ 손실 가중치를 반드시 로그에 남긴다.
    #   base 가 w_edge=0 인 줄 모르고 30,000스텝을 돌린 사고(2026-08-04)의 재발 방지.
    print(f"[loss] w_l1={args.w_l1:g} w_perc={args.w_perc:g} w_adv={args.w_adv:g} "
          f"w_edge={args.w_edge:g}({args.edge_mode}) w_edge_part={args.w_edge_part:g} "
          f"w_equiv={args.w_equiv:g} "
          f"w_fm={args.w_fm:g} w_tv={args.w_tv:g} w_flat={args.w_flat:g} id_loss={args.id_loss:g}")
    print(f"[model] gen_arch={args.gen_arch} gen_ch={args.gen_ch} "
          f"params={sum(p.numel() for p in G.parameters())/1e6:.2f}M")
    print(f"[speed] amp={args.amp} perc_size={args.perc_size or args.size} tf32={dev == 'cuda'}")

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
            with amp_context():
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
                    with amp_context():
                        vf = G(va).clamp(-1, 1)
                    losses.extend(F.l1_loss(vf, vt, reduction="none").mean((1, 2, 3)).cpu().tolist())
                    if len(batch[0]) >= 3:
                        vm = torch.stack([item[2] for item in batch]).to(dev)
                        face_error = ((vf - vt).abs() * vm).sum((1, 2, 3))
                        face_norm = vm.sum((1, 2, 3)).clamp_min(1.0) * vf.shape[1]
                        face_losses.extend((face_error / face_norm).cpu().tolist())
                suffix = f" face_l1={np.mean(face_losses):.4f}" if face_losses else ""
                print(f"[val:{step}] n={len(losses)} l1={np.mean(losses):.4f}{suffix}")
        G.train()

    log_time = time.perf_counter()
    log_step = start_step
    for step in range(start_step + 1, args.steps + 1):
        train_batch = next(loader)
        inp, tgt = train_batch[:2]
        face_mask = train_batch[2] if len(train_batch) >= 3 else None
        part_mask = train_batch[3] if len(train_batch) == 4 else None
        inp = inp.to(dev, non_blocking=True)
        tgt = tgt.to(dev, non_blocking=True)
        if face_mask is not None:
            face_mask = face_mask.to(dev, non_blocking=True)
        if part_mask is not None:
            part_mask = part_mask.to(dev, non_blocking=True)
        w_adv_eff = 0.0 if step <= args.init_steps else args.w_adv * min(1.0, (step - args.init_steps) / max(1, args.adv_ramp))
        if w_adv_eff > 0:
            set_requires_grad(D, True)
            dl = d_loss(inp, tgt); optD.zero_grad(); dl.backward(); optD.step()
        else:
            dl = torch.tensor(0.0)
        set_requires_grad(D, False)
        fake, gl, parts = g_losses(inp, tgt, w_adv_eff, face_mask, part_mask); optG.zero_grad(); gl.backward(); optG.step()
        set_requires_grad(D, True)
        if step % 100 == 0:
            elapsed = time.perf_counter() - log_time
            steps_per_second = (step - log_step) / max(elapsed, 1e-9)
            print(f"[{step}/{args.steps}] wadv={w_adv_eff:.2f} D={float(dl):.3f} G={gl.item():.3f} "
                  f"l1={parts['l1']:.3f} perc={parts['perc']:.3f} edge={parts['edge']:.3f} "
                  f"epart={parts['edge_part']:.4f} adv={parts['adv']:.2f} fm={parts['fm']:.3f} eqv={parts['equiv']:.4f} "
                  f"flat={parts['flat']:.4f} "
                  f"id={parts['idl']:.3f} step/s={steps_per_second:.2f}")
            log_time = time.perf_counter()
            log_step = step
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
