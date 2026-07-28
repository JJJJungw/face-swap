#!/usr/bin/env python3
"""
④-C 학생 학습 (White-box 방식) — flat 셀셰이딩 카툰化.

배경: paired/unpaired 둘 다 "유화(painterly)"에서 못 벗어남. 원인 = 평면화(flattening) 손실 부재.
근거: White-box Cartoonization(Wang & Yu, CVPR 2020) — feed-forward로 flat 카툰을 만드는 검증된 방법.
      3가지 표현 손실로 평면·색블록·윤곽을 명시적으로 강제함. (초상 포함 입증)
      원 코드/가중치는 CC BY-NC-SA(비상업)라 사용 불가 → "방법(손실 설계)"만 자체 재구현(클린).

구성:
  - Generator = animegan2 구조(런타임 호환) — 결과 .pt를 런타임 셸에 그대로 삽입
  - Ds(surface), Dt(texture) = PatchGAN 판별자 2개
  - 3표현 손실(White-box Eq.1~8):
      Surface : guided_filter로 평면 추출 → Ds adversarial      (매끈한 평면)
      Structure: felzenszwalb superpixel + 적응채색 → VGG 자기매칭 (셀셰이딩 색블록) ★flat 핵심
      Texture : random color shift(색 제거) → Dt adversarial       (선명한 윤곽)
      Content : VGG(G(p)) vs VGG(p)                                (구조 보존)
      TV      : 평활
  - 데이터(unpaired): photo 코퍼스(SFHQ) + cartoon 코퍼스(우리 애니 뱅크)

의존성: pip install scikit-image scipy  (felzenszwalb, ndimage.mean 사용)

사용:
  python train/train_student_whitebox.py --smoke
  python train/train_student_whitebox.py \
    --photo input/sfhq_t2i/a_small_sample_new \
    --cartoon out/pairs_dataset/target \
    --out train/student_wb --size 256 --batch 4 --steps 40000
"""
import os, argparse, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image

EXTS = (".png", ".jpg", ".jpeg", ".webp")


# ============ Generator (animegan2 구조, 런타임 호환) ============
class ConvNormLReLU(nn.Sequential):
    def __init__(self, i, o, k=3, s=1, p=1):
        super().__init__(nn.ReflectionPad2d(p), nn.Conv2d(i, o, k, s, 0, bias=False),
                         nn.GroupNorm(1, o, affine=True), nn.LeakyReLU(0.2, inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, i, o, e=2):
        super().__init__()
        self.res = (i == o); m = round(i * e); L = []
        if e != 1:
            L.append(ConvNormLReLU(i, m, k=1, p=0))
        L += [nn.ReflectionPad2d(1), nn.Conv2d(m, m, 3, 1, 0, groups=m, bias=False),
              nn.GroupNorm(1, m, affine=True), nn.LeakyReLU(0.2, inplace=True),
              nn.Conv2d(m, o, 1, 1, 0, bias=False), nn.GroupNorm(1, o, affine=True)]
        self.body = nn.Sequential(*L)

    def forward(self, x):
        y = self.body(x)
        return y + x if self.res else y


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Sequential(ConvNormLReLU(3, 32, k=7, p=3), ConvNormLReLU(32, 64, s=2), ConvNormLReLU(64, 64))
        self.b = nn.Sequential(ConvNormLReLU(64, 128, s=2), ConvNormLReLU(128, 128))
        self.c = nn.Sequential(ConvNormLReLU(128, 128), InvertedResidual(128, 256),
                               InvertedResidual(256, 256), InvertedResidual(256, 256),
                               InvertedResidual(256, 256), ConvNormLReLU(256, 128))
        self.d = nn.Sequential(ConvNormLReLU(128, 128), ConvNormLReLU(128, 128))
        self.e = nn.Sequential(ConvNormLReLU(128, 64), ConvNormLReLU(64, 64), ConvNormLReLU(64, 32, k=7, p=3))
        self.out = nn.Sequential(nn.Conv2d(32, 3, 1, 1, 0, bias=False), nn.Tanh())

    def forward(self, x):
        h = self.a(x); s1 = h.shape[-2:]
        h = self.b(h); h = self.c(h)
        h = F.interpolate(h, s1, mode="bilinear", align_corners=False)
        h = self.d(h)
        h = F.interpolate(h, x.shape[-2:], mode="bilinear", align_corners=False)
        h = self.e(h)
        return self.out(h)


class Discriminator(nn.Module):
    def __init__(self, ch=32, n=3):
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


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            v = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        except Exception as e:
            print(f"[vgg] 다운로드 실패 → 랜덤init(스모크): {e}")
            v = models.vgg19(weights=None)
        self.v = v.features[:27].eval()          # conv4_4
        for p in self.v.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):                        # x in [-1,1]
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        return self.v(x)


# ============ White-box 3표현 추출기 ============
def _box(x, r):
    return F.avg_pool2d(x, 2 * r + 1, 1, r, count_include_pad=False)


def guided_filter(x, y, r=5, eps=2e-2):
    """미분가능 guided filter (edge-preserving smoothing). x=guide, y=input. [-1,1] 그대로 동작."""
    mx = _box(x, r); my = _box(y, r)
    cov = _box(x * y, r) - mx * my
    var = _box(x * x, r) - mx * mx
    A = cov / (var + eps); b = my - A * mx
    return _box(A, r) * x + _box(b, r)


def surface_repr(x):                              # Fdgf(I,I): 질감 제거·평면
    return guided_filter(x, x, r=5)


def random_color_shift(x):
    """Frcs: RGB→랜덤 단일강도(색 제거, 선/텍스처만). White-box Eq.4 (α=0.8, β~U(-1,1))."""
    B = x.size(0); dev = x.device
    x01 = x * 0.5 + 0.5
    r, g, b = x01[:, 0:1], x01[:, 1:2], x01[:, 2:3]
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    be = torch.rand(B, 3, 1, 1, device=dev) * 2 - 1
    mix = be[:, 0:1] * r + be[:, 1:2] * g + be[:, 2:3] * b
    out = 0.2 * mix + 0.8 * Y                      # (1-α)mix + αY
    out = out.repeat(1, 3, 1, 1)
    return out * 2 - 1                              # [-1,1]


def structure_repr_np(img_m11):
    """felzenszwalb superpixel + 적응채색(평면 색블록). White-box Eq.2 근사(세그먼트 std로 mean/median 선택)."""
    from skimage.segmentation import felzenszwalb
    from scipy import ndimage
    img = (img_m11 * 0.5 + 0.5).clip(0, 1)         # HxWx3 [0,1]
    seg = felzenszwalb(img, scale=32, sigma=0.6, min_size=100)
    labels = np.unique(seg)
    out = np.zeros_like(img)
    for c in range(3):
        ch = img[:, :, c]
        means = ndimage.mean(ch, seg, labels)
        med = ndimage.median(ch, seg, labels)
        std = ndimage.standard_deviation(ch, seg, labels) * 255.0
        # Eq.2: 세그먼트 색편차(std)로 mean/median 블렌드
        t1 = np.where(std < 20, 0.0, np.where(std < 40, 0.5, 1.0))
        val = t1 * means + (1 - t1) * med
        lut = np.zeros(int(labels.max()) + 1); lut[labels] = val
        out[:, :, c] = lut[seg]
    return (out * 2 - 1).astype(np.float32)         # [-1,1]


def structure_repr_batch(x):                        # x: B,3,H,W [-1,1] (detached)
    arr = x.detach().permute(0, 2, 3, 1).cpu().numpy()   # B,H,W,3
    outs = [structure_repr_np(a) for a in arr]
    t = torch.from_numpy(np.stack(outs)).permute(0, 3, 1, 2)
    return t.to(x.device)


def tv_loss(x):
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


# ============ 데이터 (unpaired) ============
class Imgs(Dataset):
    def __init__(self, root, size):
        self.paths = sorted(p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                            if p.lower().endswith(EXTS))
        if not self.paths:
            raise SystemExit(f"이미지 없음: {root}")
        self.tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")) * 2 - 1


def cycle(dl):
    while True:
        for x in dl:
            yield x


# ============ 학습 ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", default="input/sfhq_t2i/a_small_sample_new")
    ap.add_argument("--cartoon", default="out/pairs_dataset/target")
    ap.add_argument("--out", default="train/student_wb")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--init-lr", type=float, default=1e-4, dest="init_lr")
    ap.add_argument("--init-steps", type=int, default=3000, dest="init_steps", help="content-only 워밍업(픽셀+VGG)")
    ap.add_argument("--steps", type=int, default=40000)
    # White-box 가중치(우리 VGG 스케일로 보정 — CLI로 튜닝)
    ap.add_argument("--w-surface", type=float, default=1.0, dest="w_surface", help="평면(adversarial)")
    ap.add_argument("--w-texture", type=float, default=2.0, dest="w_texture", help="윤곽선(adversarial)")
    ap.add_argument("--w-structure", type=float, default=6.0, dest="w_structure", help="색블록(VGG 자기매칭) ★flat")
    ap.add_argument("--w-content", type=float, default=6.0, dest="w_content", help="구조 보존(VGG)")
    ap.add_argument("--w-tv", type=float, default=1.0, dest="w_tv")
    ap.add_argument("--sample-every", type=int, default=500, dest="sample_every")
    ap.add_argument("--ckpt-every", type=int, default=5000, dest="ckpt_every")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator().to(dev)
    Ds, Dt = Discriminator().to(dev), Discriminator().to(dev)     # surface, texture
    vgg = VGG().to(dev)
    optG = Adam(G.parameters(), args.lr, betas=(0.5, 0.999))
    optD = Adam(list(Ds.parameters()) + list(Dt.parameters()), args.lr, betas=(0.5, 0.999))

    def g_losses(p, c):
        fake = G(p)
        # surface (adversarial, LSGAN): 평면 추출 후 Ds 속이기
        ssf = surface_repr(fake); df_s = Ds(ssf)
        l_surf = F.mse_loss(df_s, torch.ones_like(df_s))
        # texture (adversarial): 색제거 후 Dt 속이기
        tf = random_color_shift(fake); df_t = Dt(tf)
        l_tex = F.mse_loss(df_t, torch.ones_like(df_t))
        # structure (VGG 자기매칭): 출력 vs 자기 superpixel-평면화 → 색블록 강제 ★
        st = structure_repr_batch(fake)
        l_struct = F.l1_loss(vgg(fake), vgg(st))
        # content (구조 보존)
        l_con = F.l1_loss(vgg(fake), vgg(p))
        l_tv = tv_loss(fake)
        g = (args.w_surface * l_surf + args.w_texture * l_tex + args.w_structure * l_struct
             + args.w_content * l_con + args.w_tv * l_tv)
        return fake, g, dict(surf=l_surf.item(), tex=l_tex.item(), struct=l_struct.item(),
                             con=l_con.item(), tv=l_tv.item())

    def d_losses(p, c):
        fake = G(p).detach()
        # surface D: 진짜 카툰 평면=1, 생성 평면=0
        drs = Ds(surface_repr(c)); dfs = Ds(surface_repr(fake))
        l_s = F.mse_loss(drs, torch.ones_like(drs)) + F.mse_loss(dfs, torch.zeros_like(dfs))
        # texture D: 진짜 카툰 텍스처=1, 생성 텍스처=0
        drt = Dt(random_color_shift(c)); dft = Dt(random_color_shift(fake))
        l_t = F.mse_loss(drt, torch.ones_like(drt)) + F.mse_loss(dft, torch.zeros_like(dft))
        return l_s + l_t

    if args.smoke:
        print(f"[smoke] dev={dev} size={args.size}")
        p = torch.rand(2, 3, args.size, args.size, device=dev) * 2 - 1
        c = torch.rand(2, 3, args.size, args.size, device=dev) * 2 - 1
        dl = d_losses(p, c); optD.zero_grad(); dl.backward(); optD.step()
        fake, gl, parts = g_losses(p, c); optG.zero_grad(); gl.backward(); optG.step()
        print(f"[smoke] out={tuple(fake.shape)} d={dl.item():.3f} g={gl.item():.3f} {parts}")
        print(f"[smoke] surface/structure/texture 추출 OK, G params={sum(x.numel() for x in G.parameters())/1e6:.2f}M")
        return

    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    photo_ds, cart_ds = Imgs(args.photo, args.size), Imgs(args.cartoon, args.size)
    photo = cycle(DataLoader(photo_ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True))
    cart = cycle(DataLoader(cart_ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True))
    print(f"[data] photo={len(photo_ds.paths)} cartoon={len(cart_ds.paths)}")

    eval_ds = Imgs(args.photo, args.size)
    n_eval = min(4, len(eval_ds))
    eval_p = torch.stack([eval_ds[i] for i in range(n_eval)]).to(dev)

    def save_eval(step):
        G.eval()
        with torch.no_grad():
            fake = G(eval_p).clamp(-1, 1)
            st = structure_repr_batch(fake)                    # 색블록 표현도 같이 저장(디버그)
        grid = torch.cat([eval_p, fake, st], 0) * 0.5 + 0.5     # 입력 | 학생 | structure표현
        save_image(grid, os.path.join(args.out, "samples", f"s{step:06d}.png"), nrow=n_eval)
        G.train()

    # 워밍업: 픽셀 L1 + VGG (회색붕괴 방지, 검증된 방식)
    for grp in optG.param_groups:
        grp["lr"] = args.init_lr
    for step in range(1, args.init_steps + 1):
        p = next(photo).to(dev)
        out = G(p)
        recon = F.l1_loss(out, p) + 0.5 * F.l1_loss(vgg(out), vgg(p))
        optG.zero_grad(); recon.backward(); optG.step()
        if step % 200 == 0:
            print(f"[init {step}/{args.init_steps}] recon={recon.item():.3f}")
    for grp in optG.param_groups:
        grp["lr"] = args.lr
    save_eval(0)
    print("[init] 워밍업 끝 → s000000.png")

    for step in range(1, args.steps + 1):
        p, c = next(photo).to(dev), next(cart).to(dev)
        dl = d_losses(p, c); optD.zero_grad(); dl.backward(); optD.step()
        p, c = next(photo).to(dev), next(cart).to(dev)
        fake, gl, parts = g_losses(p, c); optG.zero_grad(); gl.backward(); optG.step()
        if step % 100 == 0:
            print(f"[{step}/{args.steps}] D={dl.item():.3f} G={gl.item():.3f} surf={parts['surf']:.2f} "
                  f"tex={parts['tex']:.2f} struct={parts['struct']:.3f} con={parts['con']:.3f}")
        if step % args.sample_every == 0:
            save_eval(step)
        if step % args.ckpt_every == 0:
            torch.save({"G": G.state_dict(), "step": step}, os.path.join(args.out, f"student_{step:06d}.pt"))
    torch.save({"G": G.state_dict(), "step": args.steps}, os.path.join(args.out, "student_final.pt"))
    print(f"완료 → {args.out}/student_final.pt")


if __name__ == "__main__":
    main()
