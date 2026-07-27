#!/usr/bin/env python3
"""
④ 학생 학습 — AnimeGANv2 (unpaired) + 신원 억제(identity-suppression) 손실.

증류 파이프라인 ④단계 (B: 엔드투엔드 노선).
- 정렬 페어 X.  photo 코퍼스(SFHQ 실사) + style 코퍼스(LoRA-Chroma로 뽑은 2.5D 애니) unpaired.
- 역할 분담:
    구조·표정 보존 = content loss (VGG19 conv4_4)   ← 학생이 담당(선생님 드리프트 무관)
    화풍         = GAN(adv) + grayscale gram style + color 복원
    신원 제거     = identity-suppression (얼굴 임베딩 코사인 억제)

라이선스 메모(중요):
- 본 파일 = 우리 소유. Generator 구조는 bryandlee/animegan2-pytorch(MIT) 재구현·귀속.
- VGG19(perceptual) : torchvision ImageNet 가중치 — **학습 전용, 런타임 미포함**.
- 신원 임베더(--id-backbone facenet) : **학습 전용, 런타임 미포함**. facenet(VGGFace2 학습) 가중치는
  라이선스가 애매하니 상용 배포 전 반드시 확인. 완전 클린을 원하면 `--id-loss 0` 으로 끄고
  별도 재식별 검증(후처리)으로 대체 가능. 신원 억제는 학생 가중치에 '효과'로만 남고 임베더 자체는 안 나감.

사용:
  # (A) 스모크 — 랜덤텐서 1스텝, 아키텍처/손실 배선 확인 (모델 없이 즉시)
  python train/train_student.py --smoke
  # (B) 실제 학습
  python train/train_student.py \
    --photo input/sfhq_t2i/a_small_sample_new \
    --style out/pairs_dataset/target \
    --out train/student_out --size 256 --batch 4 \
    --init-steps 2000 --steps 60000 --id-loss 0.3
  # 도메인 갭 보정(입력 augmentation) 켜기:  --aug
"""
import os, argparse, glob
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


# ============ Generator (animegan2-pytorch 구조 재구현, MIT 귀속) ============
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
        h = self.b(h)
        h = self.c(h)
        h = F.interpolate(h, s1, mode="bilinear", align_corners=False)
        h = self.d(h)
        h = F.interpolate(h, x.shape[-2:], mode="bilinear", align_corners=False)
        h = self.e(h)
        return self.out(h)


# ============ Discriminator (PatchGAN + spectral norm) ============
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


# ============ VGG19 perceptual (conv4_4 = features[:27]) ============
class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            v = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        except Exception as e:
            print(f"[vgg] 사전학습 가중치 다운로드 실패 → 랜덤init(스모크용): {e}")
            v = models.vgg19(weights=None)
        self.v = v.features[:27].eval()          # conv4_4 (AnimeGAN 정석 — content·gram 공용)
        for p in self.v.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):          # x in [-1,1] → conv4_4 특징
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        return self.v(x)


# ============ 손실 유틸 ============
def gram(f):
    b, c, h, w = f.shape
    f = f.view(b, c, h * w)
    return f.bmm(f.transpose(1, 2)) / (h * w)   # /(h*w) — Gatys 표준 스케일


def gray3(x):                      # RGB[-1,1] -> luminance, 3채널 복제
    y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return y.repeat(1, 3, 1, 1)


def rgb2yuv(x):                    # [-1,1] -> [0,1] -> YUV
    x = x * 0.5 + 0.5
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return y, u, v


def color_loss(a, b):              # 입력 색감 보존(Y 강, UV 약)
    ya, ua, va = rgb2yuv(a); yb, ub, vb = rgb2yuv(b)
    return F.l1_loss(ya, yb) + F.smooth_l1_loss(ua, ub) + F.smooth_l1_loss(va, vb)


def tv_loss(x):
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + \
           (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


def edge_smooth(x):
    """엣지 부위만 블러 처리한 버전 → D에 '가짜'로 넣어 생성자가 선명한 윤곽을 그리게 강제(AnimeGAN 엣지촉진)."""
    dev = x.device
    kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=dev).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    g = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    ex = F.conv2d(g, kx, padding=1); ey = F.conv2d(g, ky, padding=1)
    mag = torch.sqrt(ex * ex + ey * ey + 1e-8)
    m = (mag > mag.mean() * 1.5).float()          # 엣지 마스크
    m = F.max_pool2d(m, 3, 1, 1)                   # 엣지 팽창
    blur = F.avg_pool2d(x, 5, 1, 2)                # 박스 블러
    return x * (1 - m) + blur * m


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


def id_embed(m, x):                # x[-1,1] -> 160 리사이즈 -> L2정규화 임베딩
    x = F.interpolate(x, 160, mode="bilinear", align_corners=False)
    return F.normalize(m(x), dim=1)


# ============ 데이터 (unpaired) ============
def make_tf(size, aug):
    ops = [transforms.Resize((size, size))]
    if aug:   # 입력측 도메인 랜덤화(실사 스마트폰 느낌 근사) — 기본 off
        ops += [transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3)]
    ops += [transforms.ToTensor()]
    return transforms.Compose(ops)


class Imgs(Dataset):
    def __init__(self, root, size, aug=False):
        self.paths = sorted(p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                            if p.lower().endswith(EXTS))
        if not self.paths:
            raise SystemExit(f"이미지 없음: {root}")
        self.tf = make_tf(size, aug)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        return self.tf(im) * 2 - 1     # [-1,1]


def cycle(dl):
    while True:
        for x in dl:
            yield x


# ============ 학습 ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", default="input/sfhq_t2i/a_small_sample_new")
    ap.add_argument("--style", default="out/pairs_dataset/target")
    ap.add_argument("--out", default="train/student_out")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr-g", type=float, default=2e-5, dest="lr_g", help="G lr (AnimeGAN 2e-5)")
    ap.add_argument("--lr-d", type=float, default=4e-5, dest="lr_d", help="D lr (G의 2배)")
    ap.add_argument("--init-lr", type=float, default=1e-4, dest="init_lr", help="워밍업 lr")
    ap.add_argument("--init-steps", type=int, default=3000, dest="init_steps", help="content-only 워밍업(~10ep)")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--w-adv", type=float, default=300.0, dest="w_adv", help="AnimeGAN 정석 300")
    ap.add_argument("--w-con", type=float, default=1.5, dest="w_con")
    ap.add_argument("--w-sty", type=float, default=3.0, dest="w_sty")
    ap.add_argument("--w-col", type=float, default=10.0, dest="w_col")
    ap.add_argument("--w-tv", type=float, default=1.0, dest="w_tv")
    ap.add_argument("--id-loss", type=float, default=0.3, dest="id_loss", help="신원억제 가중(0=off)")
    ap.add_argument("--id-margin", type=float, default=0.3, dest="id_margin", help="이 코사인 이상만 벌점")
    ap.add_argument("--aug", action="store_true", help="입력 도메인 랜덤화 on")
    ap.add_argument("--sample-every", type=int, default=1000, dest="sample_every")
    ap.add_argument("--ckpt-every", type=int, default=5000, dest="ckpt_every")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="랜덤텐서 1스텝 검증")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G, D, vgg = Generator().to(dev), Discriminator().to(dev), VGG().to(dev)
    idm = load_id(dev) if args.id_loss > 0 else None
    optG = Adam(G.parameters(), args.lr_g, betas=(0.5, 0.999))
    optD = Adam(D.parameters(), args.lr_d, betas=(0.5, 0.999))

    def g_losses(p, s):
        fake = G(p)
        dfake = D(fake)
        adv = F.mse_loss(dfake, torch.ones_like(dfake))
        con = F.l1_loss(vgg(fake), vgg(p))                   # conv4_4 content(구조·표정 보존)
        gs = gram(vgg(gray3(s)))                             # 회색 gram(conv4_4)
        sty = F.l1_loss(gram(vgg(gray3(fake))), gs) / (gs.abs().mean() + 1e-8)  # 상대오차, O(1)
        col = color_loss(fake, p)
        tv = tv_loss(fake)
        g = args.w_adv * adv + args.w_con * con + args.w_sty * sty + args.w_col * col + args.w_tv * tv
        idl = torch.tensor(0.0, device=dev)
        if idm is not None:
            cos = (id_embed(idm, p) * id_embed(idm, fake)).sum(1)
            idl = F.relu(cos - args.id_margin).mean()
            g = g + args.id_loss * idl
        return fake, g, dict(adv=adv.item(), con=con.item(), sty=sty.detach().item(),
                             col=col.item(), tv=tv.item(), idl=float(idl))

    def d_loss(p, s):
        fake = G(p).detach()
        dr = D(s)                                  # 진짜 애니(컬러) → 1
        df = D(fake)                               # 생성물 → 0
        dg = D(gray3(s))                           # 회색 애니 → 0 (색 입히도록 강제)
        de = D(edge_smooth(gray3(s)))              # 엣지뭉갠 회색 → 0 (선명하게 강제)
        return (F.mse_loss(dr, torch.ones_like(dr))
                + F.mse_loss(df, torch.zeros_like(df))
                + F.mse_loss(dg, torch.zeros_like(dg))
                + 0.1 * F.mse_loss(de, torch.zeros_like(de)))

    if args.smoke:
        print(f"[smoke] dev={dev} size={args.size}")
        p = torch.rand(2, 3, args.size, args.size, device=dev) * 2 - 1
        s = torch.rand(2, 3, args.size, args.size, device=dev) * 2 - 1
        dl = d_loss(p, s); optD.zero_grad(); dl.backward(); optD.step()
        fake, gl, parts = g_losses(p, s); optG.zero_grad(); gl.backward(); optG.step()
        print(f"[smoke] out={tuple(fake.shape)}  D_patch={tuple(D(p).shape)}")
        print(f"[smoke] d_loss={dl.item():.3f}  g_loss={gl.item():.3f}  {parts}")
        ng = sum(x.numel() for x in G.parameters())
        print(f"[smoke] G params={ng/1e6:.2f}M  → 배선 OK")
        return

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    photo = cycle(DataLoader(Imgs(args.photo, args.size, args.aug), args.batch,
                             shuffle=True, num_workers=args.workers, drop_last=True))
    style = cycle(DataLoader(Imgs(args.style, args.size), args.batch,
                             shuffle=True, num_workers=args.workers, drop_last=True))
    print(f"[data] photo={len(Imgs(args.photo, args.size).paths)}  style={len(Imgs(args.style, args.size).paths)}")

    # 워밍업: content-only (G가 입력 구조를 재현하도록) — lr 높게(1e-4)
    for grp in optG.param_groups:
        grp["lr"] = args.init_lr
    for step in range(1, args.init_steps + 1):
        p = next(photo).to(dev)
        con = F.l1_loss(vgg(G(p)), vgg(p))
        optG.zero_grad(); (args.w_con * con).backward(); optG.step()
        if step % 200 == 0:
            print(f"[init {step}/{args.init_steps}] con={con.item():.3f}")
    for grp in optG.param_groups:                # 본 학습 lr로 복귀(2e-5)
        grp["lr"] = args.lr_g

    # 본 학습
    for step in range(1, args.steps + 1):
        p, s = next(photo).to(dev), next(style).to(dev)
        dl = d_loss(p, s); optD.zero_grad(); dl.backward(); optD.step()
        fake, gl, parts = g_losses(p, s); optG.zero_grad(); gl.backward(); optG.step()
        if step % 100 == 0:
            print(f"[{step}/{args.steps}] D={dl.item():.3f} G={gl.item():.3f} "
                  f"adv={parts['adv']:.2f} con={parts['con']:.2f} sty={parts['sty']:.3f} "
                  f"col={parts['col']:.2f} id={parts['idl']:.3f}")
        if step % args.sample_every == 0:
            grid = torch.cat([p[:4], fake[:4].clamp(-1, 1)], 0) * 0.5 + 0.5
            save_image(grid, os.path.join(args.out, "samples", f"s{step:06d}.png"), nrow=4)
        if step % args.ckpt_every == 0:
            torch.save({"G": G.state_dict(), "step": step},
                       os.path.join(args.out, f"student_{step:06d}.pt"))
    torch.save({"G": G.state_dict(), "step": args.steps}, os.path.join(args.out, "student_final.pt"))
    print(f"완료 → {args.out}/student_final.pt")


if __name__ == "__main__":
    main()
