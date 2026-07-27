#!/usr/bin/env python3
"""
④ 학생 학습 — AnimeGANv2 (unpaired) 화풍 학습 + (옵션) 신원 억제.

증류 파이프라인 ④단계 (B: 엔드투엔드 노선).
- 정렬 페어 X.  photo 코퍼스(SFHQ 실사) + style 코퍼스(LoRA-Chroma로 뽑은 2.5D 애니) unpaired.
- 역할 분담:
    구조·표정 보존 = content loss (VGG19 conv4_4)   ← 얼굴 구조 앵커
    화풍         = GAN(adv) + grayscale gram style + color 복원
    신원 제거     = identity-suppression (얼굴 임베딩 코사인 억제, 옵션)

⚠️ 근거 기반 개정(2026-07-27):
  진짜 뿌리 = "워밍업이 회색 상수로 붕괴"였다. VGG conv4_4 단독 content는 픽셀 재현 신호가
  약해서, 우리 제너레이터(animegan2, 64x64 병목)가 실제 얼굴 대신 평균색 하나로 도망감.
  그 위에서 adv를 돌리니 D가 "회색 vs 진짜애니"를 너무 쉽게 이겨(D→0.04, GAN 수렴실패의 전형)
  색 덩어리만 얹혔다. 즉 adv 300이 얼굴을 민 게 아니라, 얼굴이 처음부터 없었다.
  근거: ptran1203/pytorch-animeGAN(최다사용 구현), TachibanaYoshino/AnimeGAN, ML-Mastery GAN 실패모드.
  → 수정:
    (1) 워밍업을 픽셀 L1 + VGG 재현으로  → 얼굴을 실제로 그리게(붕괴 원천 차단) ★핵심
    (2) 검증된 레시피값 그대로: adv 300(램프로 부드럽게 진입) / con 1.5 / gram 3 / color 30 / lr 2e-5·4e-5
    (3) 고정 평가셋 4장 매 샘플 + 워밍업 끝 샘플            → s000000이 얼굴이면 토대 정상
    (4) --w-con-px : 재붕괴 대비 픽셀 앵커(기본 0, 만약 재붕괴 보이면 켬)

라이선스 메모(중요):
- 본 파일 = 우리 소유. Generator 구조는 bryandlee/animegan2-pytorch(MIT) 재구현·귀속.
- VGG19(perceptual) : torchvision ImageNet 가중치 — **학습 전용, 런타임 미포함**.
- 신원 임베더(facenet) : **학습 전용, 런타임 미포함**. --id-loss 0 이면 끔.

사용:
  # (A) 스모크 — 랜덤텐서 1스텝, 아키텍처/손실 배선 확인
  python train/train_student.py --smoke
  # (B) 실제 학습 (화풍 학습, 신원손실 끔)
  python train/train_student.py \
    --photo input/sfhq_t2i/a_small_sample_new \
    --style out/pairs_dataset/target \
    --out train/student_v6 --size 256 --batch 4 \
    --steps 40000 --init-steps 3000 --id-loss 0
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
    ap.add_argument("--lr-g", type=float, default=2e-5, dest="lr_g", help="G lr")
    ap.add_argument("--lr-d", type=float, default=4e-5, dest="lr_d", help="D lr (레시피값, G의 2배)")
    ap.add_argument("--init-lr", type=float, default=1e-4, dest="init_lr", help="워밍업 lr")
    ap.add_argument("--init-steps", type=int, default=3000, dest="init_steps", help="content-only 워밍업")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--w-adv", type=float, default=300.0, dest="w_adv", help="adv 목표(AnimeGAN 레시피 300, 램프됨)")
    ap.add_argument("--adv-ramp", type=int, default=3000, dest="adv_ramp", help="adv 0→목표 램프 스텝수(부드러운 진입)")
    ap.add_argument("--w-con", type=float, default=1.5, dest="w_con", help="VGG conv4_4 content(레시피 1.5)")
    ap.add_argument("--w-con-px", type=float, default=0.0, dest="w_con_px", help="픽셀 L1 앵커(재붕괴 대비, 기본 off)")
    ap.add_argument("--w-sty", type=float, default=3.0, dest="w_sty", help="gram style(레시피 3)")
    ap.add_argument("--w-col", type=float, default=30.0, dest="w_col", help="color 복원(레시피 30, 색 고정)")
    ap.add_argument("--w-tv", type=float, default=1.0, dest="w_tv")
    ap.add_argument("--id-loss", type=float, default=0.0, dest="id_loss", help="신원억제 가중(0=off)")
    ap.add_argument("--id-margin", type=float, default=0.3, dest="id_margin", help="이 코사인 이상만 벌점")
    ap.add_argument("--aug", action="store_true", help="입력 도메인 랜덤화 on")
    ap.add_argument("--sample-every", type=int, default=500, dest="sample_every")
    ap.add_argument("--ckpt-every", type=int, default=5000, dest="ckpt_every")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="랜덤텐서 1스텝 검증")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G, D, vgg = Generator().to(dev), Discriminator().to(dev), VGG().to(dev)
    idm = load_id(dev) if args.id_loss > 0 else None
    optG = Adam(G.parameters(), args.lr_g, betas=(0.5, 0.999))
    optD = Adam(D.parameters(), args.lr_d, betas=(0.5, 0.999))

    def g_losses(p, s, w_adv_eff):
        fake = G(p)
        dfake = D(fake)
        adv = F.mse_loss(dfake, torch.ones_like(dfake))
        con = F.l1_loss(vgg(fake), vgg(p))                   # conv4_4 content(의미적 구조)
        con_px = F.l1_loss(fake, p)                          # 픽셀 L1(기하 앵커, 상수/덩어리 붕괴 방지)
        gs = gram(vgg(gray3(s)))                             # 회색 gram(conv4_4)
        sty = F.l1_loss(gram(vgg(gray3(fake))), gs) / (gs.abs().mean() + 1e-8)  # 상대오차, O(1)
        col = color_loss(fake, p)
        tv = tv_loss(fake)
        g = (w_adv_eff * adv + args.w_con * con + args.w_con_px * con_px
             + args.w_sty * sty + args.w_col * col + args.w_tv * tv)
        idl = torch.tensor(0.0, device=dev)
        if idm is not None:
            cos = (id_embed(idm, p) * id_embed(idm, fake)).sum(1)
            idl = F.relu(cos - args.id_margin).mean()
            g = g + args.id_loss * idl
        return fake, g, dict(adv=adv.item(), con=con.item(), con_px=con_px.item(), sty=sty.detach().item(),
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
        fake, gl, parts = g_losses(p, s, args.w_adv); optG.zero_grad(); gl.backward(); optG.step()
        print(f"[smoke] out={tuple(fake.shape)}  D_patch={tuple(D(p).shape)}")
        print(f"[smoke] d_loss={dl.item():.3f}  g_loss={gl.item():.3f}  {parts}")
        ng = sum(x.numel() for x in G.parameters())
        print(f"[smoke] G params={ng/1e6:.2f}M  → 배선 OK")
        return

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    photo_ds = Imgs(args.photo, args.size, args.aug)
    style_ds = Imgs(args.style, args.size)
    photo = cycle(DataLoader(photo_ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True))
    style = cycle(DataLoader(style_ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True))
    print(f"[data] photo={len(photo_ds.paths)}  style={len(style_ds.paths)}")

    # 고정 평가셋 — 매번 같은 얼굴 4장으로 샘플(진행 추적용)
    eval_ds = Imgs(args.photo, args.size)
    n_eval = min(4, len(eval_ds))
    eval_p = torch.stack([eval_ds[i] for i in range(n_eval)]).to(dev)

    def save_eval(step):
        G.eval()
        with torch.no_grad():
            fake = G(eval_p).clamp(-1, 1)
        grid = torch.cat([eval_p, fake], 0) * 0.5 + 0.5
        save_image(grid, os.path.join(args.out, "samples", f"s{step:06d}.png"), nrow=n_eval)
        G.train()

    # 워밍업: 재현(reconstruction) — 픽셀 L1 주도 + VGG 보조. G가 입력 얼굴을 실제로 그리게.
    #  (VGG conv4_4 단독은 신호가 약해 평균색으로 붕괴함 → 픽셀 L1로 강제 재현)
    for grp in optG.param_groups:
        grp["lr"] = args.init_lr
    for step in range(1, args.init_steps + 1):
        p = next(photo).to(dev)
        out = G(p)
        recon = F.l1_loss(out, p) + 0.5 * F.l1_loss(vgg(out), vgg(p))
        optG.zero_grad(); recon.backward(); optG.step()
        if step % 200 == 0:
            print(f"[init {step}/{args.init_steps}] recon={recon.item():.3f} (px+vgg, 내려가야 정상)")
    for grp in optG.param_groups:                # 본 학습 lr로 복귀
        grp["lr"] = args.lr_g
    save_eval(0)                                  # 워밍업 끝 샘플(구조 재현 확인 = s000000.png)
    print("[init] 워밍업 끝 → samples/s000000.png 로 구조 재현 확인 가능")

    # 본 학습 (adv 램프: 0 → w_adv 목표)
    for step in range(1, args.steps + 1):
        w_adv_eff = args.w_adv * min(1.0, step / max(1, args.adv_ramp))
        p, s = next(photo).to(dev), next(style).to(dev)
        dl = d_loss(p, s); optD.zero_grad(); dl.backward(); optD.step()
        fake, gl, parts = g_losses(p, s, w_adv_eff); optG.zero_grad(); gl.backward(); optG.step()
        if step % 100 == 0:
            print(f"[{step}/{args.steps}] wadv={w_adv_eff:.1f} D={dl.item():.3f} G={gl.item():.3f} "
                  f"adv={parts['adv']:.2f} con={parts['con']:.2f} cpx={parts['con_px']:.3f} "
                  f"sty={parts['sty']:.3f} col={parts['col']:.2f} id={parts['idl']:.3f}")
        if step % args.sample_every == 0:
            save_eval(step)
        if step % args.ckpt_every == 0:
            torch.save({"G": G.state_dict(), "step": step},
                       os.path.join(args.out, f"student_{step:06d}.pt"))
    torch.save({"G": G.state_dict(), "step": args.steps}, os.path.join(args.out, "student_final.pt"))
    print(f"완료 → {args.out}/student_final.pt")


if __name__ == "__main__":
    main()
