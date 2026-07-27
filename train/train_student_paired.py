#!/usr/bin/env python3
"""
④-B 학생 학습 (PAIRED / pix2pix 증류) — 정렬된 (사진→애니) 페어로 지도 회귀.

배경: unpaired AnimeGAN(v5~v9)이 얼굴 화풍 학습에서 under-fit 천장을 침
      (sty가 gram 13배·D 강화에도 0.8에서 안 내려감, 출력=부드러운 사진).
      → 시니어 정석(Diffusion2GAN/pix2pix-turbo/Parsing-Anime): 구조조건 정렬페어 + 지도회귀.

핵심 차이: unpaired는 "학생이 스스로 화풍을 알아내야" 했지만, paired는
  "이 사진의 정답 애니는 이거다"를 직접 보여주므로 학생은 베끼기만 하면 됨(안정·확실).

- 데이터: pairs/input/<name> (실사)  ↔  pairs/target/<name> (같은 이름 = 정렬쌍, 애니)
- 손실:  L1(fake,target) + VGG perceptual(fake,target) + 가벼운 adversarial(D: target=real)
         [+ optional 신원억제 --id-loss]
- Generator = animegan2 구조(런타임 셸 호환). 학습 결과 .pt를 런타임에 그대로 삽입.

사용:
  # 스모크(배선 확인)
  python train/train_student_paired.py --smoke
  # 실제 학습
  python train/train_student_paired.py \
    --pairs out/pairs_dataset_v2 --out train/student_paired \
    --size 256 --batch 4 --steps 40000
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


# ============ Generator (animegan2 구조 재구현, MIT 귀속 — 런타임 호환) ============
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


# ============ VGG19 perceptual (conv4_4) ============
class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            v = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        except Exception as e:
            print(f"[vgg] 가중치 다운로드 실패 → 랜덤init(스모크용): {e}")
            v = models.vgg19(weights=None)
        self.v = v.features[:27].eval()
        for p in self.v.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        return self.v(x)


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


# ============ 데이터 (PAIRED: input/<name> ↔ target/<name>) ============
class Pairs(Dataset):
    def __init__(self, root, size):
        self.di = os.path.join(root, "input")
        self.dt = os.path.join(root, "target")
        tnames = {os.path.basename(p) for p in glob.glob(self.dt + "/*") if p.lower().endswith(EXTS)}
        inames = {os.path.basename(p) for p in glob.glob(self.di + "/*") if p.lower().endswith(EXTS)}
        self.names = sorted(tnames & inames)          # 양쪽에 다 있는 정렬쌍만
        if not self.names:
            raise SystemExit(f"정렬 페어 없음: {root} (input/ 와 target/ 에 같은 이름 파일 필요)")
        self.tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        a = self.tf(Image.open(os.path.join(self.di, n)).convert("RGB")) * 2 - 1   # 실사 [-1,1]
        b = self.tf(Image.open(os.path.join(self.dt, n)).convert("RGB")) * 2 - 1   # 애니 [-1,1]
        return a, b


def cycle(dl):
    while True:
        for x in dl:
            yield x


# ============ 학습 ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="out/pairs_dataset_v2", help="input/ · target/ 를 담은 정렬페어 루트")
    ap.add_argument("--out", default="train/student_paired")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr-g", type=float, default=2e-4, dest="lr_g")
    ap.add_argument("--lr-d", type=float, default=2e-4, dest="lr_d")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--w-l1", type=float, default=10.0, dest="w_l1", help="타깃 픽셀 L1(주력)")
    ap.add_argument("--w-perc", type=float, default=10.0, dest="w_perc", help="VGG perceptual(구조·질감)")
    ap.add_argument("--w-adv", type=float, default=1.0, dest="w_adv", help="가벼운 GAN(선명도)")
    ap.add_argument("--gan-start", type=int, default=1000, dest="gan_start", help="이 스텝부터 GAN 켬(먼저 L1로 타깃 근사)")
    ap.add_argument("--id-loss", type=float, default=0.0, dest="id_loss", help="신원억제(0=off)")
    ap.add_argument("--id-margin", type=float, default=0.3, dest="id_margin")
    ap.add_argument("--sample-every", type=int, default=500, dest="sample_every")
    ap.add_argument("--ckpt-every", type=int, default=5000, dest="ckpt_every")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G, D, vgg = Generator().to(dev), Discriminator().to(dev), VGG().to(dev)
    idm = load_id(dev) if args.id_loss > 0 else None
    optG = Adam(G.parameters(), args.lr_g, betas=(0.5, 0.999))
    optD = Adam(D.parameters(), args.lr_d, betas=(0.5, 0.999))

    def g_losses(inp, tgt, use_gan):
        fake = G(inp)
        l1 = F.l1_loss(fake, tgt)                        # 타깃 애니에 픽셀 근사
        perc = F.l1_loss(vgg(fake), vgg(tgt))            # 타깃 애니에 특징 근사
        g = args.w_l1 * l1 + args.w_perc * perc
        adv = torch.tensor(0.0, device=dev)
        if use_gan:
            df = D(fake)
            adv = F.mse_loss(df, torch.ones_like(df))    # 타깃 분포로(선명도)
            g = g + args.w_adv * adv
        idl = torch.tensor(0.0, device=dev)
        if idm is not None:                              # 원본 실사와 신원 억제(비식별)
            cos = (id_embed(idm, inp) * id_embed(idm, fake)).sum(1)
            idl = F.relu(cos - args.id_margin).mean()
            g = g + args.id_loss * idl
        return fake, g, dict(l1=l1.item(), perc=perc.item(), adv=float(adv), idl=float(idl))

    def d_loss(inp, tgt):
        fake = G(inp).detach()
        dr = D(tgt); df = D(fake)
        return F.mse_loss(dr, torch.ones_like(dr)) + F.mse_loss(df, torch.zeros_like(df))

    if args.smoke:
        print(f"[smoke] dev={dev} size={args.size}")
        inp = torch.rand(2, 3, args.size, args.size, device=dev) * 2 - 1
        tgt = torch.rand(2, 3, args.size, args.size, device=dev) * 2 - 1
        dl = d_loss(inp, tgt); optD.zero_grad(); dl.backward(); optD.step()
        fake, gl, parts = g_losses(inp, tgt, True); optG.zero_grad(); gl.backward(); optG.step()
        print(f"[smoke] out={tuple(fake.shape)} d_loss={dl.item():.3f} g_loss={gl.item():.3f} {parts}")
        print(f"[smoke] G params={sum(x.numel() for x in G.parameters())/1e6:.2f}M → 배선 OK")
        return

    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    ds = Pairs(args.pairs, args.size)
    print(f"[data] 정렬 페어 {len(ds.names)}쌍  (from {args.pairs})")
    dl = cycle(DataLoader(ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True))

    # 고정 평가셋 4쌍 (input | fake | target 3행으로 저장 → 학생이 타깃에 근접하는지 추적)
    n_eval = min(4, len(ds))
    ei = torch.stack([ds[i][0] for i in range(n_eval)]).to(dev)
    et = torch.stack([ds[i][1] for i in range(n_eval)]).to(dev)

    def save_eval(step):
        G.eval()
        with torch.no_grad():
            fake = G(ei).clamp(-1, 1)
        grid = torch.cat([ei, fake, et], 0) * 0.5 + 0.5     # 윗줄 입력 / 가운데 학생출력 / 아랫줄 정답
        save_image(grid, os.path.join(args.out, "samples", f"s{step:06d}.png"), nrow=n_eval)
        G.train()

    for step in range(1, args.steps + 1):
        inp, tgt = next(dl)
        inp, tgt = inp.to(dev), tgt.to(dev)
        use_gan = step >= args.gan_start
        if use_gan:
            dl_ = d_loss(inp, tgt); optD.zero_grad(); dl_.backward(); optD.step()
        else:
            dl_ = torch.tensor(0.0)
        fake, gl, parts = g_losses(inp, tgt, use_gan); optG.zero_grad(); gl.backward(); optG.step()
        if step % 100 == 0:
            print(f"[{step}/{args.steps}] D={float(dl_):.3f} G={gl.item():.3f} "
                  f"l1={parts['l1']:.3f} perc={parts['perc']:.3f} adv={parts['adv']:.3f} "
                  f"id={parts['idl']:.3f} {'(GAN)' if use_gan else '(L1만)'}")
        if step % args.sample_every == 0:
            save_eval(step)
        if step % args.ckpt_every == 0:
            torch.save({"G": G.state_dict(), "step": step}, os.path.join(args.out, f"student_{step:06d}.pt"))
    torch.save({"G": G.state_dict(), "step": args.steps}, os.path.join(args.out, "student_final.pt"))
    print(f"완료 → {args.out}/student_final.pt")


if __name__ == "__main__":
    main()
