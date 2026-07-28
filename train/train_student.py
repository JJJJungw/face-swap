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


# ============ 데이터 (PAIRED: input↔target 같은 파일명) ============
class PairImgs(Dataset):
    def __init__(self, root, size, aug=False):
        din, dtg = os.path.join(root, "input"), os.path.join(root, "target")
        namesin = {os.path.basename(p) for p in glob.glob(os.path.join(din, "*")) if p.lower().endswith(EXTS)}
        namestg = {os.path.basename(p) for p in glob.glob(os.path.join(dtg, "*")) if p.lower().endswith(EXTS)}
        self.names = sorted(namesin & namestg)          # 공통(정렬된) 페어만
        if not self.names:
            raise SystemExit(f"페어 없음: {din} ∩ {dtg}")
        self.din, self.dtg, self.size, self.aug = din, dtg, size, aug
        self.jit = transforms.ColorJitter(0.15, 0.15, 0.15, 0.02)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        a = Image.open(os.path.join(self.din, n)).convert("RGB").resize((self.size, self.size), Image.LANCZOS)
        b = Image.open(os.path.join(self.dtg, n)).convert("RGB").resize((self.size, self.size), Image.LANCZOS)
        if self.aug:                                     # 광학적 aug은 input에만(정렬 유지)
            a = self.jit(a)
        ta = transforms.functional.to_tensor(a) * 2 - 1
        tb = transforms.functional.to_tensor(b) * 2 - 1
        return ta, tb


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
    ap.add_argument("--gen-ch", type=int, default=32, dest="gen_ch", help="제너레이터 기본채널(용량↑=32→48→64, 속도↓)")
    ap.add_argument("--d-ch", type=int, default=48, dest="d_ch")
    ap.add_argument("--d-n", type=int, default=3, dest="d_n")
    ap.add_argument("--aug", action="store_true", help="input 광학 aug(도메인 랜덤화)")
    ap.add_argument("--sample-every", type=int, default=500, dest="sample_every")
    ap.add_argument("--ckpt-every", type=int, default=5000, dest="ckpt_every")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator(args.gen_ch).to(dev)
    D = Discriminator(args.d_ch, args.d_n).to(dev)
    vgg = VGGPerceptual().to(dev)
    idm = load_id(dev) if args.id_loss > 0 else None
    optG = Adam(G.parameters(), args.lr, betas=(0.5, 0.999))
    optD = Adam(D.parameters(), args.lr, betas=(0.5, 0.999))

    def g_losses(inp, tgt, w_adv_eff):
        fake = G(inp)
        l1 = F.l1_loss(fake, tgt)                        # ★ target 직접 재현 → 유화 없이 2.5D 그대로
        perc = vgg(fake, tgt)                            # 다층 perceptual(디테일·선명)
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
    ds = PairImgs(args.data, args.size, args.aug)
    loader = cycle(DataLoader(ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True))
    print(f"[data] 페어 {len(ds)}쌍  (input↔target 정렬)")

    # 고정 평가셋 4쌍(진행 추적)
    ne = min(4, len(ds))
    ev = [ds[i] for i in range(ne)]
    ev_in = torch.stack([a for a, _ in ev]).to(dev)
    ev_tg = torch.stack([b for _, b in ev]).to(dev)

    def save_eval(step):
        G.eval()
        with torch.no_grad():
            fk = G(ev_in).clamp(-1, 1)
        grid = torch.cat([ev_in, fk, ev_tg], 0) * 0.5 + 0.5   # 입력 / 학생출력 / 정답target
        save_image(grid, os.path.join(args.out, "samples", f"s{step:06d}.png"), nrow=ne)
        G.train()

    for step in range(1, args.steps + 1):
        inp, tgt = next(loader)
        inp, tgt = inp.to(dev), tgt.to(dev)
        w_adv_eff = 0.0 if step <= args.init_steps else args.w_adv * min(1.0, (step - args.init_steps) / max(1, args.adv_ramp))
        if w_adv_eff > 0:
            dl = d_loss(inp, tgt); optD.zero_grad(); dl.backward(); optD.step()
        else:
            dl = torch.tensor(0.0)
        fake, gl, parts = g_losses(inp, tgt, w_adv_eff); optG.zero_grad(); gl.backward(); optG.step()
        if step % 100 == 0:
            print(f"[{step}/{args.steps}] wadv={w_adv_eff:.2f} D={float(dl):.3f} G={gl.item():.3f} "
                  f"l1={parts['l1']:.3f} perc={parts['perc']:.3f} adv={parts['adv']:.2f} id={parts['idl']:.3f}")
        if step % args.sample_every == 0:
            save_eval(step)
        if step % args.ckpt_every == 0:
            torch.save({"G": G.state_dict(), "step": step}, os.path.join(args.out, f"student_{step:06d}.pt"))
    torch.save({"G": G.state_dict(), "step": args.steps}, os.path.join(args.out, "student_final.pt"))
    print(f"완료 → {args.out}/student_final.pt")


if __name__ == "__main__":
    main()
