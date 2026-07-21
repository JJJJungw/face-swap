#!/usr/bin/env python3
"""animegan2 (MIT) 이미지 2.5D 스타일화 — 실시간급 런타임 후보
- 코드/가중치 없으면 자동 다운로드 (raw.githubusercontent, MIT, face_paint_512_v2)
- 크롭 없이 비율 유지 리사이즈
사용법:
  python run/animegan_stylize.py --image input/foo.jpg
  python run/animegan_stylize.py --image input/foo.jpg --size 768 --out out/foo_anime.jpg
"""
import os, sys, argparse, ssl, urllib.request, torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

CKPT_DIR = "gan_ckpt"
BASE = "https://raw.githubusercontent.com/bryandlee/animegan2-pytorch"
FILES = {"model.py": f"{BASE}/master/model.py",
         "w.pt": f"{BASE}/main/weights/face_paint_512_v2.pt"}

def ensure_ckpt():
    os.makedirs(CKPT_DIR, exist_ok=True)
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    for name, url in FILES.items():
        dst = os.path.join(CKPT_DIR, name)
        if not os.path.exists(dst):
            print("download", dst)
            with urllib.request.urlopen(url, context=ctx, timeout=90) as r, open(dst, "wb") as f:
                f.write(r.read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", type=int, default=768, help="긴 변 기준 리사이즈")
    ap.add_argument("--out", default="out/animegan_result.jpg")
    args = ap.parse_args()

    ensure_ckpt()
    sys.path.insert(0, CKPT_DIR)
    from model import Generator
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = Generator().to(dev).eval()
    m.load_state_dict(torch.load(os.path.join(CKPT_DIR, "w.pt"), map_location=dev))

    img = Image.open(args.image).convert("RGB")
    w, h = img.size
    sc = args.size / max(w, h)
    nw = max(64, int(round(w * sc / 8)) * 8); nh = max(64, int(round(h * sc / 8)) * 8)
    img = img.resize((nw, nh), Image.LANCZOS)
    x = (to_tensor(img).unsqueeze(0) * 2 - 1).to(dev)
    with torch.no_grad():
        y = (m(x)[0] * 0.5 + 0.5).clip(0, 1)
    to_pil_image(y.cpu()).save(args.out)
    print("saved", args.out, (nw, nh))

if __name__ == "__main__":
    main()
