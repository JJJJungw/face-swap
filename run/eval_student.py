#!/usr/bin/env python3
"""[⑤ 학생 평가] 학습된 animeganv2 학생을 3축으로 한 번에 평가한다.

왜 이 3개를 같이 재는가:
  세 요구사항이 서로 싸우기 때문이다. 하나만 보면 반드시 다른 하나를 망친다.

    ① 비식별화   cos(input, 학생출력)      ↓ 낮아야 함  ← id-loss가 밀어냄
    ② 화풍 재현   학생출력 vs teacher target ↑ 높아야 함  ← L1/perceptual이 당김
    ③ 속도       ms/face                   ↓ 낮아야 함  ← 하드 요구사항 ≤2×

  id-loss를 올리면 ①은 좋아지지만 ②(표정·구조 포함)가 깨진다.
  따라서 id-loss 가중치는 "①이 목표선을 넘는 최소값"으로 잡아야 하고,
  그 지점을 찾으려면 ①과 ②를 **동시에** 봐야 한다.

  ※ teacher target의 신원 점수(run/measure_id.py)는 '출발점' 견적일 뿐이다.
     런타임에 도는 것은 학생이므로 **제품 판정은 이 스크립트의 값**으로 한다.

사용:
  # 단일 체크포인트
  python3 run/eval_student.py --ckpt train/student_2511/student_final.pt \
      --data out/pairs_2511 --n 64

  # id-loss 스윕 비교 (여러 학습 결과를 한 표로)
  python3 run/eval_student.py --data out/pairs_2511 --n 64 \
      --ckpt train/student_id00/student_final.pt \
      --ckpt train/student_id05/student_final.pt \
      --ckpt train/student_id20/student_final.pt
"""
import argparse, os, sys, time
import numpy as np
from pair_utils import discover_pairs

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
except ImportError as e:
    raise SystemExit(f"torch/torchvision/PIL 필요: {e}")

# Generator를 재정의하지 않고 학습 스크립트에서 그대로 가져온다.
# (구조가 어긋나면 state_dict 로드가 조용히 실패하거나 다른 모델을 평가하게 된다)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    from train_student import build_generator, checkpoint_generator_kwargs
except ImportError:
    try:
        from train.train_student import build_generator, checkpoint_generator_kwargs
    except ImportError as e:
        raise SystemExit(f"train_student.py의 Generator를 못 찾음: {e}\n"
                         "  레포 루트에서 실행할 것: python3 run/eval_student.py ...")


def load_img(p, size):
    im = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
    return transforms.functional.to_tensor(im) * 2 - 1


def load_id(device):
    try:
        from facenet_pytorch import InceptionResnetV1
    except ImportError:
        print("[경고] facenet-pytorch 없음 → 비식별화 측정 건너뜀 (pip install facenet-pytorch)")
        return None
    m = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def id_embed(m, x):
    """train_student.py 의 id_embed()와 동일 — 학습 중 id-loss가 본 값과 일치시켜야 비교가 성립"""
    x = F.interpolate(x, 160, mode="bilinear", align_corners=False)
    return F.normalize(m(x), dim=1)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="student_*.pt (반복 지정 가능)")
    ap.add_argument("--data", required=True, help="input/ 와 target/ 을 가진 페어 폴더")
    ap.add_argument("--include-file", default=None, dest="include_file",
                    help="평가할 stem 목록(한 줄에 하나). train/validation 고정 평가용")
    ap.add_argument("--n", type=int, default=64, help="평가 표본 수")
    ap.add_argument("--size", type=int, default=512, help="학습 때 쓴 --size 와 동일해야 함")
    ap.add_argument("--gen-ch", type=int, default=None, dest="gen_ch",
                    help="생략하면 체크포인트에서 자동 감지")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sheet", default="out/eval_student.png")
    ap.add_argument("--bench", type=int, default=50, help="속도 측정 반복 횟수(0=건너뜀)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    din, dtg = os.path.join(args.data, "input"), os.path.join(args.data, "target")
    pairs = discover_pairs(din, dtg)
    if not pairs:
        raise SystemExit(f"페어 없음: {args.data}")
    if args.include_file:
        with open(args.include_file, encoding="utf-8") as handle:
            stems = [
                line.strip() for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
        pair_by_stem = {pair.stem: pair for pair in pairs}
        missing = [stem for stem in stems if stem not in pair_by_stem]
        if missing:
            raise SystemExit(
                f"--include-file의 {len(missing)}개 stem을 찾지 못함: {missing[:10]}"
            )
        pairs = [pair_by_stem[stem] for stem in stems]
    if len(pairs) > args.n:                      # 앞쪽만 쓰면 편향 → 균등 간격
        pairs = pairs[::max(1, len(pairs) // args.n)][:args.n]
    print(f"[eval] device={dev}  표본 {len(pairs)}쌍  size={args.size} include={args.include_file}")

    idm = load_id(dev)
    rows = {}
    outs = {}

    for ck in args.ckpt:
        tag = os.path.basename(os.path.dirname(ck)) or os.path.basename(ck)
        try:
            sd = torch.load(ck, map_location=dev, weights_only=False)
        except TypeError:
            sd = torch.load(ck, map_location=dev)
        weights = sd["G"] if "G" in sd else sd
        detected_ch = int(weights["in_conv.1.weight"].shape[0])
        if args.gen_ch is not None and args.gen_ch != detected_ch:
            raise SystemExit(
                f"--gen-ch {args.gen_ch} != checkpoint channel count {detected_ch}: {ck}"
            )
        gen_kwargs = checkpoint_generator_kwargs(sd, weights)
        G = build_generator(**gen_kwargs).to(dev).eval()
        G.load_state_dict(weights, strict=True)
        print(f"\n[{tag}] {ck}  (step {sd.get('step','?')})")
        print(f"[model] gen_arch={detected_arch} gen_ch={detected_ch}")

        cos, l1, perc_proxy = [], [], []
        saved = []
        for i in range(0, len(pairs), args.batch):
            ch = pairs[i:i + args.batch]
            a = torch.stack([load_img(pair.input_path, args.size) for pair in ch]).to(dev)
            t = torch.stack([load_img(pair.target_path, args.size) for pair in ch]).to(dev)
            with torch.no_grad():
                f = G(a).clamp(-1, 1)
                if idm is not None:
                    cos += (id_embed(idm, a) * id_embed(idm, f)).sum(1).cpu().tolist()
                l1 += F.l1_loss(f, t, reduction="none").mean((1, 2, 3)).cpu().tolist()
                # 화풍 근접도 프록시: 다운샘플 후 L1(구조 무시, 색·톤 분포 위주)
                fa = F.avg_pool2d(f, 16); ta = F.avg_pool2d(t, 16)
                perc_proxy += F.l1_loss(fa, ta, reduction="none").mean((1, 2, 3)).cpu().tolist()
            if len(saved) < 6:
                saved += [(a[j].cpu(), f[j].cpu(), t[j].cpu()) for j in range(min(len(ch), 6 - len(saved)))]
        outs[tag] = saved
        rows[tag] = dict(cos=np.array(cos) if cos else None,
                         l1=np.array(l1), tone=np.array(perc_proxy))

        if args.bench:
            x = torch.randn(1, 3, args.size, args.size, device=dev)
            with torch.no_grad():
                for _ in range(5):
                    G(x)
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(args.bench):
                    G(x)
                if dev == "cuda":
                    torch.cuda.synchronize()
            rows[tag]["ms"] = (time.time() - t0) / args.bench * 1000
        else:
            rows[tag]["ms"] = float("nan")

    # ── 결과표 ────────────────────────────────────────────────────────────
    print("\n=== 학생 평가 ===")
    print(f"{'체크포인트':<22}{'신원cos↓':>10}{'>0.3':>7}{'화풍L1↓':>9}{'톤L1↓':>8}{'ms/face':>9}")
    for tag, r in rows.items():
        c = f"{r['cos'].mean():10.3f}" if r["cos"] is not None else f"{'-':>10}"
        o = f"{(r['cos']>0.3).mean()*100:6.0f}%" if r["cos"] is not None else f"{'-':>7}"
        print(f"{tag:<22}{c}{o}{r['l1'].mean():9.4f}{r['tone'].mean():8.4f}{r['ms']:9.1f}")

    print("\n읽는 법:")
    print("  신원cos  낮을수록 비식별 성공. 0.3(=--id-margin) 아래가 목표")
    print("  화풍L1   낮을수록 teacher 화풍을 잘 재현. 높으면 뭉갠 것")
    print("  → 이 둘은 서로 반대로 움직인다. id-loss를 올리면 신원cos↓ 화풍L1↑")
    print("     '신원cos가 0.3 아래로 내려가는 지점 중 화풍L1이 가장 낮은' 설정이 최적")
    if not np.isnan(list(rows.values())[0]["ms"]):
        print(f"  ms/face  512 기준. 영상 1분(30fps, 얼굴 1개)이면 ×1800 = 총 GAN 시간")
        print("           ※ eager PyTorch 값이다. 실제 런타임은 ONNX→TensorRT로 ~6.8배 빨라진다")

    # ── 비교 시트 (input | 학생 | target) ─────────────────────────────────
    try:
        from torchvision.utils import save_image
        os.makedirs(os.path.dirname(args.sheet) or ".", exist_ok=True)
        cols = []
        for tag, saved in outs.items():
            for a, f, t in saved:
                cols += [a, f, t]
        if cols:
            save_image((torch.stack(cols) + 1) / 2, args.sheet, nrow=3)
            print(f"\n비교 시트 → {args.sheet}   (열 순서: input | 학생 | teacher target)")
    except Exception as e:
        print(f"[경고] 시트 저장 실패: {e}")


if __name__ == "__main__":
    main()
