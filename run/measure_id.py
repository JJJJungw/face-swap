#!/usr/bin/env python3
"""[③-d 재식별 측정] 페어 코퍼스의 신원 잔존도를 cos(input, target)으로 측정한다.

왜 필요한가:
  본 모듈의 하드 요구사항은 "예쁜 카툰"이 아니라 **비식별화**다.
  그런데 지금까지 화풍(Laplacian·CV)만 재고 신원은 한 번도 재지 않았다.
  README의 StyleID 0.744는 외부 조사 수치이고 우리 코퍼스 값이 아니다.

무엇을 재는가:
  facenet(vggface2) 임베딩의 코사인 유사도. **train_student.py의 id_embed()와 동일한 전처리**
  (160px 리사이즈 + L2 정규화)를 쓴다 → 여기서 나온 값이 곧 학습 중 id-loss가 보는 값이다.

읽는 법:
  cos ≈ 1.0   완전히 같은 사람 (비식별 실패)
  cos > 0.5   대체로 동일인으로 판정될 수준
  cos < 0.3   train_student.py 의 --id-margin 기본값. 이 아래면 id-loss가 더 밀 필요 없음
  cos < 0.0   임베딩 공간에서 무관

중요:
  teacher가 신원을 남겨도 **학생 쪽 id-loss로 밀어낼 수 있다**(별개 노브).
  따라서 이 수치는 teacher 탈락 기준이 아니라 **id-loss를 얼마나 세게 걸어야 하는지**의 지표다.
  단, id-loss는 content-loss와 싸우므로 시작점이 높을수록 표정·구조가 깨질 위험이 커진다.

사용:
  python3 run/measure_id.py --dir out/pairs_2511
  # 여러 코퍼스 비교
  python3 run/measure_id.py --dir out/pairs_2511 --dir out/pairs_fp3 --n 100
"""
import argparse, os, glob
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
except ImportError as e:
    raise SystemExit(f"torch/torchvision/PIL 필요: {e}")


def load_embedder(device):
    try:
        from facenet_pytorch import InceptionResnetV1
    except ImportError:
        raise SystemExit(
            "facenet-pytorch 없음 → pip install facenet-pytorch\n"
            "  (train_student.py 의 --id-loss 도 이 패키지를 쓴다)")
    m = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def embed(m, x):
    """train_student.py 의 id_embed() 와 동일 — 값이 학습 중 id-loss와 일치해야 한다."""
    x = F.interpolate(x, 160, mode="bilinear", align_corners=False)
    return F.normalize(m(x), dim=1)


def load(p, size=256):
    im = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
    return transforms.functional.to_tensor(im) * 2 - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True, help="페어 폴더(반복 지정 가능)")
    ap.add_argument("--n", type=int, default=0, help="폴더당 표본 수 (0=전부)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--margin", type=float, default=0.3, help="train_student.py --id-margin 과 맞출 것")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[id] device={dev}")
    m = load_embedder(dev)

    results = {}
    for d in args.dir:
        din, dtg = os.path.join(d, "input"), os.path.join(d, "target")
        names = sorted(set(os.path.basename(p) for p in glob.glob(f"{din}/*.png"))
                       & set(os.path.basename(p) for p in glob.glob(f"{dtg}/*.png")))
        if not names:
            print(f"  [건너뜀] 페어 없음: {d}")
            continue
        if args.n > 0 and len(names) > args.n:      # 앞쪽만 쓰면 편향 → 균등 간격
            names = names[::max(1, len(names) // args.n)][:args.n]

        cos = []
        for i in range(0, len(names), args.batch):
            chunk = names[i:i + args.batch]
            a = torch.stack([load(os.path.join(din, n)) for n in chunk]).to(dev)
            b = torch.stack([load(os.path.join(dtg, n)) for n in chunk]).to(dev)
            with torch.no_grad():
                c = (embed(m, a) * embed(m, b)).sum(1)
            cos += c.cpu().tolist()
        results[d] = np.array(cos)
        print(f"  {d}: {len(cos)}쌍")

    print(f"\n=== 신원 잔존도 cos(input, target) — 낮을수록 비식별 성공 ===")
    print(f"{'코퍼스':<26}{'평균':>8}{'중앙값':>8}{'최소':>8}{'최대':>8}{'>0.5':>8}{'>margin':>9}")
    for d, v in results.items():
        over_m = (v > args.margin).mean() * 100
        print(f"{os.path.basename(d.rstrip('/')):<26}{v.mean():8.3f}{np.median(v):8.3f}"
              f"{v.min():8.3f}{v.max():8.3f}{(v>0.5).mean()*100:7.0f}%{over_m:8.0f}%")

    print(f"\n해석 (margin={args.margin}):")
    for d, v in results.items():
        n = os.path.basename(d.rstrip("/"))
        med = float(np.median(v))
        if med > 0.6:
            s = "스타일화만으로는 비식별 실패 → id-loss 필수, 가중치 높게 필요"
        elif med > 0.35:
            s = "부분 비식별 → id-loss 중간 가중치로 margin 아래까지 밀 수 있음"
        elif med > args.margin:
            s = "margin 근처 → id-loss 약하게만 걸어도 충분"
        else:
            s = "이미 margin 이하 → id-loss 없이도 비식별 조건 충족"
        print(f"  {n:<24} 중앙값 {med:.3f} — {s}")

    print("\n※ 주의: 이 값은 teacher target 기준이다. 최종 판정은 **학생 출력** 기준으로 다시 재야 한다")
    print("   (학생은 target을 완벽히 재현하지 못하므로 값이 달라진다).")


if __name__ == "__main__":
    main()
