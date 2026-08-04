#!/usr/bin/env python3
"""[진단] "선은 또렷한가 / 면은 평평한가"를 학생과 teacher에서 따로 잰다.

■ 왜 Laplacian 을 쓰면 안 되는가 (2026-08-04, 두 번째 재발)
  Laplacian 분산은 **굵고 깔끔한 외곽선**과 **자잘한 주름**을 구분하지 못한다.
  주름 많은 노인 사진에서 teacher 가 주름을 전부 선으로 그리면 값이 치솟는데,
  그걸 학생이 단순화하면 "재현 실패"로 읽힌다. 그러나 카툰화 목적에서 주름 단순화는
  실패가 아니라 바람직한 동작이고, 주름 패턴은 신원 단서라 비식별화에는 오히려 이롭다.
  이 프로젝트는 2026-07-29 에도 같은 이유로 Laplacian 을 폐기하고
  "PNG 크기 + 내부 평탄도"로 갈아탄 전례가 있다. 같은 함정에 두 번 빠지지 않는다.

■ 대신 재는 것 — 카툰다움은 두 성분의 곱이다
  edge_contrast  엣지 위치에서의 평균 기울기 세기. **선이 얼마나 또렷한가.**
                 (선의 개수가 아니라 세기라서 주름 수에 덜 흔들린다)
  flatness       엣지가 아닌 영역의 국소 표준편차. **낮을수록 색면이 평평하다.**
  edge_density   Canny 엣지 픽셀 비율. 선의 **양**. 낮다고 나쁜 게 아니다(단순화).
  png_bytes      무손실 압축 크기. 전체 정보량의 대리 지표.

■ 읽는 법
  선이 무디다        → edge_contrast 비율이 1.0 보다 뚜렷이 낮다
  면이 덜 평평하다   → flatness 비율이 1.0 보다 높다 (학생이 더 울퉁불퉁)
  주름만 덜 그렸다   → edge_density 만 낮고 edge_contrast 는 유지 → **문제 아님**

사용:
  python3 run/style_sharpness.py --ckpt train/localface_occ65_deep8/student_final.pt \
      --data out/pairs_anime12_13500 \
      --localize-manifest out/localface_idx_occ65/manifest.jsonl --n 64
"""
import argparse, io, os, sys
import numpy as np
import cv2
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_student import (build_generator, checkpoint_generator_kwargs, PairImgs)


def metrics(bgr, canny_lo=60, canny_hi=140, flat_win=5):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                         cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    edges = cv2.Canny(gray, canny_lo, canny_hi) > 0
    edge_band = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    # 선의 세기: 엣지 위에서만 본다. 선이 적어도 굵고 진하면 높게 나온다.
    contrast = float(grad[edge_band].mean()) if edge_band.any() else 0.0

    # 면의 평탄도: 엣지에서 충분히 떨어진 곳의 국소 표준편차
    interior = cv2.erode((~edge_band).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    g32 = gray.astype(np.float32)
    mean = cv2.blur(g32, (flat_win, flat_win))
    var = cv2.blur(g32 * g32, (flat_win, flat_win)) - mean * mean
    std = np.sqrt(np.clip(var, 0, None))
    flat = float(std[interior].mean()) if interior.any() else 0.0

    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(buf, format="PNG", optimize=False)
    return dict(edge_contrast=contrast, flatness=flat,
                edge_density=float(edges.mean()) * 100.0, png_kb=buf.tell() / 1024.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--localize-manifest", default=None, dest="localize_manifest")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--split", choices=("val", "train", "all"), default="val")
    ap.add_argument("--worst", type=int, default=6, help="edge_contrast 비율이 낮은 표본 N개 출력")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    weights = ck["G"] if isinstance(ck, dict) and "G" in ck else ck
    kwargs = checkpoint_generator_kwargs(ck, weights)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = build_generator(**kwargs).to(dev).eval()
    G.load_state_dict(weights)
    print(f"[model] {kwargs}")

    ds = PairImgs(args.data, args.size, aug_level=0, localize_manifest=args.localize_manifest)
    stem_to_index = {p.stem: i for i, p in enumerate(ds.pairs)}
    split = (ck.get("split") or {}) if isinstance(ck, dict) else {}
    stems = split.get(args.split) if args.split != "all" else [p.stem for p in ds.pairs]
    if not stems:
        stems = [p.stem for p in ds.pairs]
        print(f"[warn] 체크포인트에 {args.split} split 이 없어 전체에서 뽑는다")
    stems = [s for s in stems if s in stem_to_index][:args.n]
    print(f"[data] {args.split} {len(stems)}장 측정\n")

    to_bgr = lambda t: ((t.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)[:, :, ::-1]
    rows = []
    with torch.no_grad():
        for s in stems:
            a, b = ds[stem_to_index[s]]
            fake = G(a.unsqueeze(0).to(dev))[0]
            rows.append((s, metrics(np.ascontiguousarray(to_bgr(fake))),
                            metrics(np.ascontiguousarray(to_bgr(b)))))

    keys = ("edge_contrast", "flatness", "edge_density", "png_kb")
    print(f"{'지표':<16}{'학생':>10}{'teacher':>10}{'비율':>9}   해석")
    print("-" * 68)
    note = {
        "edge_contrast": "높을수록 선이 또렷 (비율 <1 이면 학생이 무디다)",
        "flatness":      "낮을수록 면이 평평 (비율 >1 이면 학생이 울퉁불퉁)",
        "edge_density":  "선의 양. 비율 <1 은 단순화 — 문제 아닐 수 있음",
        "png_kb":        "전체 정보량 대리 지표",
    }
    for k in keys:
        st = np.mean([r[1][k] for r in rows]); te = np.mean([r[2][k] for r in rows])
        print(f"{k:<16}{st:>10.2f}{te:>10.2f}{st/max(te,1e-9):>9.2f}   {note[k]}")

    ratios = np.array([r[1]["edge_contrast"] / max(r[2]["edge_contrast"], 1e-9) for r in rows])
    dens = np.array([r[2]["edge_density"] for r in rows])
    print(f"\ncorr(teacher 선 밀도, 선 세기 재현율) = {np.corrcoef(dens, ratios)[0,1]:+.3f}")
    print("  음수가 크면 '선이 많은 그림일수록 학생이 무뎌진다' = 대역폭 한계")
    print("  0 근처면 선 세기는 유지되고 선의 양만 줄어든 것 = 단순화, 문제 아님")

    order = np.argsort(ratios)[:args.worst]
    print(f"\n=== 선 세기 재현율 최저 {len(order)}장 ===")
    for i in order:
        s, stu, tea = rows[i]
        print(f"  {s:<28} 세기 {stu['edge_contrast']:6.1f}/{tea['edge_contrast']:6.1f}={ratios[i]:.2f}"
              f"   평탄 {stu['flatness']:5.2f}/{tea['flatness']:5.2f}"
              f"   밀도 {stu['edge_density']:5.2f}/{tea['edge_density']:5.2f}")


if __name__ == "__main__":
    main()
