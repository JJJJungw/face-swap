#!/usr/bin/env python3
"""[③-e 피부톤 편향 검사] teacher가 피부톤을 체계적으로 바꾸는지 전수 측정한다.

■ 왜 필요한가 (2026-07-30 발견)
  QC 컨택트시트에서 어두운 피부 입력이 밝은 피부 타겟으로 바뀐 사례가 확인됐다.
  극단적 예: pair_06543 = 갈색 피부·브레이드 여성 → 금발 백인 여성.
    ① 공정성 — 비식별화 제품이 특정 인종의 외형을 바꾸면 그 자체로 결함
    ② 학습 오염 — 학생이 "밝게 + 피부톤 올리기"를 규칙으로 학습해버린다
  QC의 ALIGN 판정(ECC<0.55)에 걸린 건 13장뿐인데, 그건 **조명이 크게 바뀐 것만** 잡은 것이다.
  피부톤만 이동한 케이스는 ECC가 높아 안 걸리므로 전수 측정이 필요하다.

■ 지표
  1) ΔL*  : CIELAB 밝기 변화. 안정적이지만 조명과 피부톤이 섞인다.
  2) ΔITA : ITA = arctan((L*-50)/b*) × 180/π — 피부과학·공정성 문헌의 표준 피부톤 지표.
            조명 영향을 L*보다 덜 받지만 **b*가 0에 가까우면 발산**하므로
            b* >= B_MIN 인 표본에서만 계산한다(그렇지 않으면 -50°같은 허수치가 나온다).
  ΔITA > 0 → 피부가 밝아졌다.
  **입력이 어두울수록 ΔITA가 크다면 = 체계적 편향**(균일한 스타일화가 아님).

■ 측정 영역
  SFHQ-T2I는 중앙 정렬 인물이라 볼 주변(세로 40~58%, 가로 38~62%)을 고정 영역으로 쓴다.
  피부 검출 마스크(YCrCb 등)는 어두운 피부에서 검출률이 떨어져 **그 자체가 편향**을
  만들므로 쓰지 않는다. input·target에 같은 영역을 적용하므로 비교는 성립한다.
  영역 내에서도 반사광·짙은 그림자는 백분위(25~90)로 잘라낸다.
  ※ 검증: 이 조합에서 L* 38.6 / a* 19.0 / b* 18.0 (정상 피부 범위) 확인.

사용:
  python3 -u run/skin_tone_check.py --dir out/pairs_2511            # 전수
  python3 -u run/skin_tone_check.py --dir out/pairs_2511 --n 3000   # 균등 표본
"""
import argparse, os, glob, csv
import numpy as np
import cv2

R = 256          # 측정용 축소 (피부톤은 저주파라 충분)
REG = (0.40, 0.58, 0.38, 0.62)      # 볼 주변 (r0, r1, c0, c1)
PCT = (25, 90)                       # 그림자·반사광 제외 백분위
B_MIN = 8.0                          # ITA 안정 하한 (b* 가 이보다 작으면 발산)


def lab_stats(bgr):
    """볼 영역의 L*, a*, b* 중앙값."""
    b = cv2.resize(bgr, (R, R), interpolation=cv2.INTER_AREA)
    r0, r1, c0, c1 = REG
    reg = b[int(R * r0):int(R * r1), int(R * c0):int(R * c1)]
    lab = cv2.cvtColor(reg, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * 100.0 / 255.0          # OpenCV 8bit L: 0~255 → 0~100
    A = lab[..., 1] - 128.0
    B = lab[..., 2] - 128.0
    lo, hi = np.percentile(L, PCT[0]), np.percentile(L, PCT[1])
    m = (L >= lo) & (L <= hi)
    if m.sum() < 20:
        m = np.ones_like(L, bool)
    return float(np.median(L[m])), float(np.median(A[m])), float(np.median(B[m]))


def ita_of(L, B):
    """b* 가 충분할 때만 ITA. 아니면 None (발산 구간)."""
    if B < B_MIN:
        return None
    return float(np.degrees(np.arctan((L - 50.0) / B)))


def bucket(v):
    for lim, name in [(-30, "dark"), (10, "brown"), (28, "tan"),
                      (41, "intermediate"), (55, "light")]:
        if v < lim:
            return name
    return "very light"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=0, help="표본 수(0=전부). 전체에 균등 분포")
    ap.add_argument("--flag", type=float, default=12.0,
                    help="이 값 이상 밝아진(ΔITA) 페어를 불량 후보로 표시")
    args = ap.parse_args()

    din, dtg = os.path.join(args.dir, "input"), os.path.join(args.dir, "target")
    names = sorted(set(os.path.basename(p) for p in glob.glob(f"{din}/*.png"))
                   & set(os.path.basename(p) for p in glob.glob(f"{dtg}/*.png")))
    if not names:
        raise SystemExit(f"페어 없음: {args.dir}")
    if args.n > 0 and len(names) > args.n:
        names = names[::max(1, len(names) // args.n)][:args.n]
    print(f"[skin] {len(names)}쌍 측정 중... (영역 {REG}, 백분위 {PCT})")

    rows = []
    for i, n in enumerate(names):
        a, b = cv2.imread(os.path.join(din, n)), cv2.imread(os.path.join(dtg, n))
        if a is None or b is None:
            continue
        La, Aa, Ba = lab_stats(a)
        Lt, At, Bt = lab_stats(b)
        ia, it = ita_of(La, Ba), ita_of(Lt, Bt)
        rows.append(dict(name=n, idx=int(os.path.splitext(n)[0].split("_")[-1]),
                         L_in=La, a_in=Aa, b_in=Ba, L_tg=Lt, a_tg=At, b_tg=Bt,
                         d_L=Lt - La, d_a=At - Aa, d_b=Bt - Ba,
                         ita_in=ia if ia is not None else "",
                         ita_tg=it if it is not None else "",
                         d_ita=(it - ia) if (ia is not None and it is not None) else ""))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(names)}")

    with open(os.path.join(args.dir, "skin_tone.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    dL = np.array([r["d_L"] for r in rows])
    print(f"\n=== 전체 밝기 변화 ΔL* (target − input) ===")
    print(f"  평균 {dL.mean():+.2f}   중앙값 {np.median(dL):+.2f}   표준편차 {dL.std():.2f}")
    print(f"  밝아진 비율 {100*(dL>0).mean():.1f}%")
    print(f"  Δa* {np.mean([r['d_a'] for r in rows]):+.2f}   Δb* {np.mean([r['d_b'] for r in rows]):+.2f}")

    ok = [r for r in rows if r["d_ita"] != ""]
    print(f"\n=== ITA 분석 (b* >= {B_MIN} 인 안정 표본 {len(ok)}/{len(rows)}쌍) ===")
    if len(ok) < 30:
        print("  표본 부족 — ITA 판정 생략. ΔL* 만 참고할 것.")
        return

    d = np.array([r["d_ita"] for r in ok], float)
    ii = np.array([r["ita_in"] for r in ok], float)
    print(f"  ΔITA 평균 {d.mean():+.2f}   중앙값 {np.median(d):+.2f}   밝아진 비율 {100*(d>0).mean():.1f}%")

    print(f"\n=== ★ 입력 피부톤별 ΔITA — 편향의 핵심 ===")
    print(f"{'입력 구간':<14}{'n':>7}{'ITA_in 평균':>13}{'ΔITA 평균':>11}{'밝아짐':>9}")
    means = {}
    for name in ["dark", "brown", "tan", "intermediate", "light", "very light"]:
        m = np.array([bucket(r["ita_in"]) == name for r in ok])
        if m.sum() < 5:
            continue
        means[name] = d[m].mean()
        print(f"{name:<14}{int(m.sum()):>7}{ii[m].mean():>13.1f}{d[m].mean():>+11.2f}{100*(d[m]>0).mean():>8.0f}%")

    corr = float(np.corrcoef(ii, d)[0, 1])
    print(f"\n  상관계수 corr(입력 ITA, ΔITA) = {corr:+.3f}")
    print("  (음수가 클수록 '어두운 입력일수록 더 밝아진다' = 체계적 편향)")

    dark_side = [means[k] for k in ("dark", "brown", "tan") if k in means]
    light_side = [means[k] for k in ("light", "very light") if k in means]
    if dark_side and light_side:
        gap = float(np.mean(dark_side) - np.mean(light_side))
        print(f"  어두운 구간 − 밝은 구간 평균 ΔITA = {gap:+.2f}")
        if gap > 8 or corr < -0.3:
            print("\n  ⚠️ **체계적 편향 있음** — 어두운 피부가 유의하게 더 밝아진다.")
            print("     대책 ① 프롬프트에 'preserve the original skin tone and complexion' 추가 후 일부 재생성")
            print("          ② ΔITA 큰 페어 제외  ③ 학습 손실에 색 보존 항 추가")
        else:
            print("\n  ✅ 구간별 차이가 작다 — 화풍에 따른 균일한 이동으로 보인다.")
    else:
        print("  (구간 표본이 한쪽에 몰려 편향 판정 보류)")

    bad = sorted([r for r in ok if r["d_ita"] > args.flag], key=lambda r: -r["d_ita"])
    print(f"\n=== ΔITA > {args.flag} (크게 밝아진 페어) {len(bad)}장 / {len(ok)} ===")
    for r in bad[:15]:
        print(f"  #{r['idx']:<6} ITA {r['ita_in']:+6.1f} → {r['ita_tg']:+6.1f}  (Δ {r['d_ita']:+.1f})"
              f"   L* {r['L_in']:.0f}→{r['L_tg']:.0f}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5), dpi=110)
        ax.scatter(ii, d, s=4, alpha=0.25)
        ax.axhline(0, color="k", lw=1)
        z = np.polyfit(ii, d, 1)
        xs = np.linspace(ii.min(), ii.max(), 50)
        ax.plot(xs, np.polyval(z, xs), color="crimson", lw=2,
                label=f"slope={z[0]:+.3f}  corr={corr:+.3f}")
        ax.set_xlabel("input ITA   (lower = darker skin)")
        ax.set_ylabel("delta ITA   (positive = lightened)")
        ax.set_title("Skin tone shift: teacher input vs target")
        ax.legend(); fig.tight_layout()
        out = os.path.join(args.dir, "skin_tone.png")
        fig.savefig(out); print(f"\n산점도 → {out}")
    except Exception as e:
        print(f"\n[안내] 산점도 생략({e}). pip install matplotlib 로 활성화")
    print(f"CSV → {os.path.join(args.dir, 'skin_tone.csv')}")


if __name__ == "__main__":
    main()
