#!/usr/bin/env python3
"""[③-c 페어 자동 QC] 코퍼스 전체를 스캔해 불량 페어를 자동 검출한다.

왜 필요한가:
  - Lightning(4step)은 true_cfg_scale=1.0이라 **negative_prompt가 무시된다.**
    2509에서 쓰던 `extra person, multiple people, deformed` 가드가 없으므로
    인물 추가·구도 붕괴가 섞일 수 있는데, 1000장을 눈으로 훑는 건 비현실적이다.
  - 증류에서 결정적인 건 개별 장의 완성도가 아니라 **코퍼스 화풍 분산**이다.
    2M 학생은 분산을 고르지 못하고 평균내어 뭉갠다.
    → 화풍 이탈 장(outlier)을 걷어내면 CV가 내려가고 학생 선명도가 올라간다.

검출 축 (모두 input↔target 쌍 기준):
  ① 정합(ECC)      — 낮으면 teacher가 구도를 흔든 것. paired L1이 이 오차를 평균내 blur가 된다.
  ② 전역 이동(px)  — 드리프트
  ③ 화풍 이탈      — 코퍼스 중앙값 대비 robust z-score (median/MAD).
                     Laplacian·내부평탄도·색면수·채도·엣지밀도 5개 중 하나라도 크게 벗어나면 이탈.

출력:
  - <dir>/qc.csv                모든 페어의 지표
  - <dir>/qc_worst.png          최악 N장 컨택트시트(육안 확인용)
  - qc_reject.txt               불량 stem 목록(pair_curate.py 입력)
  - stdout                      코퍼스 통계 + 안전한 큐레이션 명령
                                + 제외했을 때 CV가 얼마나 개선되는지

사용:
  python3 run/pair_qc.py --dir out/pairs_2511
  # 결과의 컨택트시트와 reject 목록을 확인 후:
  python3 run/pair_curate.py --dir out/pairs_2511 --reject-file out/pairs_2511/qc_reject.txt
  python3 run/pair_curate.py --dir out/pairs_2511 --reject-file out/pairs_2511/qc_reject.txt --apply
"""
import argparse, os, csv
import numpy as np
import cv2
from pair_utils import discover_pairs

R_FEAT = 512    # 지표 정규화 해상도(학생 학습 해상도와 맞춤)
R_ECC = 256     # ECC는 반복법이라 비용이 큼 → 절반 해상도로


def feats(bgr):
    b = cv2.resize(bgr, (R_FEAT, R_FEAT), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    gf = g.astype(np.float32) / 255.
    lap = float(cv2.Laplacian(gf, cv2.CV_32F).var())
    q = (b // 32).astype(np.int32)
    idx = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
    cnt = np.bincount(idx.ravel(), minlength=512).astype(np.float32)
    p = cnt / cnt.sum()
    nc = float((p > 0.001).sum())
    sat = float(cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[..., 1].mean())
    edges = cv2.Canny(g, 100, 200)
    edge = float((edges > 0).mean())
    e = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    m = cv2.blur(g.astype(np.float32), (9, 9))
    s2 = cv2.blur(g.astype(np.float32) ** 2, (9, 9))
    std = np.sqrt(np.maximum(s2 - m * m, 0))
    inner = float(std[e == 0].mean()) if (e == 0).sum() else float("nan")
    return dict(lap=lap, inner=inner, nc=nc, sat=sat, edge=edge)


def align(a, b):
    ga = cv2.cvtColor(cv2.resize(a, (R_ECC, R_ECC)), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, (R_ECC, R_ECC)), cv2.COLOR_BGR2GRAY)
    try:
        w = np.eye(2, 3, dtype=np.float32)
        cc, w = cv2.findTransformECC(
            cv2.GaussianBlur(ga, (5, 5), 0), cv2.GaussianBlur(gb, (5, 5), 0),
            w, cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
        # R_ECC 기준 px → 원본 비율로 환산하지 않고 512 기준으로 통일
        sh = float((w[0, 2] ** 2 + w[1, 2] ** 2) ** 0.5) * (R_FEAT / R_ECC)
        return float(cc), sh
    except cv2.error:
        return float("nan"), float("nan")


def robust_z(v):
    """median/MAD 기반 z. 평균/표준편차는 이상치 자신에게 오염되므로 쓰지 않는다."""
    v = np.asarray(v, float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    if mad < 1e-9:
        return np.zeros_like(v), med, mad
    return 0.6745 * (v - med) / mad, med, mad


def cv_of(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return v.std() / v.mean() if len(v) and v.mean() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="input/ 와 target/ 을 가진 페어 폴더")
    ap.add_argument("--ecc-min", type=float, default=0.55, dest="ecc_min",
                    help="이 값 미만이면 정합 불량 (기본 0.55)")
    ap.add_argument("--shift-max", type=float, default=12.0, dest="shift_max",
                    help="512 기준 전역이동 허용치 px (기본 12)")
    ap.add_argument("--z-max", type=float, default=3.5, dest="z_max",
                    help="화풍 이탈 robust z 임계 (기본 3.5)")
    ap.add_argument("--n-worst", type=int, default=24, dest="n_worst",
                    help="컨택트시트에 실을 최악 장수")
    ap.add_argument("--sheet", default=None, help="컨택트시트 경로(기본 <dir>/qc_worst.png)")
    args = ap.parse_args()

    din = os.path.join(args.dir, "input")
    dtg = os.path.join(args.dir, "target")
    pairs = discover_pairs(din, dtg)
    if not pairs:
        raise SystemExit(f"페어 없음: {din} ∩ {dtg}")
    print(f"[qc] {len(pairs)}쌍 스캔 중...")

    rows = []
    pair_by_stem = {pair.stem: pair for pair in pairs}
    for i, pair in enumerate(pairs):
        a = cv2.imread(str(pair.input_path))
        b = cv2.imread(str(pair.target_path))
        if a is None or b is None:
            print(f"  [경고] 읽기 실패: {pair.stem}")
            continue
        f = feats(b)
        cc, sh = align(a, b)
        f.update(name=pair.stem, ecc=cc, shift=sh)
        rows.append(f)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pairs)}")

    if not rows:
        raise SystemExit("읽을 수 있는 페어가 없음")

    KEYS = ["lap", "inner", "nc", "sat", "edge"]
    zs = {}
    print("\n=== 코퍼스 화풍 통계 (target 기준) ===")
    print(f"{'지표':<10}{'중앙값':>10}{'MAD':>10}{'CV':>8}")
    for k in KEYS:
        z, med, mad = robust_z([r[k] for r in rows])
        zs[k] = z
        print(f"{k:<10}{med:10.3f}{mad:10.3f}{cv_of([r[k] for r in rows]):8.3f}")

    ecc = np.array([r["ecc"] for r in rows], float)
    shift = np.array([r["shift"] for r in rows], float)
    print(f"\n정합 ECC  중앙값 {np.nanmedian(ecc):.3f}   최저 {np.nanmin(ecc):.3f}")
    print(f"전역이동  중앙값 {np.nanmedian(shift):.1f}px  최대 {np.nanmax(shift):.1f}px")

    # ── 불량 판정 ──────────────────────────────────────────────────────────
    zmax = np.nanmax(np.abs(np.vstack([zs[k] for k in KEYS])), axis=0)
    bad_align = (ecc < args.ecc_min) | (shift > args.shift_max)
    bad_style = zmax > args.z_max
    bad = bad_align | bad_style | np.isnan(ecc)

    for r, za, bs, ba in zip(rows, zmax, bad_style, bad_align):
        r["zmax"] = float(za)
        # cv2.putText는 한글을 못 그리므로 시트용 코드는 ASCII로 둔다(CSV도 동일 값 사용)
        r["reason"] = "+".join(x for x, c in [("ALIGN", ba), ("STYLE", bs)] if c) or ""

    with open(os.path.join(args.dir, "qc.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["name", "ecc", "shift", "zmax", "reason"] + KEYS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    nb = int(bad.sum())
    print(f"\n=== 불량 {nb}장 / {len(rows)}장 ({nb/len(rows)*100:.1f}%) ===")
    print(f"  정합 불량(ALIGN)  {int(bad_align.sum())}장  (ECC<{args.ecc_min} 또는 이동>{args.shift_max}px)")
    print(f"  화풍 이탈(STYLE)  {int(bad_style.sum())}장  (robust z>{args.z_max})")

    worst = sorted([r for r, c in zip(rows, bad) if c],
                   key=lambda r: (r["ecc"] if not np.isnan(r["ecc"]) else -1))
    if worst:
        print(f"\n  최악 10장: " + ", ".join(
            f"{r['name']}(ecc {r['ecc']:.2f},z {r['zmax']:.1f})" for r in worst[:10]))

    # ── 제외 시 CV 개선 추정 ───────────────────────────────────────────────
    keep = [r for r, c in zip(rows, bad) if not c]
    if keep and nb:
        print(f"\n=== 제외 효과 (남는 {len(keep)}장) ===")
        print(f"{'지표':<10}{'제외전 CV':>11}{'제외후 CV':>11}{'개선':>8}")
        for k in KEYS:
            b0, b1 = cv_of([r[k] for r in rows]), cv_of([r[k] for r in keep])
            print(f"{k:<10}{b0:11.3f}{b1:11.3f}{(b0-b1)/b0*100:7.1f}%")

    # ── 컨택트시트 ─────────────────────────────────────────────────────────
    sheet = args.sheet or os.path.join(args.dir, "qc_worst.png")
    if worst:
        S, COLS = 200, 4
        tiles = []
        for r in worst[:args.n_worst]:
            pair = pair_by_stem[r["name"]]
            a = cv2.resize(cv2.imread(str(pair.input_path)), (S, S))
            b = cv2.resize(cv2.imread(str(pair.target_path)), (S, S))
            t = np.hstack([a, b])
            cv2.rectangle(t, (0, 0), (t.shape[1], 22), (0, 0, 0), -1)
            cv2.putText(t, f"{r['name'][-18:]} {r['reason']} ecc{r['ecc']:.2f}",
                        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            tiles.append(t)
        while len(tiles) % COLS:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)])
        cv2.imwrite(sheet, grid)
        print(f"\n컨택트시트 → {sheet}  (최악 {min(len(worst), args.n_worst)}장)")

    # ── pair_curate.py 로 넘길 stem 목록 ───────────────────────────────────
    rejected_stems = sorted(r["name"] for r, c in zip(rows, bad) if c)
    reject_file = os.path.join(args.dir, "qc_reject.txt")
    if rejected_stems:
        with open(reject_file, "w", encoding="utf-8") as handle:
            handle.write("".join(f"{stem}\n" for stem in rejected_stems))
        print(f"\n=== 다음 명령 ===")
        print(f"  # 반드시 컨택트시트 먼저 확인할 것 (자동 판정은 참고용)")
        print(f"  python3 run/pair_curate.py --dir {args.dir} --reject-file {reject_file}")
        print(f"  python3 run/pair_curate.py --dir {args.dir} --reject-file {reject_file} --apply")
    else:
        if os.path.exists(reject_file):
            os.remove(reject_file)
        print("\n불량 없음 — 큐레이션 불필요.")


if __name__ == "__main__":
    main()
