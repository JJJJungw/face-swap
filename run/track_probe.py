#!/usr/bin/env python3
"""트랙 프로브 — 검출된 얼굴을 IoU로 프레임 간 연결(gate.py 방식)해 트랙 ID·크기를 부여하고,
영상에 오버레이 + 트랙별 크기 타임라인 통계를 낸다. 캐싱/리인액트 설계 판단용 디버그 도구.

핵심 관찰 포인트:
  1) 컷에서 트랙이 끊기는지  2) 한 인물이 큰↔작은 얼마나 오가는지(캐싱 이득 구간)
  3) 작은 얼굴이 빌려올 '큰 프레임 소스'가 있는지  4) 다인물 교차 시 ID switch(=색 튐)

사용:
  bash run/run_probe.sh --video input/swap2.mp4 --trt
"""
import os, sys, argparse, subprocess, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deid_cartoon import Detector          # TRT 검출 그대로 재사용

PALETTE = [(66,133,244),(219,68,55),(244,180,0),(15,157,88),(171,71,188),
           (0,172,193),(255,112,67),(158,157,36),(240,98,146),(121,85,72),
           (96,125,139),(255,167,38)]      # BGR, ID별 고정색 → 교차 시 색 튐이 ID switch

def _iou(a, b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1]); ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    iw,ih = max(0.,ix2-ix1), max(0.,iy2-iy1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

class IoUTracker:
    """온라인 IoU 트래커(그리디). gate.py의 프레임간 재구성 로직을 실시간 ID 부여형으로."""
    def __init__(self, iou_thr=0.3, max_gap=5):
        self.iou_thr = iou_thr; self.max_gap = max_gap
        self.active = []          # {"id","box","last"}
        self.arch = {}            # id -> {"sizes":[(frame,size)], "first":f, "last":f}
        self.next_id = 0

    def reset(self):              # 컷 → 활성 트랙 비움(아카이브는 유지)
        self.active = []

    def step(self, dets, idx):
        used = set(); out = []
        for d in dets:
            box = d[:4]; best = self.iou_thr; bi = -1
            for ti, tr in enumerate(self.active):
                if ti in used or (idx - tr["last"]) > self.max_gap: continue
                v = _iou(box, tr["box"])
                if v >= best: best = v; bi = ti
            size = int(max(box[2]-box[0], box[3]-box[1]))
            if bi >= 0:
                tr = self.active[bi]; tr["box"] = box; tr["last"] = idx; used.add(bi)
                tid = tr["id"]
            else:
                tid = self.next_id; self.next_id += 1
                self.active.append({"id": tid, "box": box, "last": idx}); used.add(len(self.active)-1)
                self.arch[tid] = {"sizes": [], "first": idx, "last": idx}
            self.arch[tid]["sizes"].append((idx, size)); self.arch[tid]["last"] = idx
            out.append((tid, box, size))
        self.active = [t for t in self.active if (idx - t["last"]) <= self.max_gap]  # stale 제거
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="models/base_v2f2_1280_fp16.onnx")
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--trt", action="store_true")
    ap.add_argument("--encoder", default="nvenc", choices=["nvenc", "x264"])
    ap.add_argument("--scene-cut", type=float, default=28.0, dest="scene_cut", help="장면전환 임계(0=끔)")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--max-gap", type=int, default=5, dest="max_gap")
    ap.add_argument("--good-size", type=int, default=200, dest="good_size", help="이 이상=캐시 소스로 양호")
    ap.add_argument("--small-size", type=int, default=150, dest="small_size", help="이 미만=뭉개짐 위험")
    ap.add_argument("--max-frames", type=int, default=0, dest="max_frames")
    ap.add_argument("--out", default="out/track_probe.mp4")
    ap.add_argument("--csv", default="out/track_probe.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    det = Detector(args.model, args.size, use_trt=args.trt)
    trk = IoUTracker(args.iou, args.max_gap)

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 30.0; total = int(cap.get(7))
    print(f"{W}x{H} @ {fps:.0f}fps, {total} frames | trt={args.trt}")

    enc = (["-c:v","h264_nvenc","-preset","p4","-cq","23"] if args.encoder=="nvenc" else ["-c:v","libx264","-crf","23"])
    cmd = ["ffmpeg","-y","-loglevel","error","-nostats","-f","rawvideo","-pix_fmt","bgr24",
           "-s",f"{W}x{H}","-r",f"{fps}","-i","-"] + enc + ["-pix_fmt","yuv420p", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    idx = ncut = 0; prev_g = None; multi_frames = 0; t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok: break
        idx += 1
        cut = False
        if args.scene_cut > 0:
            g = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY)
            if prev_g is not None and np.abs(g.astype(np.int16) - prev_g.astype(np.int16)).mean() > args.scene_cut:
                trk.reset(); ncut += 1; cut = True
            prev_g = g
        dets = det.detect(frame, W, H)
        tracks = trk.step(dets, idx)
        if len(tracks) >= 2: multi_frames += 1
        for tid, box, size in tracks:
            x1, y1, x2, y2 = [int(v) for v in box]; col = PALETTE[tid % len(PALETTE)]
            small = size < args.small_size
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if not small else 1)
            tag = f"ID{tid} {size}px" + (" SMALL" if small else "")
            cv2.rectangle(frame, (x1, y1-22), (x1+len(tag)*11+6, y1), col, -1)
            cv2.putText(frame, tag, (x1+3, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if cut:
            cv2.rectangle(frame, (0,0), (W,40), (0,0,255), -1)
            cv2.putText(frame, f"SCENE CUT #{ncut}", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(frame, f"f{idx} tracks={len(tracks)}", (10,H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        proc.stdin.write(frame.tobytes())
        if idx % 30 == 0: print(f"  {idx}/{total}  {idx/(time.perf_counter()-t0):.1f}fps", end="\r")
        if args.max_frames and idx >= args.max_frames: break
    cap.release(); proc.stdin.close(); proc.wait(); print()

    # ── 트랙별 통계 ──
    rows = []
    borrow = []      # 캐시 이득: 큰 프레임(>=good) 있고 작은 프레임(<small)도 있음
    always_small = []  # 평생 작음(<small) → 샷 안에 빌려올 소스 없음 = 하드케이스
    with open(args.csv, "w") as f:
        f.write("track_id,frame,size\n")
        for tid, a in sorted(trk.arch.items()):
            sizes = [s for _, s in a["sizes"]]; span = a["last"]-a["first"]+1; n = len(sizes)
            mx, mn = max(sizes), min(sizes)
            rows.append((tid, n, span, mn, mx))
            for fr, s in a["sizes"]: f.write(f"{tid},{fr},{s}\n")
            if mx >= args.good_size and mn < args.small_size: borrow.append(tid)
            if mx < args.small_size: always_small.append(tid)

    dur = time.perf_counter()-t0
    print(f"\n===== 트랙 프로브 요약 =====")
    print(f"프레임 {idx} | 컷 {ncut} | 다인물(≥2얼굴) 프레임 {multi_frames} ({100*multi_frames/max(idx,1):.0f}%)")
    print(f"총 트랙 ID {len(trk.arch)}개  (컷마다 리셋되어 실제 인물보다 많을 수 있음)")
    print(f"{'ID':>4} {'출현':>5} {'구간':>5} {'최소px':>6} {'최대px':>6}")
    for tid, n, span, mn, mx in rows[:40]:
        flag = ""
        if tid in borrow: flag = "  ← 캐시이득(큰↔작은)"
        elif tid in always_small: flag = "  ← 평생작음(소스없음)"
        print(f"{tid:>4} {n:>5} {span:>5} {mn:>6} {mx:>6}{flag}")
    if len(rows) > 40: print(f"  ... 외 {len(rows)-40}개 (전체는 {args.csv})")
    print(f"\n캐시 이득 트랙(큰≥{args.good_size} & 작은<{args.small_size}): {len(borrow)}개 {borrow[:20]}")
    print(f"평생 작음 트랙(<{args.small_size} 내내, 빌려올 소스 없음): {len(always_small)}개")
    print(f"-> out: {args.out} / {args.csv}  ({dur:.1f}s)")

if __name__ == "__main__":
    main()
