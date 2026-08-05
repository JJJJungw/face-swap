#!/usr/bin/env python3
"""[평가] 폴더 전체를 학생 모델로 스타일화한다. A/B 리뷰 입력을 만드는 용도.

체크포인트(.pt)와 ONNX 를 모두 받는다. 여러 모델을 같은 입력에 돌려 폴더별로 저장한 뒤
run/ab_review.py 로 블라인드 비교한다.

  python3 run/stylize_dir.py --input out/holdout_crops/input --out out/eval_eq \
      --ckpt gan_ckpt/keep/student_d8_edge3_eq_final.pt
"""
import argparse, glob, os, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))


def load_torch(ckpt, size, device):
    import torch
    from train_student import build_generator, checkpoint_generator_kwargs
    state = torch.load(ckpt, map_location=device, weights_only=False)
    weights = state["G"] if isinstance(state, dict) and "G" in state else state
    generator = build_generator(**checkpoint_generator_kwargs(state, weights)).to(device).eval()
    generator.load_state_dict(weights, strict=True)
    print(f"[model] {ckpt} step={state.get('step','?') if isinstance(state, dict) else '?'}")

    def run(bgr):
        rgb = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device)
        with torch.no_grad():
            out = generator(tensor).clamp(-1, 1)[0].permute(1, 2, 0).cpu().numpy()
        return cv2.cvtColor(((out + 1.0) * 127.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return run


def load_onnx(path, size):
    from deid_cartoon import build_providers
    import onnxruntime as ort
    session = ort.InferenceSession(path, providers=build_providers(path, False))
    name = session.get_inputs()[0].name
    print(f"[model] {path} (onnx)")

    def run(bgr):
        rgb = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        out = session.run(None, {name: rgb.transpose(2, 0, 1)[None]})[0][0]
        out = np.clip(out.transpose(1, 2, 0), -1, 1)
        return cv2.cvtColor(((out + 1.0) * 127.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--ckpt")
    group.add_argument("--onnx")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--include-file", default=None, dest="include_file")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    paths = {Path(p).stem: p for p in glob.glob(os.path.join(args.input, "*"))
             if Path(p).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")}
    if args.include_file:
        wanted = [s.strip() for s in Path(args.include_file).read_text(encoding="utf-8").splitlines() if s.strip()]
        missing = [s for s in wanted if s not in paths]
        if missing:
            print(f"[warn] include-file 의 {len(missing)}개 stem 이 입력에 없다: {missing[:5]}")
        stems = [s for s in wanted if s in paths]
    else:
        stems = sorted(paths)
    if not stems:
        raise SystemExit(f"입력 없음: {args.input}")

    if args.ckpt:
        import torch
        device = args.device if torch.cuda.is_available() else "cpu"
        run = load_torch(args.ckpt, args.size, device)
    else:
        run = load_onnx(args.onnx, args.size)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    done = 0
    for stem in stems:
        image = cv2.imread(paths[stem], cv2.IMREAD_COLOR)
        if image is None:
            continue
        cv2.imwrite(os.path.join(args.out, f"{stem}.png"), run(image))
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(stems)}")
    print(f"[완료] {done}장 → {args.out}")


if __name__ == "__main__":
    main()
