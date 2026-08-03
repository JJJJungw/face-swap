#!/usr/bin/env python3
"""[⑤ 런타임 교체] ④ 학생(.pt) → ONNX(512 고정) export.
런타임(deid_cartoon.py --gan-backend onnx --gan-onnx <이 파일>)의 animegan2 placeholder 슬롯에 꽂는다.
우리 Generator 구조 그대로 export → 입력 x[1,3,512,512] in[-1,1] → 출력 y in[-1,1] (animegan2 슬롯과 동일 시그니처).

사용:
  python run/export_student_onnx.py --ckpt train/student_v6/student_final.pt --out gan_ckpt/student_512.onnx
"""
import os, sys, argparse, torch

# 우리 Generator 정의 재사용 (train/train_student.py)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from train_student import build_generator, checkpoint_generator_arch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="학생 체크포인트 .pt ({'G':state_dict} 또는 순수 state_dict)")
    ap.add_argument("--out", default="gan_ckpt/student_512.onnx")
    ap.add_argument("--size", type=int, default=512, help="런타임 고정 입력 크기(기본 512)")
    args = ap.parse_args()

    try:
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
    weights = checkpoint["G"] if isinstance(checkpoint, dict) and "G" in checkpoint else checkpoint
    _ch = weights["in_conv.1.weight"].shape[0]      # 체크포인트가 채널 수를 알고 있다
    arch = checkpoint_generator_arch(checkpoint, weights)
    print(f"[export] arch={arch} ch={_ch} 자동 감지")
    m = build_generator(ch=_ch, arch=arch).eval()
    m.load_state_dict(weights)
    print(f"[load] {args.ckpt}  (params={sum(p.numel() for p in m.parameters())/1e6:.2f}M)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    dummy = torch.randn(1, 3, args.size, args.size)
    # torch 2.13 기본 dynamo(onnxscript 필요) 회피 → legacy TorchScript exporter(TRT 친화)
    try:
        torch.onnx.export(m, dummy, args.out, input_names=["x"], output_names=["y"],
                          opset_version=17, dynamo=False)
    except TypeError:
        torch.onnx.export(m, dummy, args.out, input_names=["x"], output_names=["y"], opset_version=17)
    print(f"[export] {args.out}  (size={args.size}, x[-1,1]→y[-1,1])")

    # 간단 검증: onnxruntime로 1회 추론 shape 확인(설치돼 있으면)
    try:
        import numpy as np, onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        y = sess.run(["y"], {"x": np.random.randn(1, 3, args.size, args.size).astype("float32")})[0]
        print(f"[verify] onnxruntime out shape = {y.shape}  → OK")
    except Exception as e:
        print(f"[verify] 스킵({e}) — 런타임에서 확인")


if __name__ == "__main__":
    main()
