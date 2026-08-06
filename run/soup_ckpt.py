#!/usr/bin/env python3
"""[실험] 두 체크포인트의 가중치를 선형 보간한다 (model soup).

■ 왜 되는가
  파인튜닝을 --init-ckpt 로 이어붙이면 두 모델이 **같은 손실 분지 안**에 남는다.
  이 경우 가중치 평균이 두 모델의 성질을 섞은 모델로 동작하는 경우가 많다
  (Model Soups, ICML 2022). 학습이 아니라 산술이므로 몇 초면 된다.

■ 언제 쓰면 안 되는가
  서로 다른 초기화에서 독립적으로 학습한 모델은 분지가 달라 평균이 무너진다.
  **--init-ckpt 계보가 이어진 모델끼리만** 쓴다.

  out = alpha * A + (1 - alpha) * B

  예) A=eq(신원 낮음), B=tgt3k(선명함) → alpha 를 낮출수록 선명해지고 신원이 남는다.
"""
import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--alpha", type=float, action="append", required=True,
                    help="A 의 비중. 반복 지정 가능 (예: --alpha 0.75 --alpha 0.5)")
    ap.add_argument("--out-prefix", required=True, dest="out_prefix",
                    help="출력 경로 접두사. <prefix>_a075.pt 형태로 저장")
    args = ap.parse_args()

    def load(path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        weights = state["G"] if isinstance(state, dict) and "G" in state else state
        return state, weights

    state_a, wa = load(args.a)
    _, wb = load(args.b)

    if set(wa) != set(wb):
        only_a = sorted(set(wa) - set(wb))[:5]
        only_b = sorted(set(wb) - set(wa))[:5]
        raise SystemExit(f"구조가 다르다. A만: {only_a} / B만: {only_b}")

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    for alpha in args.alpha:
        if not 0.0 <= alpha <= 1.0:
            raise SystemExit(f"alpha 는 0~1: {alpha}")
        merged = {}
        for key in wa:
            va, vb = wa[key], wb[key]
            if va.dtype.is_floating_point:
                merged[key] = (va.float() * alpha + vb.float() * (1.0 - alpha)).to(va.dtype)
            else:
                merged[key] = va.clone()          # 정수 버퍼는 보간하지 않는다
        out_state = dict(state_a) if isinstance(state_a, dict) else {}
        out_state["G"] = merged
        out_state["soup"] = {"a": args.a, "b": args.b, "alpha": alpha}
        for drop in ("D", "optG", "optD", "opt_g", "opt_d"):
            out_state.pop(drop, None)             # 옵티마이저 상태는 의미 없다
        path = f"{args.out_prefix}_a{int(round(alpha * 100)):03d}.pt"
        torch.save(out_state, path)
        print(f"[soup] alpha={alpha:.2f}  →  {path}")

    print("\n※ --init-ckpt 계보가 이어진 모델끼리만 유효하다.")


if __name__ == "__main__":
    main()
