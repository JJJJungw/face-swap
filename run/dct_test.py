#!/usr/bin/env python3
"""[실험] DCT-Net 화질 테스트 (이미지 1장, 여러 스타일). 메인 파이프라인과 무관.
공식 DCT-Net(ModelScope, TensorFlow) 사전학습 모델로 얼굴 이미지를 스타일화해 animegan2와 비교.
별도 venv(.venv_dct)에서 실행. CPU라도 이미지 몇 장은 금방.

  python run/dct_test.py out/testface.png
"""
import sys, os, cv2

img = sys.argv[1] if len(sys.argv) > 1 else "out/testface.png"
if not os.path.exists(img):
    sys.exit(f"입력 이미지 없음: {img}  (ffmpeg로 프레임 먼저 추출)")
os.makedirs("out", exist_ok=True)

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

# 스타일별 사전학습 compound 모델 (첫 실행 시 각각 다운로드 ~수백MB)
STYLES = {
    "anime":     "damo/cv_unet_person-image-cartoon_compound-models",
    "3d":        "damo/cv_unet_person-image-cartoon-3d_compound-models",
    "handdrawn": "damo/cv_unet_person-image-cartoon-handdrawn_compound-models",
    # "sketch":   "damo/cv_unet_person-image-cartoon-sketch_compound-models",
    # "artstyle": "damo/cv_unet_person-image-cartoon-artstyle_compound-models",
}

for name, model in STYLES.items():
    try:
        p = pipeline(Tasks.image_portrait_stylization, model=model)
        r = p(img)
        out = f"out/dct_{name}.png"
        cv2.imwrite(out, r[OutputKeys.OUTPUT_IMG])
        print("saved", out)
    except Exception as e:
        print(f"[{name}] 실패: {e}")

print("완료 — out/dct_*.png 를 animegan2 결과와 비교")
