# 시작 가이드 — FLUX.1-schnell 사진→반실사 애니 테스트 (24GB GPU)

## 확정 스택 (전부 Apache/MIT)
- 베이스: **FLUX.1-schnell** (Apache 2.0, 게이트 없음)
- 실행: **diffusers** (Apache 2.0), PyTorch(BSD)
- 방식: **img2img** (원본 사진에 스타일만 입힘 → 포즈·표정 유지)

## 실행 순서

### 1) 파일 3개를 인스턴스로 복사
`setup.sh`, `flux_img2img_test.py`, `START_HERE.md`

### 2) 환경 설치
```bash
bash setup.sh          # CUDA 버전 다르면 setup.sh의 cu121을 인스턴스에 맞게 수정
source .venv/bin/activate
```
> `nvidia-smi`로 CUDA 버전 확인. 예) 12.1→cu121, 12.4→cu124, 11.8→cu118

### 3) 얼굴 사진 1장으로 테스트
```bash
python flux_img2img_test.py --image 내얼굴.jpg
```
→ `out/` 폴더에 strength(0.35~0.65)별 "원본|결과" 비교 이미지가 저장됨.

### 4) 프리셋 결정
- **낮은 strength(0.35)**: 원본에 가깝고 스타일 약함 (표정·신원 안전)
- **높은 strength(0.65)**: 스타일 강하고 원본에서 멀어짐 (딴사람 위험)
- 원하는 반실사 페인터리 + 표정 유지가 되는 값을 **프리셋으로 확정**
- 화풍이 아쉬우면 `PROMPT` 문구 조정 (예: "watercolor", "webtoon", "3d render" 등)

## 다음 단계 (프리셋 확정 후)
1. **신원 강화**(선택): Flux Depth/Canny ControlNet(Apache 판) 또는 IP-Adapter 추가
2. **영상 확장**: ffmpeg 프레임분해 → 얼굴검출/정렬 → 프리셋 per-frame 적용 → 시간일관성(Wan2.2/AnimateDiff or optical flow) → 재결합
3. **화풍 정밀화**(선택): Chroma/Flux schnell 위에 자체 LoRA 학습

## 메모리 팁 (24GB)
- 기본 `enable_model_cpu_offload()`로 24GB에서 구동됨(약간 느림).
- 더 빠르게: fp8 양자화 또는 텍스트 인코더 8bit 로드.
- OOM 나면 `--size 768`로 낮춰서 먼저 확인.

## 라이선스 체크포인트
- FLUX.1-schnell = Apache 2.0 ✅ / diffusers = Apache 2.0 ✅
- 나중에 ControlNet/LoRA 추가 시 **각 가중치가 Apache/MIT인지** 반드시 확인
- (주의: FLUX.1-**dev**는 비상업 → 반드시 **schnell** 사용)
