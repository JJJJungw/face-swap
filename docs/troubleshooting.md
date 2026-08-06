# 트러블슈팅

**증상으로 찾는다.** 실제로 이 프로젝트에서 발생했고 원인이 규명된 것만 적는다.
설계 판단(“이 방법을 쓸까”)은 [rejected.md](rejected.md)를 본다.

---

## 빠른 색인

| 증상 | 바로가기 |
|---|---|
| `ImportError: libcudart.so.13` | [의존성이 torch를 끌어내렸다](#importerror-libcudartso13) |
| `.git/index.lock: File exists` | [브리지로 git을 돌렸다](#gitindexlock-file-exists) |
| 학습·QC가 에러 없이 사라짐 | [SSH 끊김 = SIGHUP](#장시간-작업이-조용히-죽는다) |
| `size mismatch` / `NameError: detected_arch` | [체크포인트 구조 복원](#체크포인트-로딩-실패) |
| `ImportError: cannot import name ...` | [crop_utils 리팩터링 이후](#importerror-cannot-import-name) |
| `TypeError: RNG state must be a torch.ByteTensor` | [학습 재개 버그](#typeerror-rng-state-must-be-a-torch-bytetensor) |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | [MediaPipe 1.0.0](#mediapipe-attributeerror) |
| 판별자 손실 D가 0.01 아래로 붕괴 | [D 붕괴](#d가-붕괴한다-d--005) |
| 속도가 갑자기 몇 배 느려짐 | [속도 퇴행](#속도가-갑자기-느려졌다) |
| TensorRT가 매번 엔진을 새로 빌드 | [trt_cache 권한](#tensorrt가-매번-엔진을-다시-빌드한다) |
| `No space left on device` | [디스크](#디스크가-찼다) |
| 지표가 `nan` | [분모가 0](#지표가-nan으로-나온다) |
| `scp ...{a,b,c}.mp4: No such file` | [중괄호 확장](#scp-중괄호-확장이-안-된다) |
| 터미널이 `>` 프롬프트에 갇힘 | [히어독](#히어독이-붙여넣기에서-깨진다) |
| `manifest is missing N selected pairs` | [설계된 거부](#manifest-is-missing-n-selected-pairs) |
| 결과 영상이 세로로 눌려 보인다 | [아나모픽 입력(SAR)](#결과-영상이-눌려-보인다--아나모픽-입력) |

---

## 환경

### `ImportError: libcudart.so.13`

`pip install facenet-pytorch` **한 번에 런타임이 전부 죽었다.** 선언 핀이 `torch<2.3.0` / `numpy<2.0.0`
이라 pip이 **torch 2.13.0+cu130 → 2.2.2+cu121** 로 끌어내렸고, onnxruntime(CUDA13 빌드)이
import조차 불가능해졌다.

```bash
bash run/setup_venv.sh                     # 1) 런타임 핀 먼저
pip install -r run/requirements-train.txt  # 2) 학습·teacher 스택
pip install --no-deps facenet-pytorch      # 3) ★ --no-deps 필수
pip install requests tqdm                  # 4) --no-deps 로 빠지는 것만
```

**모든 설치는 `--dry-run`으로 먼저 확인한다.** 무엇이 다운그레이드되는지 보고 나서 실행한다.

**절대 금지:** `nvidia-*-cu12`(libcudart 충돌) · `bitsandbytes` 단독 설치(cu12 유발) ·
`opencv-python-headless`(심볼 충돌) · `HF_HUB_OFFLINE=1`(diffusers가 허브 메타데이터를 못 읽음) ·
torchvision 버전 직접 지정(resolver가 torch에 맞는 짝을 고르게 둔다).

스크립트를 `run_deid.sh` 밖에서 직접 실행할 때 같은 에러가 나면 `LD_LIBRARY_PATH` 문제다.
`.venv/bin/activate` 끝에 NVIDIA lib 경로 빌더가 추가돼 있으니 **`source .venv/bin/activate`를 먼저** 한다.

### MediaPipe AttributeError

MediaPipe 1.0.0은 레거시 `mp.solutions`를 제거했다. **Tasks API**(`vision.FaceLandmarker`)와
`face_landmarker.task` 모델 다운로드가 필요하다. `run/landmark_probe.py`가 자동으로 받는다.

### TensorRT가 매번 엔진을 다시 빌드한다

`gan_ckpt/keep/`을 `chmod -w` 로 잠글 때 하위의 `trt_cache/` 까지 잠기면 엔진 캐시를 못 쓴다.

```bash
chmod u+w gan_ckpt/keep/trt_cache
```

또한 ORT 1.27 TRT EP는 `libnvinfer.so.10`을 요구한다 → **TensorRT 10.x 필수**
(`tensorrt-cu13==10.16.1.11`). 11.x는 SONAME 불일치로 로드 실패.

### 디스크가 찼다

`No space left on device`가 나면 삭제는 여전히 되고 쓰기만 실패한다.
`out/`의 옛 실험 폴더, 중간 크롭, teacher 출력이 대부분을 차지한다.

```bash
df -h /
du -sh out/* | sort -h | tail -20
```

10.5시간짜리 teacher 굽기는 **시작 전에 여유 10GB 이상**을 확인한다.

---

## 운영

### `.git/index.lock: File exists`

원격 브리지로 git을 실행하면 잠금 파일이 남고, 브리지는 파일을 지울 수 없어 이후 모든 git이 막힌다.

```bash
rm -f .git/index.lock
```

**규약: git은 Mac 로컬 터미널에서만 실행한다.** 브리지는 파일 편집 전용이다.

### 장시간 작업이 조용히 죽는다

SSH가 끊기면 SIGHUP으로 **에러 없이** 종료된다. 실제로 학습 2회·QC 1회가 이렇게 날아갔다.

```bash
tmux new -s <이름>
python3 -u ...        # -u 로 버퍼링을 꺼야 진행이 보인다
# Ctrl+B, D 로 빠져나옴 / tmux attach -t <이름>
```

### `scp` 중괄호 확장이 안 된다

```bash
# ✗ 따옴표 안이라 로컬 zsh도 원격 셸도 확장하지 않는다
scp -i key.pem "user@host:~/out/{a,b,c}.mp4" .

# ✓ 루프
for N in a b c; do scp -i key.pem user@host:/home/ubuntu/face-swap/out/$N.mp4 .; done
```

### 히어독이 붙여넣기에서 깨진다

긴 마크다운을 `cat > file <<'EOF'` 로 터미널에 붙여넣으면 중간이 잘려 `>` 프롬프트에 갇힌다.
`Ctrl+C` 로 빠져나온 뒤, **긴 문서는 히어독 대신 레포에 커밋해서 `git pull` 로 옮긴다.**

---

## 학습

### 체크포인트 로딩 실패

`size mismatch` 또는 `NameError: detected_arch` 는 체크포인트 구조를 모르고 생성기를 만들 때 난다.
`train/train_student.py` 의 헬퍼를 쓴다.

```python
from train_student import build_generator, checkpoint_generator_kwargs
sd = torch.load(ckpt, map_location=dev, weights_only=False)
weights = sd["G"] if "G" in sd else sd
G = build_generator(**checkpoint_generator_kwargs(sd, weights))
G.load_state_dict(weights, strict=True)
```

저장된 `args`에서 `ch`/`arch`/`antialias`/게이트 초기값을 복원하고, 없으면 가중치에서 역추론한다.
**BlurPool을 켜면 `.kernel` 버퍼가 생기므로 구조를 모르고 만들면 strict 로딩이 실패한다.**

### `ImportError: cannot import name ...`

`crop_with_reflect` · `square_crop_bounds` 는 `run/crop_utils.py` 로 이동했고
`crop_with_reflect` 는 `crop_with_edge_padding`(BORDER_REPLICATE)으로 대체됐다.

```python
from build_localface_pairs import largest_face
from crop_utils import crop_with_edge_padding, occupancy_crop_bounds, square_crop_bounds
```

EC2와 Mac의 버전이 어긋나면 이 에러가 난다. **`git pull` 을 먼저 확인한다.**

### `TypeError: RNG state must be a torch.ByteTensor`

학습 재개 시 CUDA RNG 상태 복원에서 발생한다(`localface_probe1k` 500스텝에서 크래시).
`--resume` 대신 `--init-ckpt`(G 가중치만 로드, optimizer·step·split 초기화)를 쓰면 우회된다.

### D가 붕괴한다 (D < 0.05)

LSGAN의 평형점은 **D = 0.25** 다. 0.05 아래는 판별자 승리이고 생성기가 학습되지 않는다.

| 원인 | 조치 |
|---|---|
| adv 램프가 짧다 | `--adv-ramp` 를 2000 → 3000~4000 |
| **BlurPool을 켰다** | 구조적으로 공존 불가. 켜지 않는다 → [rejected](rejected.md#학습) |
| `--init-ckpt` 로 시작 (D가 새로 시작) | `--init-steps 500 --adv-ramp 1000` 으로 D에게 따라잡을 시간을 준다 |

### `manifest is missing N selected pairs`

**설계된 동작이다.** occupancy 크롭에서 패딩이 `--max-pad`를 넘는 페어는 거부된다
(13,500 중 209장 탈락 → 13,291). 학습 쪽은 매칭 실패 페어를 조용히 버리고 개수를 출력한다.

### 손실 항 하나가 갑자기 값이 커지고 평평하다

가중치가 0으로 떨어졌을 가능성이 크다. 실제로 `w_edge`가 코퍼스 교체 중 0이 된 채
30,000스텝이 돌았다 → [training-history](training-history.md#조용히-사라진-edge-loss)

**규약: 규격을 바꿀 때 손실 가중치를 이전 실행에서 복사해 오고, 첫 100스텝 로그에서
각 항이 이전과 같은 자릿수인지 대조한다.**

```bash
python3 -c "
import torch; sd=torch.load('<ckpt>',map_location='cpu')
a=sd.get('args') or {}; a=a if isinstance(a,dict) else vars(a)
print('\n'.join(f'{k}={v}' for k,v in sorted(a.items())))"
```

---

## 런타임

### 결과 영상이 눌려 보인다 — 아나모픽 입력

```
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,sample_aspect_ratio,display_aspect_ratio \
  -of default=nw=1 <영상>
```

`sample_aspect_ratio` 가 `1:1` 이 아니면 **아나모픽**이다.
유튜브에서 받은 팬캠이 그랬다 — 저장은 `720x1280` 인데 `SAR 256:81` 이라 표시는 `16:9` 가로.

**증상은 두 겹인데, 눈에 보이는 쪽이 덜 중요하다.**

1. 출력 표시 비율이 뒤집힌다(`9:16`). 우리가 SAR 을 안 이어붙이기 때문.
2. **모델이 찌그러진 얼굴을 본다.** `cv2.VideoCapture` 는 SAR 을 무시하고 저장된 픽셀만 준다.
   저장 버퍼 안에서 얼굴은 가로로 3.16배 눌려 있고, 그대로 검출기와 GAN 에 들어간다.
   학습 코퍼스에 없는 종횡비 = 완전한 분포 밖 입력.

→ **이 조건에서 나온 결과로 화질을 판단하면 안 된다. 무효다.**

`run/deid_cartoon.py` 는 시작할 때 SAR 을 프로브해서 자동으로 편다
(총 화소수는 유지 — `720x1280 · 256:81` → `1280x720`). 로그에 이렇게 찍힌다.

```
[sar] 아나모픽 입력 SAR=256:81 → 720x1280 를 1280x720 로 펴서 처리
```

이 줄이 안 보이면 SAR 이 `1:1` 이라 정규화가 불필요했다는 뜻이다.
진단용으로 끄려면 `--no-sar-fix`.

> **규칙: 새 영상으로 결과가 이상하면 화질을 논하기 전에 `ffprobe` 로 SAR 부터 본다.**

### 속도가 갑자기 느려졌다

CPU 합성이 7.8ms → **167.1ms**, 5.55× 실시간까지 퇴행한 적이 있다.
원인은 `cv2.GaussianBlur` 를 sigma≈53으로 float32 3채널 크롭에 프레임당 두 번 호출한 것이다
(534px 크롭에 약 80ms/회).

**큰 sigma 블러는 직접 돌리지 않는다.** `run/deid_cartoon.py` 의 `lowpass()` 처럼
축소 → 작은 블러 → 확대로 1ms 미만이 된다. bilateral도 축소해서 건다.

같은 GPU에서 학습이 돌고 있으면 속도 수치는 무의미하다. **측정 전에 학습을 멈춘다.**

### 지표가 `nan`으로 나온다

분모가 0이다. `shift_probe` 에서 입력을 `torch.roll` 로 밀었다가 되돌리면 원본과 **정확히 같아져서**
입력 변화량이 0이 됐다. 정렬은 출력에만 적용하고 분모는 원시 입력 변화량을 쓴다.

### 크롭 경계가 보인다 / 색이 뜬다

`--mask-feather` 를 키우는 것은 해법이 아니다(단차를 흐릿한 띠로 바꿀 뿐).
색은 `--color-mode chroma --color-match 1.0`, 밝기는 `--luma-match 0.7`
→ [runtime-pipeline](runtime-pipeline.md#크롭-경계-단차의-진짜-원인은-밝기였다)

### 선이 흔들린다

후처리로 못 고친다. 모델의 **비등변성**이며 `shift_probe` 의 "이동(비등변)" 열로 측정한다.
해법은 학습(`--w-equiv`)이다 → [training-history](training-history.md#equivariance-loss--흔들림의-해답이었다)

`--sharpen` 은 흔들림을 **새로 만든다**. 선명도가 필요하면 `--darken` 을 쓴다.

### 얼굴이 카툰화되지 않고 모자이크로 나온다

`--cartoon-min 150` 보다 작은 얼굴은 블러로 빠진다. 로그 끝의 카툰/블러 처리 개수를 확인한다.
소스에 작은 얼굴이 많으면 임계값을 낮춰야 하는데, 학생이 작은 얼굴에서 무너지지 않는지 먼저 본다.
