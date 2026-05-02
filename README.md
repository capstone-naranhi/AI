# AI 파트 — 영유아 안전 모니터링

## 환경

- Ubuntu 22.04, Python 3.10
- CPU-only 노트북 (내장그래픽)

(- 이후 Jetson orin nano 에 이식 및 GPU 사용 계획)

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행

```bash
python main.py
```

- 창이 뜨고 웹캠 프레임에 person bbox(초록) + face bbox(노랑) + pose 뼈대(파랑) + safe ROI(연빨강) + climb_rail ROI(주황)가 그려짐
- 첫 실행 시 `yolov8n.pt`, `yolov8n-pose.pt`, `models/face_detection_yunet_2023mar.onnx` 자동 다운로드
- 키
  - `q` 종료
  - `r` safe_roi 재선택
  - `c` climb_rail ROI 추가 (최대 4개, 화면에 rail0~rail3 표시)
  - `x` climb_rail ROI 전체 초기화
- HUD에 `cry_score` / `cry_elapsed`, `babble_score` / `babble_elapsed` 표시 (off = 마이크 없음)
- 첫 실행 시 YAMNet 모델 자동 다운로드 (~200MB)

## 위험 판정 규칙 (v2)

### 영상 (YOLOv8n + pose)

| 이벤트 | 조건 | 지속 시간 |
|---|---|---|
| `suffocation_risk` | person 내 face 없음 + pose로 cause 분기 | 5초 |
| `climbing_risk` | wrist가 난간 ROI 내 + 서있음 자세 | 2초 |
| `roi_exit_risk` | person 중심이 안전 ROI 밖 | 즉시(0.5s) |
| `fall_risk` | y좌표 하강 속도 ≥ 200px/s | 0.3초 |

**suffocation_risk cause 분기** (어깨·엉덩이 4개 키포인트 기준):
- 3개 이상 보임 → `flipped` (뒤집혀 얼굴이 눌린 상태)
- 2개 이하 보임 → `blanket` (이불에 덮여 키포인트 미검출)

### 음성 (YAMNet AudioSet)

| 이벤트 | YAMNet 클래스 | score 임계값 | 지속 시간 |
|---|---|---|---|
| `cry_detected` | Baby cry (20), Crying (19) | 0.3 | 1초 |
| `babble_detected` | Babbling (4) | 0.25 | 2초 |

- 두 이벤트 모두 **person이 화면에 있을 때만** 판정


### 공통

- 좌표 스무딩: 지수이동평균(α=0.4)
- 순간 튐 완화: `DurationTracker` grace 0.5초
- 알림 폭주 방지: 동일 이벤트 타입 30초 쿨다운
- 모든 이벤트 payload에 `duration_s` 포함


## 진행 상황

- [x] 1: 웹캠 + YOLOv8n 렌더 루프
- [x] 2: pose + ROI 기반 영상 휴리스틱 (suffocation/climbing/roi_exit)
- [x] 3 (v2): fall_risk 분리 + YAMNet 울음·옹알이 감지 + duration_s 전송
- [ ] 4: 서버 연동 (MQTT)
- [ ] 5: 통합 테스트·튜닝
- [ ] 6: 데모 준비
