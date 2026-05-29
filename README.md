# AI 파트 — 영유아 안전 모니터링

침대(또는 놀이 영역)를 카메라로 지켜보며 **낙상·기어오름·질식(덮임)·영역 이탈·울음**을 감지하고 전송한다.

## 환경

- Ubuntu 22.04, Python 3.10
- CPU-only 노트북 (내장그래픽)에서 동작 확인
- (이후 Jetson orin nano 이식 및 GPU 사용 계획)

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

- 창이 뜨고 웹캠 프레임에 person bbox(초록) + face bbox(노랑) + pose 뼈대(파랑) + safe ROI(연빨강) 폴리곤이 그려짐
- 첫 실행 시 `yolov8n.pt`, `yolov8n-pose.pt`, `models/face_detection_yunet_2023mar.onnx`, YAMNet(~200MB) 자동 다운로드
- 키
  - `q` 종료
  - `r` ROI(안전 영역) 재정의 — 네 꼭짓점 다시 클릭
- HUD에 `cry_score` / `babble_score`와 각 휴리스틱 진단(diag) 표시 (audio off = 마이크 없음)

## ROI 설정 (안전 영역 지정)

안전 영역은 **4점 폴리곤** `[TL, TR, BR, BL]`로 정의한다. `config.yaml`의 `auto_roi.detector`로 방식 선택:

| 방식 | 설명 |
|---|---|
| `manual` | 첫 실행 시 마우스로 네 꼭짓점 클릭 → `saved_roi.json`에 저장 후 재사용. `r` 키로 재정의 |

- 클릭 선택 중: 좌클릭=점 추가(최대 4), `r`=초기화, `Enter`/`c`=확정, `q`/`ESC`=취소
- 검출 실패 시 `fallback_polygon` 사용
- `saved_roi.json`은 **카메라·침대 위치에 종속**

## 위험 판정 규칙

### 영상 (YOLOv8n + pose)

| 이벤트 | 조건 | 지속 시간 |
|---|---|---|
| `fall_risk` | person 중심 y 하강 속도 ≥ 200px/s (**raw bbox center**, EMA 미적용) | 0.3초 |
| `climbing_risk` | wrist가 ROI 변(난간)에 `rail_band_px` 이내 + 서있음 자세 | 2초 |
| `suffocation_risk` | 아래 두 경로 중 하나 | 5초 |
| `roi_exit_risk` | person 중심이 안전 ROI 밖 | 즉시(grace 0.5s) |

**suffocation_risk 두 경로** (키포인트 가시성에 의존하지 않음):
- `face_covered`: person은 보이는데 그 안에 face가 없고, face가 최근(`face_memory_s`) 보인 적 있음 → 얼굴 가림
- `disappeared`: ROI 안에서 보이던 person이 `roi_exit` 없이 사라짐 → 이불이 몸 전체를 덮어 person 검출이 실패한 경우를 위험으로 해석

> 두 경로 모두 서버 이벤트로는 `BLANKET_SUFFOCATION`(DANGER) 하나로 통합.
> **알려진 한계:** 보호자가 아기를 손으로 들어올려 ROI 밖으로 빼면 `roi_exit`가 경계를 못 잡아 `disappeared` 오탐이 날 수 있음. 안전 우선(false positive < false negative) 원칙으로 그대로 둠.

### 음성 (YAMNet AudioSet)

| 이벤트 | YAMNet 클래스 | score 임계값 | 지속 시간 |
|---|---|---|---|
| `cry_detected` | Baby cry, Crying | 0.3 | 1초 |
| `babble_detected` | Babbling | 0.25 | 2초 |

- 두 이벤트 모두 **person이 화면에 있을 때만** 판정

### 서버 이벤트 매핑 (MQTT)

| 내부 신호 | eventType | severity |
|---|---|---|
| `fall_risk` | `FALL` | DANGER |
| `suffocation_risk` | `BLANKET_SUFFOCATION` | DANGER |
| `climbing_risk` | `CLIMBING` | CAUTION |
| `roi_exit_risk` | `ROI_EXIT` | CAUTION |
| `cry_detected` | `CRYING` | CAUTION |
| `babble_detected` | `WHINING` | INFO |

- 위험 시작/종료를 각각 publish (`phase`: START / END), payload에 `duration_s`·`startedAt`·`endedAt` 포함
- 토픽·디바이스 시리얼은 `config.yaml`의 `mqtt` 섹션에서 설정

### 공통

- 좌표 스무딩: 지수이동평균(α=0.4) — 단, **fall은 raw center 사용**
- 순간 튐 완화: `DurationTracker` grace 0.5초
- 알림 폭주 방지: 동일 이벤트 타입 30초 쿨다운


## 진행 상황

- [x] 1: 웹캠 + YOLOv8n 렌더 루프
- [x] 2: pose + ROI 기반 영상 휴리스틱 (suffocation/climbing/roi_exit)
- [x] 3: fall_risk 분리 + YAMNet 울음·옹알이 감지 + duration_s 전송
- [x] 4: 서버 연동 (MQTT)
- [x] 5: 폴리곤 ROI 전환 + 질식 사라짐 추적 재설계 + 실물(인형) 검증
- [ ] 6: 데모 준비 / Jetson 이식
