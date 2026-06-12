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
- 첫 실행 시 `yolo26n.pt`, `yolo26n-pose.pt`, `models/face_detection_yunet_2023mar.onnx`, YAMNet(~200MB) 자동 다운로드
- 키
  - `q` 종료
  - `r` ROI(안전 영역) 재정의 — 네 꼭짓점 다시 클릭
- HUD에 핵심 상태(persons/faces/cause/suf·clm_elapsed/cry/motion/head/torso/roi_in) 표시 (audio off = 마이크 없음)

## ROI 설정 (안전 영역 지정)

안전 영역은 **4점 폴리곤** `[TL, TR, BR, BL]`로 정의한다. `config.yaml`의 `auto_roi.detector`로 방식 선택:

| 방식 | 설명 |
|---|---|
| `manual` | 첫 실행 시 마우스로 네 꼭짓점 클릭 → `saved_roi.json`에 저장 후 재사용. `r` 키로 재정의 |

- 클릭 선택 중: 좌클릭=점 추가(최대 4), `r`=초기화, `Enter`/`c`=확정, `q`/`ESC`=취소
- 검출 실패 시 `fallback_polygon` 사용
- `saved_roi.json`은 **카메라·침대 위치에 종속**

## 위험 판정 규칙

### 영상 (YOLO26n + pose)

| 이벤트 | 조건 | 지속 시간 |
|---|---|---|
| `fall_risk` | 1.5초 윈도우 내 순 하강 ≥ 80px (**raw bbox center**, EMA 미적용) | — |
| `climbing_risk` | wrist가 ROI 변(난간)에 `rail_band_px` 이내 + 서있음 자세. `suffocation_risk` 활성 중엔 발행 안 함(상호 배타, 질식 우선) | 2초 |
| `suffocation_risk` | 아래 두 경로 중 하나 | 5초 |

**suffocation_risk — 2층 분리 구조** (감지와 원인 분류를 분리. 분류 신호는 감지를 차단하지 못한다 — 2026-06-10 카메라 검증에서 분류용 게이트(side 가드·climbing 베토)의 오탐이 그대로 질식 미탐으로 전이되는 구조적 결함이 확인돼 재설계):

- **1층 감지**: `face 미가시 AND 진입 게이트`가 5초 지속되면 위험. 진입 게이트 통과 후 억제 조건은 face_visible 단 하나.
  - `face_visible` = YuNet face가 person 안에 검출 OR ROI 안에 face 검출 OR pose 얼굴 키포인트(`face_kp_conf_threshold` 이상 `face_kp_min_visible`개 이상). 천장 보고 누우면(supine) 얼굴 키포인트가 살아 안전으로 빠지고, 엎드리면(prone) 죽어 위험으로 남는다(실측 supine 5/5 conf 0.93~0.98 vs prone 1/5 nose 0.08). HUD `face_src`(yunet/roi/kp)가 발동 항을 표시 — prone인데 face_visible이 켜지면 범인을 특정할 수 있다.
  - 진입 게이트(빈 방·인형 오탐 방지) = (face가 최근 `face_memory_s` 안에 보임 AND subject가 ROI 안에 있었음) OR **subject 중심이 ROI 안에 `presence_entry_s`(10초) 연속 존재**(`presence_gap_s` 2초 이내 검출 깜빡임은 연속 취급) OR 진행 중 래치. presence 경로가 "입장부터 얼굴을 한 번도 안 보여준 엎드림" 미탐을 해소한다. HUD `entry`(face/presence/latch)가 통과 경로를 표시.
- **2층 원인 라벨** (비차단 — 어떤 값이어도 감지를 끄지 못한다): cause는 항상 `flipped` 또는 `face_covered` 둘 중 하나(unknown 없음 — 서버 이벤트가 둘뿐이라 애매해도 둘 중 하나로 보내고 신뢰 맥락은 flags로 싣는다).
  - subject(pose 우선, 없으면 person)가 아예 없음 → `face_covered` (완전 파묻힘 = 검출 붕괴).
  - head 검출(crowdhuman) 보임 → `flipped`(뒤통수 노출=엎드림) / 안 보임 → `face_covered`(얼굴만 천 덮임 포함). head 검출기 미사용·예외 시 torso 폴백: 몸통 키포인트(어깨·엉덩이, 실측 prone 4/4 conf 0.95~0.99 vs 천 덮임 0/4)가 보이면 `flipped`, 소실이면 `face_covered`.
  - `out_of_view`(ROI 포함율 < `out_of_view_roi_threshold`)·`active_motion`(활동량 ≥ `motion_threshold`)·`side_lying`(옆누움 기하, 가드 활성 시) — **flag(메타데이터)로만** 이벤트·HUD에 실린다. 이전 구조에선 이들이 위험을 차단하는 게이트였고, 그 오류가 그대로 질식 미탐이 됐다.
- 이벤트 cause는 START 시점(5초가 차는 프레임)의 라벨로 고정된다(transition이 START에만 발행).

> 두 원인은 **서버 이벤트가 분리됨**: `flipped`→`PRONE_SUFFOCATION`, `face_covered`→`BLANKET_SUFFOCATION` (둘 다 DANGER).



### 음성 (YAMNet AudioSet)

| 이벤트 | YAMNet 클래스 | score 임계값 | 지속 시간 |
|---|---|---|---|
| `cry_detected` | Baby cry, Crying | 0.3 | 1초 |
| `babble_detected` | Babbling | 0.25 | 2초 |

- **person이 화면에 있을 때만** 판정

### 서버 이벤트 매핑 (MQTT)

| 내부 신호 | eventType | severity |
|---|---|---|
| `fall_risk` | `FALL` | DANGER |
| `suffocation_risk` (cause=`face_covered`) | `BLANKET_SUFFOCATION` | DANGER |
| `suffocation_risk` (cause=`flipped`) | `PRONE_SUFFOCATION` | DANGER |
| `climbing_risk` | `CLIMBING` | CAUTION |
| `cry_detected` | `CRYING` | CAUTION |
| `babble_detected` | `WHINING` | INFO |

- 위험 시작/종료를 각각 publish (`phase`: START / END), payload에 `duration_s`·`startedAt`·`endedAt` 포함
- 토픽·디바이스 시리얼은 `config.yaml`의 `mqtt` 섹션에서 설정

### 공통

- 좌표 스무딩: 지수이동평균(α=0.4) — 단, **fall은 raw center 사용**
- 순간 튐 완화: `DurationTracker` grace 0.5초
- 알림 폭주 방지: 동일 이벤트 타입 30초 쿨다운


## 진행 상황

- [x] 1: 웹캠 + YOLO26n 렌더 루프
- [x] 2: pose + ROI 기반 영상 휴리스틱 (suffocation/climbing/roi_exit)
- [x] 3: fall_risk 분리 + YAMNet 울음 감지 + duration_s 전송
- [x] 4: 서버 연동 (MQTT)
- [x] 5: 폴리곤 ROI 전환 + 질식 사라짐 추적 재설계 + 실물(인형) 검증
- [x] 6: 데모 준비 완료 / Jetson 이식 예정
