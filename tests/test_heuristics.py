from vision.heuristics import evaluate_fall, evaluate_roi_exit

SAFE_POLY = [(0, 0), (100, 0), (100, 100), (0, 100)]  # TL,TR,BR,BL


def test_roi_exit_none_center():
    active, diag = evaluate_roi_exit(None, SAFE_POLY)
    assert active is False
    assert diag["block"] == "no_center"


def test_roi_exit_inside():
    active, diag = evaluate_roi_exit((50.0, 50.0), SAFE_POLY)
    assert active is False
    assert diag["block"] == "inside_polygon"


def test_roi_exit_outside():
    active, diag = evaluate_roi_exit((150.0, 50.0), SAFE_POLY)
    assert active is True


from vision.heuristics import evaluate_climbing
from vision.pose import Pose


def _pose(**overrides):
    base = {
        "nose": (0.0, 0.0, 0.0), "left_eye": (0.0, 0.0, 0.0), "right_eye": (0.0, 0.0, 0.0),
        "left_ear": (0.0, 0.0, 0.0), "right_ear": (0.0, 0.0, 0.0),
        "left_shoulder": (10.0, 100.0, 0.9), "right_shoulder": (20.0, 100.0, 0.9),
        "left_elbow": (0.0, 0.0, 0.0), "right_elbow": (0.0, 0.0, 0.0),
        "left_wrist": (0.0, 0.0, 0.0), "right_wrist": (0.0, 0.0, 0.0),
        "left_hip": (10.0, 200.0, 0.9), "right_hip": (20.0, 200.0, 0.9),
        "left_knee": (0.0, 0.0, 0.0), "right_knee": (0.0, 0.0, 0.0),
        "left_ankle": (10.0, 300.0, 0.9), "right_ankle": (20.0, 300.0, 0.9),
    }
    base.update(overrides)
    return Pose(bbox=(0, 0, 100, 400), keypoints=base)


# 사람 bbox/포즈 좌표계에 맞춘 안전 폴리곤. 하단 변(y=300 근처)을 난간으로 사용
SAFE_POLY_CLIMB = [(0, 0), (100, 0), (100, 300), (0, 300)]
RAIL_BAND_PX = 40.0


def test_climbing_empty_polygon():
    active, diag = evaluate_climbing((50.0, 290.0), _pose(), [], RAIL_BAND_PX, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "no_polygon"


def test_climbing_pose_none():
    active, diag = evaluate_climbing(None, None, SAFE_POLY_CLIMB, RAIL_BAND_PX, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "no_pose"


def test_climbing_no_wrist():
    active, diag = evaluate_climbing(None, _pose(), SAFE_POLY_CLIMB, RAIL_BAND_PX, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "no_wrist"


def test_climbing_wrist_outside_polygon():
    active, diag = evaluate_climbing((500.0, 500.0), _pose(), SAFE_POLY_CLIMB, RAIL_BAND_PX, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "wrist_outside_polygon"


def test_climbing_wrist_far_from_rail():
    # 폴리곤 내부지만 어느 변과도 band보다 멀리 (중앙 부근)
    active, diag = evaluate_climbing((50.0, 150.0), _pose(), SAFE_POLY_CLIMB, RAIL_BAND_PX, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "not_near_rail"


def test_climbing_shoulders_invisible():
    pose = _pose(left_shoulder=(10.0, 100.0, 0.1), right_shoulder=(20.0, 100.0, 0.1))
    active, diag = evaluate_climbing((50.0, 290.0), pose, SAFE_POLY_CLIMB, RAIL_BAND_PX, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "shoulder_or_hip_invisible"


def test_climbing_hips_invisible():
    pose = _pose(left_hip=(10.0, 200.0, 0.1), right_hip=(20.0, 200.0, 0.1))
    active, diag = evaluate_climbing((50.0, 290.0), pose, SAFE_POLY_CLIMB, RAIL_BAND_PX, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "shoulder_or_hip_invisible"


def test_climbing_margin_too_small():
    # 어깨와 엉덩이 y가 가까움 → 서있지 않음
    pose = _pose(left_hip=(10.0, 105.0, 0.9), right_hip=(20.0, 105.0, 0.9))
    active, diag = evaluate_climbing((50.0, 290.0), pose, SAFE_POLY_CLIMB, RAIL_BAND_PX, 0.5, 20.0)
    assert active is False
    assert diag["block"] == "not_standing"


def test_climbing_all_conditions_met():
    # 폴리곤 내부 + 하단 변(y=300)까지 거리 10 ≤ band + 서있음
    active, diag = evaluate_climbing((50.0, 290.0), _pose(), SAFE_POLY_CLIMB, RAIL_BAND_PX, 0.5, 20.0)
    assert active is True
    assert diag["rail_edge"] == "bottom"


from vision.face import Face
from vision.heuristics import evaluate_suffocation
from vision.person import Person


def _person_at(x1, y1, x2, y2):
    return Person(bbox=(x1, y1, x2, y2), confidence=0.9)


def _face_at(x1, y1, x2, y2):
    return Face(bbox=(x1, y1, x2, y2), confidence=0.9)


def test_suffocation_person_absent_never_in_roi():
    # person을 ROI 안에서 본 적 없으면(빈 방) 판정 안 함
    active, cause, diag = evaluate_suffocation(
        None, [], face_recently_seen=True, person_was_in_roi=False)
    assert active is False
    assert cause is None
    assert diag["block"] == "no_person_not_in_roi"


def test_suffocation_disappeared_in_roi():
    # ROI 안에서 보이던 person이 ROI 이탈 없이 사라짐 → 덮임 의심
    active, cause, diag = evaluate_suffocation(
        None, [], face_recently_seen=False, person_was_in_roi=True)
    assert active is True
    assert cause == "disappeared"


def test_suffocation_face_detected_inside_person():
    person = _person_at(0, 0, 100, 200)
    face = _face_at(20, 20, 60, 60)
    active, cause, diag = evaluate_suffocation(
        person, [face], face_recently_seen=True, person_was_in_roi=True)
    assert active is False
    assert cause is None


def test_suffocation_face_absent_recently_seen():
    # face가 최근 보였는데 지금 없으면 risk
    person = _person_at(0, 0, 100, 200)
    active, cause, diag = evaluate_suffocation(
        person, [], face_recently_seen=True, person_was_in_roi=True)
    assert active is True
    assert cause == "face_covered"


def test_suffocation_face_absent_never_seen():
    # face를 한 번도 못 봤으면 오탐 방지
    person = _person_at(0, 0, 100, 200)
    active, cause, diag = evaluate_suffocation(
        person, [], face_recently_seen=False, person_was_in_roi=True)
    assert active is False
    assert diag["block"] == "face_never_seen"


def test_suffocation_face_outside_person_ignored():
    # person bbox 밖의 face는 무시 — recently_seen이면 risk
    person = _person_at(0, 0, 100, 200)
    face = _face_at(200, 200, 240, 240)
    active, cause, diag = evaluate_suffocation(
        person, [face], face_recently_seen=True, person_was_in_roi=True)
    assert active is True
    assert cause == "face_covered"


def test_fall_none_center():
    active, diag = evaluate_fall(None, (50.0, 100.0), 0.1, 200.0)
    assert active is False
    assert diag["block"] == "no_center_or_dt"


def test_fall_none_prev():
    active, diag = evaluate_fall((50.0, 200.0), None, 0.1, 200.0)
    assert active is False
    assert diag["block"] == "no_center_or_dt"


def test_fall_slow_descent():
    # 0.1초에 5px 하강 = 50px/s < 200px/s
    active, diag = evaluate_fall((50.0, 105.0), (50.0, 100.0), 0.1, 200.0)
    assert active is False
    assert diag["block"] == "velocity_too_low"


def test_fall_fast_descent():
    # 0.1초에 30px 하강 = 300px/s >= 200px/s
    active, diag = evaluate_fall((50.0, 130.0), (50.0, 100.0), 0.1, 200.0)
    assert active is True
    assert diag["fall_velocity"] == 300.0


def test_fall_ascending_ignored():
    # 위로 올라가는 것은 낙상 아님
    active, diag = evaluate_fall((50.0, 70.0), (50.0, 100.0), 0.1, 200.0)
    assert active is False
    assert diag["block"] == "velocity_too_low"
