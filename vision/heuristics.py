"""영상 기반 위험 평가 — v1.

규칙:
  1. suffocation_risk: face가 최근 보였는데 person 안에 face 없음 지속
  2. climbing_risk: pose wrist가 난간 ROI 안 + 서있음 자세 지속
  3. roi_exit_risk: person 중심이 안전 ROI 밖
  4. fall_risk: person 중심 y가 빠르게 하강 (낙상)

각 evaluate_* 는 순수 함수로 (판정, 진단) 반환.
"""
from dataclasses import dataclass, field
from typing import Optional

from .face import Face
from .person import Person
from .pose import Pose
from .roi_geometry import EDGE_LABELS, nearest_edge, point_in_polygon

__all__ = [
    "RiskSignal", "main_person",
    "evaluate_roi_exit", "evaluate_fall", "evaluate_climbing", "evaluate_suffocation",
    "face_inside_person",
]


@dataclass
class RiskSignal:
    type: str
    confidence: float
    metadata: dict = field(default_factory=dict)


def main_person(persons: list[Person]) -> Optional[Person]:
    if not persons:
        return None
    return max(persons, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]))


def evaluate_roi_exit(
    center: Optional[tuple[float, float]],
    safe_polygon: list[tuple[float, float]],
) -> tuple[bool, dict]:
    diag: dict = {"polygon_n": len(safe_polygon)}
    if center is None:
        diag["block"] = "no_center"
        return False, diag
    cx, cy = center
    diag["center"] = (round(cx), round(cy))
    if point_in_polygon((cx, cy), safe_polygon):
        diag["block"] = "inside_polygon"
        return False, diag
    return True, diag


def evaluate_fall(
    center: Optional[tuple[float, float]],
    prev_center: Optional[tuple[float, float]],
    dt: float,
    min_velocity_px_s: float,
) -> tuple[bool, dict]:
    diag: dict = {}
    if center is None or prev_center is None or dt <= 0:
        diag["block"] = "no_center_or_dt"
        return False, diag
    delta_y = center[1] - prev_center[1]  # 양수 = 아래로
    velocity = delta_y / dt
    diag["fall_velocity"] = round(velocity, 1)
    if velocity < min_velocity_px_s:
        diag["block"] = "velocity_too_low"
        return False, diag
    return True, diag


def evaluate_climbing(
    smoothed_wrist: Optional[tuple[float, float]],
    pose: Optional[Pose],
    safe_polygon: list[tuple[float, float]],
    rail_band_px: float,
    keypoint_conf_threshold: float,
    standing_y_margin: float,
) -> tuple[bool, dict]:
    diag: dict = {"polygon_n": len(safe_polygon)}
    if not safe_polygon:
        diag["block"] = "no_polygon"
        return False, diag
    if pose is None:
        diag["block"] = "no_pose"
        return False, diag
    if smoothed_wrist is None:
        diag["block"] = "no_wrist"
        return False, diag
    wx, wy = smoothed_wrist
    diag["wrist"] = (round(wx), round(wy))
    if not point_in_polygon((wx, wy), safe_polygon):
        diag["block"] = "wrist_outside_polygon"
        return False, diag
    edge_idx, dist = nearest_edge((wx, wy), safe_polygon)
    diag["rail_dist"] = round(dist, 1)
    if dist > rail_band_px:
        diag["block"] = "not_near_rail"
        return False, diag
    diag["rail_edge"] = EDGE_LABELS[edge_idx]

    shoulders = [pose.keypoints[k] for k in ("left_shoulder", "right_shoulder")
                 if pose.keypoints[k][2] >= keypoint_conf_threshold]
    hips = [pose.keypoints[k] for k in ("left_hip", "right_hip")
            if pose.keypoints[k][2] >= keypoint_conf_threshold]
    if not shoulders or not hips:
        diag["block"] = "shoulder_or_hip_invisible"
        return False, diag

    sy = sum(k[1] for k in shoulders) / len(shoulders)
    hy = sum(k[1] for k in hips) / len(hips)
    margin = hy - sy
    diag["standing_margin"] = round(margin, 1)
    if margin < standing_y_margin:
        diag["block"] = "not_standing"
        return False, diag
    return True, diag


def face_inside_person(face: Face, person: Person) -> bool:
    fx = (face.bbox[0] + face.bbox[2]) / 2
    fy = (face.bbox[1] + face.bbox[3]) / 2
    px1, py1, px2, py2 = person.bbox
    return px1 <= fx <= px2 and py1 <= fy <= py2


def evaluate_suffocation(
    person: Optional[Person],
    faces: list[Face],
    face_recently_seen: bool,
    person_was_in_roi: bool,
) -> tuple[bool, Optional[str], dict]:
    """두 경로로 질식/덮임을 감지.

    1. face_covered: person은 보이는데 그 안에 face가 없고, face가 최근 보였음.
    2. disappeared: person이 ROI 안에서 보이다가 ROI 이탈 없이 사라짐.
       (이불이 몸 전체를 덮으면 person 검출 자체가 실패 → 사라짐을 위험으로 해석)

    face_recently_seen / person_was_in_roi 는 호출자가 프레임 간 추적한다.
    person을 ROI 안에서 본 적 없으면(빈 방) 오탐 방지를 위해 판정하지 않는다.
    """
    diag: dict = {"person_n": 1 if person else 0, "face_n": len(faces)}
    if person is None:
        if not person_was_in_roi:
            diag["block"] = "no_person_not_in_roi"
            return False, None, diag
        return True, "disappeared", diag
    matching = [f for f in faces if face_inside_person(f, person)]
    diag["face_in_p"] = len(matching)
    if matching:
        diag["block"] = "face_detected"
        return False, None, diag
    if not face_recently_seen:
        diag["block"] = "face_never_seen"
        return False, None, diag
    return True, "face_covered", diag
