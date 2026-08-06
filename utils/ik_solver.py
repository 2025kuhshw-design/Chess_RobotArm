"""
역기구학(IK) / 순기구학(FK) 모듈
3축 로봇팔: joint1(Z축 베이스), joint2(Y축 어깨), joint3(Y축 팔꿈치)
"""

import math
import os
import json
import numpy as np

# ─────────────────────────────────────────
# 하이퍼파라미터 / 상수
# ─────────────────────────────────────────
L1_DEFAULT = 0.140   # 상완 (m) — 실측 140mm
L2_DEFAULT = 0.155   # 전완 (m) — 실측 155mm
L3_DEFAULT = 0.075   # 손목~흡착기 (m) — 실측 75mm

# 어깨(joint2) 관절이 체스판 표면보다 얼마나 높은가 (m).
# ⚠️ 반드시 실측할 것. 이 값이 틀리면 흡착기가 기물 높이에 정확히 닿지 않는다.
#    (예전 코드는 이 항이 아예 없어 어깨가 보드면과 같은 높이라고 가정했다)
SHOULDER_HEIGHT = 0.089   # 실측: 테이블 위 9cm - 보드두께 1mm

# 팔꿈치 자세 선택: +1 = elbow-up, -1 = elbow-down.
# 2링크 IK는 같은 목표점에 두 가지 해가 있다. 실제 팔의 관절이 꺾이는 방향과
# 서보 가동범위에 맞는 쪽을 골라야 한다. 한쪽이 서보 범위를 벗어나면 반대로.
ELBOW_SIGN = -1

# 체스판 위치 조정 (팔 길이 295mm 기준, 전체 칸 도달 가능하도록)
# 체스판 중심을 원점에서 x=130mm 거리에 배치
# 가장 먼 칸까지 거리 ≈ 270mm < 유효 도달 283mm
BOARD_ORIGIN_X = 0.110   # 체스판 근단(랭크8 쪽) x (m) — 도달범위 최적화 배치
BOARD_ORIGIN_Y = -0.060  # 체스판 우단(파일a 쪽) y (m) — 베이스 가동범위에 맞춰 편향
CELL_SIZE      = 0.029125  # 한 칸 크기 (m) — 23.3cm ÷ 8칸
PIECE_Z        = 0.0045  # 기물 두께 (m) — 자작 원판/사각판 4~5mm
                         # ⚠️ 기물을 바꾸면 반드시 다시 잴 것. 이 값이 실제보다
                         #    크면 흡착컵이 기물에 닿기 전에 멈춰 진공이 안 걸린다.
                         #    (이전 기물은 7mm였다)
LIFT_DEFAULT   = 0.04    # 안전 접근 높이 (m) — 기물이 7mm라 4cm면 충분


# ─────────────────────────────────────────
# 함수 1: 역기구학
# ─────────────────────────────────────────
def inverse_kinematics(x: float, y: float, z: float,
                       L1: float = L1_DEFAULT,
                       L2: float = L2_DEFAULT,
                       L3: float = L3_DEFAULT) -> tuple:
    """
    목표 끝단 좌표 (x, y, z) → 관절각 (q1, q2, q3) [라디안]
    L3(손목~흡착기)는 z축 방향으로 수직 하강 가정 → 목표 z에서 L3 제외.
    z는 체스판 표면 기준이며, 어깨 관절 기준으로는 SHOULDER_HEIGHT만큼 낮다.
    """
    # 어깨 관절 기준 손목 높이 (흡착기 오프셋 제거 + 어깨 높이 보정)
    wz = z + L3 - SHOULDER_HEIGHT

    q1 = math.atan2(y, x)

    r = math.sqrt(x**2 + y**2)

    reach = math.sqrt(r**2 + wz**2)
    if reach > L1 + L2:
        raise ValueError(
            f"도달 불가능한 좌표: ({x:.3f}, {y:.3f}, {z:.3f}), "
            f"거리={reach:.3f}m > 팔 길이={L1+L2:.3f}m"
        )
    if reach < abs(L1 - L2):
        raise ValueError(
            f"도달 불가능한 좌표: ({x:.3f}, {y:.3f}, {z:.3f}), "
            f"거리={reach:.3f}m < |L1-L2|={abs(L1-L2):.3f}m"
        )

    cos_q3 = (r**2 + wz**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_q3 = max(-1.0, min(1.0, cos_q3))   # 수치 오차 클리핑
    q3 = ELBOW_SIGN * math.acos(cos_q3)    # elbow-up(+1) / elbow-down(-1)

    q2 = math.atan2(wz, r) - math.atan2(L2 * math.sin(q3), L1 + L2 * math.cos(q3))

    return (q1, q2, q3)


# ─────────────────────────────────────────
# 함수 2: 순기구학 (IK 검증용)
# ─────────────────────────────────────────
def forward_kinematics(q1: float, q2: float, q3: float,
                       L1: float = L1_DEFAULT,
                       L2: float = L2_DEFAULT,
                       L3: float = L3_DEFAULT) -> tuple:
    """관절각 (q1, q2, q3) [라디안] → 끝단 (x, y, z) [m]
    z는 체스판 표면 기준 (어깨 높이 SHOULDER_HEIGHT를 더해서 환산)."""
    # 수평 거리 (XY 평면 투영)
    r = L1 * math.cos(q2) + L2 * math.cos(q2 + q3)

    x = r * math.cos(q1)
    y = r * math.sin(q1)
    z = SHOULDER_HEIGHT + L1 * math.sin(q2) + L2 * math.sin(q2 + q3) - L3

    return (x, y, z)


# ─────────────────────────────────────────
# 함수 3: 체스 칸 → 월드 xyz
# ─────────────────────────────────────────
# ─────────────────────────────────────────
# 실측 보정 (hardware/calibrate_board.py 가 만든 파일)
# ─────────────────────────────────────────
# 로봇이 실제로 짚는 위치와 목표의 차이를 1차식으로 모델링한 계수.
# 파일이 없으면 보정 없이 동작한다(기존과 동일).
_BOARD_FIT_PATH = os.path.join(os.path.dirname(__file__), "..", "vision", "board_fit.json")
_BOARD_FIT = None
try:
    if os.path.exists(_BOARD_FIT_PATH):
        with open(_BOARD_FIT_PATH) as _f:
            _BOARD_FIT = json.load(_f)
        print(f"[IK] 보드 실측 보정 적용 ({_BOARD_FIT.get('mode')}, "
              f"{_BOARD_FIT.get('n_samples')}점)")
except Exception as _e:      # 손상된 파일이 조용히 무시되지 않도록 알린다
    print(f"[IK] board_fit.json 읽기 실패 → 보정 없이 진행: {_e}")
    _BOARD_FIT = None


def apply_board_fit(x: float, y: float) -> tuple:
    """실측 보정을 적용한 (x, y). 보정 파일이 없으면 그대로 반환.

    측정된 오차(실제 - 목표)를 목표 좌표에서 빼주면 실제가 목표에 맞는다.
    """
    if _BOARD_FIT is None:
        return (x, y)
    cx, cy = _BOARD_FIT["coef_x"], _BOARD_FIT["coef_y"]
    dx = cx[0] * x + cx[1] * y + cx[2]
    dy = cy[0] * x + cy[1] * y + cy[2]
    return (x - dx, y - dy)


def chess_square_to_xyz(col: int, row: int) -> tuple:
    """
    체스 칸 (col=0~7, row=0~7) → 월드 좌표 (x, y, z) [m]
    col: 파일 (a=0 ~ h=7)
    row: 랭크 (1=0 ~ 8=7)

    배치 규약 — 로봇(BLACK)과 사람(WHITE)이 보드를 사이에 두고 마주 앉는다.
      · 랭크는 로봇↔사람 방향(x). 랭크8(row=7)이 로봇 쪽(x 최소),
        랭크1(row=0)이 사람 쪽(x 최대).
        → 흑 기물은 로봇 앞, 백 기물은 사람 앞에 놓이고 폰 전진 방향이 맞는다.
      · 파일은 좌우 방향(y). 파일a(col=0)가 로봇 기준 오른쪽(y 최소).
        → 사람(백) 시점에서 a1이 왼쪽 앞 = 체스 관례와 일치.
    """
    x = BOARD_ORIGIN_X + (7 - row) * CELL_SIZE + CELL_SIZE / 2
    y = BOARD_ORIGIN_Y + col * CELL_SIZE + CELL_SIZE / 2
    x, y = apply_board_fit(x, y)      # 실측 보정 (파일 없으면 그대로)
    z = PIECE_Z
    return (x, y, z)


# ─────────────────────────────────────────
# 함수 4: 안전 접근 좌표 (기물 위 lift 높이)
# ─────────────────────────────────────────
def safe_approach_xyz(col: int, row: int, lift: float = LIFT_DEFAULT) -> tuple:
    """목표 칸 바로 위 lift 높이의 안전 접근 좌표 반환."""
    x, y, z = chess_square_to_xyz(col, row)
    return (x, y, z + lift)


# ─────────────────────────────────────────
# 단독 실행: IK → FK 왕복 검증
# ─────────────────────────────────────────
if __name__ == "__main__":
    test_targets = [
        (0.20, 0.05, 0.10),
        (0.30, -0.10, 0.05),
        (0.15, 0.10, 0.20),
    ]

    print("=== IK → FK 왕복 검증 ===")
    all_pass = True
    for target in test_targets:
        x, y, z = target
        try:
            q1, q2, q3 = inverse_kinematics(x, y, z)
            fx, fy, fz = forward_kinematics(q1, q2, q3)
            err = math.sqrt((x - fx)**2 + (y - fy)**2 + (z - fz)**2) * 1000
            status = "PASS" if err < 1.0 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  목표({x:.3f},{y:.3f},{z:.3f}) | "
                  f"q=({math.degrees(q1):.1f}°,{math.degrees(q2):.1f}°,{math.degrees(q3):.1f}°) | "
                  f"오차={err:.3f}mm [{status}]")
        except ValueError as e:
            print(f"  [{x},{y},{z}] → {e}")
            all_pass = False

    print()
    print("=== 체스 칸 좌표 변환 ===")
    for col, row in [(0, 0), (3, 3), (7, 7)]:
        xyz = chess_square_to_xyz(col, row)
        safe = safe_approach_xyz(col, row)
        print(f"  칸({col},{row}) → {xyz}  안전접근→ {safe}")

    print()
    print("✅ test_ik 완료" if all_pass else "❌ test_ik 실패 (오차 > 1mm)")
