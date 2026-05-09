"""
체스 로봇팔 쇼케이스 데모 (발표/영상 촬영용)
IK로 체스판 여러 칸을 순서대로 이동하며 실제 동작을 시각화한다.

python sim/showcase.py                    # 랜덤 순서로 무한 반복
python sim/showcase.py --sequence e2 e4 d7 d5   # 지정한 칸만 순서대로
python sim/showcase.py --demo             # 체스 오프닝 시연 (e4 오프닝)
"""

import os
import sys
import time
import argparse
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.ik_solver import (inverse_kinematics,
                              BOARD_ORIGIN_X, BOARD_ORIGIN_Y,
                              CELL_SIZE, PIECE_Z)

L1, L2, L3 = 0.140, 0.155, 0.075

# 발표용 체스 기물 이동 시퀀스 (실제 대국 느낌)
DEMO_SEQUENCE = [
    ("e2", "e4"),   # 백 e폰 전진
    ("e7", "e5"),   # 흑 e폰 전진
    ("g1", "f3"),   # 백 나이트
    ("b8", "c6"),   # 흑 나이트
    ("f1", "c4"),   # 백 비숍
    ("f8", "c5"),   # 흑 비숍
]

HOME_Q = np.array([0.0, 0.3, -0.6])   # 홈 자세 (체스판 위를 향한 대기 자세)


def square_to_xyz(square: str) -> np.ndarray:
    """체스 칸 이름(예: 'e4') → 월드 XYZ (m)"""
    col = ord(square[0].lower()) - ord('a')   # a=0 ~ h=7
    row = int(square[1]) - 1                   # 1=0 ~ 8=7
    x = BOARD_ORIGIN_X + col * CELL_SIZE + CELL_SIZE / 2
    y = BOARD_ORIGIN_Y + row * CELL_SIZE + CELL_SIZE / 2
    return np.array([x, y, PIECE_Z], dtype=np.float32)


def compute_ik(target: np.ndarray) -> np.ndarray | None:
    try:
        q1, q2, q3 = inverse_kinematics(target[0], target[1], target[2], L1, L2, L3)
        return np.array([q1, q2, q3], dtype=np.float64)
    except ValueError:
        return None


def smooth_move(p, robot_id, rev_indices, q_from, q_to,
                duration: float = 1.2, fps: float = 60.0):
    """두 관절각 사이를 부드럽게 보간하며 이동."""
    n = max(int(duration * fps), 2)
    for k in range(1, n + 1):
        t = k / n
        # ease-in-out (smoothstep)
        t_ease = t * t * (3.0 - 2.0 * t)
        q = q_from + t_ease * (q_to - q_from)
        for idx, angle in zip(rev_indices, q):
            p.resetJointState(robot_id, idx, float(angle))
        p.stepSimulation()
        time.sleep(1.0 / fps)


def run_showcase(sequence=None, n_repeat: int = 0):
    import pybullet as p
    import pybullet_data

    # ── PyBullet 초기화 ──────────────────────────────────────
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    urdf_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "setup", "urdf", "robot_simple.urdf")
    )
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)

    n_joints = p.getNumJoints(robot_id)
    rev_indices = [
        i for i in range(n_joints)
        if p.getJointInfo(robot_id, i)[2] == p.JOINT_REVOLUTE
    ][:3]

    # ── 카메라 & UI ──────────────────────────────────────────
    p.resetDebugVisualizerCamera(
        cameraDistance=0.80, cameraYaw=30,
        cameraPitch=-45, cameraTargetPosition=[0.20, 0.0, 0.05]
    )
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # ── 체스판 시각화 ─────────────────────────────────────────
    for row in range(8):
        for col in range(8):
            x = BOARD_ORIGIN_X + col * CELL_SIZE + CELL_SIZE / 2
            y = BOARD_ORIGIN_Y + row * CELL_SIZE + CELL_SIZE / 2
            color = ([0.95, 0.90, 0.75, 1] if (row + col) % 2 == 0
                     else [0.35, 0.18, 0.05, 1])
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[CELL_SIZE * 0.49, CELL_SIZE * 0.49, 0.002],
                rgbaColor=color,
            )
            p.createMultiBody(0, -1, vis, [x, y, 0.001])

    # 목표 마커 (빨간 구)
    target_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.012,
                                     rgbaColor=[1, 0.1, 0.1, 0.9])
    target_body = p.createMultiBody(0, -1, target_vis, [0, 0, -1])

    print("\n[조작법]")
    print("  ← → ↑ ↓  : 카메라 회전")
    print("  마우스 드래그/스크롤: 시점 변경")
    print("  Ctrl+C    : 종료\n")

    # ── 홈 자세로 이동 ────────────────────────────────────────
    for idx, angle in zip(rev_indices, HOME_Q):
        p.resetJointState(robot_id, idx, float(angle))
    for _ in range(30):
        p.stepSimulation()
        time.sleep(1 / 60)
    time.sleep(0.5)

    current_q = HOME_Q.copy()

    def move_to_square(sq: str, label: str = ""):
        nonlocal current_q
        xyz = square_to_xyz(sq)
        q_target = compute_ik(xyz)
        if q_target is None:
            print(f"  [경고] {sq}: IK 해 없음, 스킵")
            return

        # 마커 이동
        p.resetBasePositionAndOrientation(target_body, xyz.tolist(), [0, 0, 0, 1])

        dist_rad = float(np.max(np.abs(q_target - current_q)))
        duration = max(dist_rad / 1.5, 0.4)   # 최소 0.4초, 최대 이동각 비례

        if label:
            print(f"  → {sq:4s} {label}")
        smooth_move(p, robot_id, rev_indices, current_q, q_target, duration=duration)
        current_q = q_target.copy()
        time.sleep(0.4)   # 도달 후 잠깐 대기

    # ── 메인 루프 ─────────────────────────────────────────────
    repeat = 0
    try:
        while n_repeat == 0 or repeat < n_repeat:
            if sequence:
                # 지정 칸 순서대로
                for sq in sequence:
                    move_to_square(sq)
                # 홈 복귀
                print("  → 홈 복귀")
                smooth_move(p, robot_id, rev_indices, current_q, HOME_Q, duration=1.0)
                current_q = HOME_Q.copy()
                time.sleep(1.0)
            else:
                # 랜덤 칸으로 이동
                cols = list("abcdefgh")
                sq = np.random.choice(cols) + str(np.random.randint(1, 9))
                move_to_square(sq)

            repeat += 1

    except KeyboardInterrupt:
        print("\n[종료]")
    finally:
        p.disconnect(client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="체스 로봇팔 쇼케이스 데모")
    parser.add_argument("--sequence", nargs="+", default=None,
                        metavar="SQUARE",
                        help="이동할 칸 목록 (예: e2 e4 d7 d5)")
    parser.add_argument("--demo", action="store_true",
                        help="체스 오프닝 시연 시퀀스 실행")
    parser.add_argument("--repeat", type=int, default=0,
                        help="반복 횟수 (0=무한)")
    args = parser.parse_args()

    if args.demo:
        print("[데모] 체스 오프닝 시퀀스 시연")
        squares = [sq for pair in DEMO_SEQUENCE for sq in pair]
        run_showcase(sequence=squares, n_repeat=args.repeat or 3)
    elif args.sequence:
        run_showcase(sequence=args.sequence, n_repeat=args.repeat or 0)
    else:
        print("[랜덤] 체스판 랜덤 이동 (Ctrl+C로 종료)")
        run_showcase(sequence=None, n_repeat=0)
