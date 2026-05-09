"""
체스 로봇팔 쇼케이스 데모 (발표/영상 촬영용)
집기 → 이동 → 놓기 → 홈 복귀 사이클로 실제 체스 로봇 동작을 시각화한다.

python sim/showcase.py                  # 랜덤 이동 무한 반복
python sim/showcase.py --demo           # 체스 오프닝 시연 (e4 오프닝)
python sim/showcase.py --demo --repeat 2
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.ik_solver import (inverse_kinematics,
                              BOARD_ORIGIN_X, BOARD_ORIGIN_Y,
                              CELL_SIZE, PIECE_Z)

L1, L2, L3 = 0.140, 0.155, 0.075

URDF_MESH   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "setup", "urdf", "robot_full.urdf"))
URDF_SIMPLE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "setup", "urdf", "robot_simple.urdf"))

# 체스 오프닝 기물 이동 시퀀스: (출발칸, 도착칸)
DEMO_SEQUENCE = [
    ("e2", "e4"),
    ("e7", "e5"),
    ("g1", "f3"),
    ("b8", "c6"),
    ("f1", "c4"),
    ("f8", "c5"),
]

# 홈 자세: 판 위 중앙을 향해 팔을 들고 대기
# joint1=0(정면), joint2=0.6(어깨 올림), joint3=-1.0(팔꿈치 접음)
HOME_Q = np.array([0.0, 0.6, -1.0])

# 접근 높이: 기물 집기/놓기 전 정지 높이 (PIECE_Z보다 위)
APPROACH_LIFT = 0.08   # m


def square_to_xyz(square: str, z_offset: float = 0.0) -> np.ndarray:
    col = ord(square[0].lower()) - ord('a')
    row = int(square[1]) - 1
    x = BOARD_ORIGIN_X + col * CELL_SIZE + CELL_SIZE / 2
    y = BOARD_ORIGIN_Y + row * CELL_SIZE + CELL_SIZE / 2
    return np.array([x, y, PIECE_Z + z_offset], dtype=np.float32)


def compute_ik(target: np.ndarray) -> np.ndarray | None:
    try:
        q1, q2, q3 = inverse_kinematics(target[0], target[1], target[2], L1, L2, L3)
        return np.array([q1, q2, q3], dtype=np.float64)
    except ValueError:
        return None


def smooth_move(p, robot_id, rev_indices, q_from, q_to,
                duration: float = 1.0, fps: float = 60.0):
    """ease-in-out 보간으로 두 자세 사이를 부드럽게 이동."""
    n = max(int(duration * fps), 2)
    for k in range(1, n + 1):
        t = k / n
        t_ease = t * t * (3.0 - 2.0 * t)   # smoothstep
        q = q_from + t_ease * (q_to - q_from)
        for idx, angle in zip(rev_indices, q):
            p.resetJointState(robot_id, idx, float(angle))
        p.stepSimulation()
        time.sleep(1.0 / fps)


def run_showcase(moves: list, n_repeat: int = 0, urdf_path: str = None):
    """
    moves: [(from_sq, to_sq), ...] 형식의 이동 목록
    각 이동마다: 홈→출발접근→집기→이동→도착접근→놓기→홈 사이클
    """
    import pybullet as p
    import pybullet_data

    # ── URDF 선택 ──────────────────────────────────────────────
    if urdf_path is None:
        urdf_path = URDF_MESH if os.path.exists(URDF_MESH) else URDF_SIMPLE
    print(f"[URDF] {os.path.basename(urdf_path)}")

    # ── PyBullet 초기화 ──────────────────────────────────────
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)

    n_joints = p.getNumJoints(robot_id)
    rev_indices = [
        i for i in range(n_joints)
        if p.getJointInfo(robot_id, i)[2] == p.JOINT_REVOLUTE
    ][:3]

    # ── 카메라 ───────────────────────────────────────────────
    p.resetDebugVisualizerCamera(
        cameraDistance=0.85, cameraYaw=25,
        cameraPitch=-40, cameraTargetPosition=[0.20, 0.0, 0.05]
    )
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # ── 체스판 ───────────────────────────────────────────────
    for row in range(8):
        for col in range(8):
            x = BOARD_ORIGIN_X + col * CELL_SIZE + CELL_SIZE / 2
            y = BOARD_ORIGIN_Y + row * CELL_SIZE + CELL_SIZE / 2
            color = ([0.95, 0.90, 0.75, 1] if (row + col) % 2 == 0
                     else [0.35, 0.18, 0.05, 1])
            vis = p.createVisualShape(p.GEOM_BOX,
                halfExtents=[CELL_SIZE * 0.49, CELL_SIZE * 0.49, 0.002],
                rgbaColor=color)
            p.createMultiBody(0, -1, vis, [x, y, 0.001])

    # 마커: 빨간(출발), 초록(도착)
    vis_from = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[1, 0.2, 0.2, 0.9])
    vis_to   = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[0.2, 0.9, 0.2, 0.9])
    body_from = p.createMultiBody(0, -1, vis_from, [0, 0, -1])
    body_to   = p.createMultiBody(0, -1, vis_to,   [0, 0, -1])

    print("\n[조작법]  ← → ↑ ↓: 카메라 회전  마우스 드래그/스크롤: 시점  Ctrl+C: 종료\n")

    # ── 홈 자세로 초기화 ──────────────────────────────────────
    for idx, angle in zip(rev_indices, HOME_Q):
        p.resetJointState(robot_id, idx, float(angle))
    for _ in range(60):
        p.stepSimulation(); time.sleep(1/60)
    time.sleep(0.5)

    current_q = HOME_Q.copy()

    def go(target_xyz, duration=None):
        """target_xyz 위치로 IK 이동. 실패 시 False 반환."""
        nonlocal current_q
        q = compute_ik(target_xyz)
        if q is None:
            return False
        dist = float(np.max(np.abs(q - current_q)))
        dur  = duration if duration else max(dist / 1.2, 0.5)
        smooth_move(p, robot_id, rev_indices, current_q, q, duration=dur)
        current_q = q.copy()
        return True

    def go_home(duration=1.2):
        nonlocal current_q
        smooth_move(p, robot_id, rev_indices, current_q, HOME_Q, duration=duration)
        current_q = HOME_Q.copy()

    # ── 메인 루프 ─────────────────────────────────────────────
    repeat = 0
    try:
        while n_repeat == 0 or repeat < n_repeat:
            for from_sq, to_sq in moves:
                xyz_from_high = square_to_xyz(from_sq, z_offset=APPROACH_LIFT)
                xyz_from_low  = square_to_xyz(from_sq)
                xyz_to_high   = square_to_xyz(to_sq,   z_offset=APPROACH_LIFT)
                xyz_to_low    = square_to_xyz(to_sq)

                # 마커 표시
                p.resetBasePositionAndOrientation(body_from, xyz_from_low.tolist(), [0,0,0,1])
                p.resetBasePositionAndOrientation(body_to,   xyz_to_low.tolist(),   [0,0,0,1])

                print(f"  [{from_sq} → {to_sq}]", end="  ", flush=True)

                # 1. 출발칸 위로 접근
                print("접근", end="→", flush=True)
                if not go(xyz_from_high): continue
                time.sleep(0.2)

                # 2. 내려가서 집기
                print("집기", end="→", flush=True)
                if not go(xyz_from_low, duration=0.5): continue
                time.sleep(0.4)   # 흡착 대기

                # 3. 들어올리기
                print("상승", end="→", flush=True)
                go(xyz_from_high, duration=0.5)
                time.sleep(0.2)

                # 4. 도착칸 위로 이동
                print("이동", end="→", flush=True)
                go(xyz_to_high)
                time.sleep(0.2)

                # 5. 내려가서 놓기
                print("놓기", end="→", flush=True)
                go(xyz_to_low, duration=0.5)
                time.sleep(0.4)   # 해제 대기

                # 6. 들어올리기
                go(xyz_to_high, duration=0.5)
                time.sleep(0.2)

                # 7. 홈 복귀
                print("홈 복귀")
                go_home()
                time.sleep(0.6)

                # 마커 숨기기
                p.resetBasePositionAndOrientation(body_from, [0, 0, -1], [0,0,0,1])
                p.resetBasePositionAndOrientation(body_to,   [0, 0, -1], [0,0,0,1])

            repeat += 1
            if n_repeat == 0:
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[종료]")
    finally:
        p.disconnect(client)


def random_moves(n: int = 6) -> list:
    """체스판에서 랜덤 이동 쌍 생성."""
    cols = list("abcdefgh")
    moves = []
    for _ in range(n):
        from_sq = np.random.choice(cols) + str(np.random.randint(1, 9))
        to_sq   = np.random.choice(cols) + str(np.random.randint(1, 9))
        while to_sq == from_sq:
            to_sq = np.random.choice(cols) + str(np.random.randint(1, 9))
        moves.append((from_sq, to_sq))
    return moves


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="체스 로봇팔 쇼케이스 데모")
    parser.add_argument("--demo",   action="store_true", help="체스 오프닝 시연")
    parser.add_argument("--repeat", type=int, default=0, help="반복 횟수 (0=무한)")
    parser.add_argument("--urdf",   default=None,        help="사용할 URDF 경로")
    args = parser.parse_args()

    if args.demo:
        print("[데모] 체스 오프닝 시퀀스")
        run_showcase(DEMO_SEQUENCE, n_repeat=args.repeat or 0, urdf_path=args.urdf)
    else:
        print("[랜덤] Ctrl+C로 종료")
        while True:
            run_showcase(random_moves(6), n_repeat=1, urdf_path=args.urdf)
