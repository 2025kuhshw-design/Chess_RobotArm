"""
1단계 PyBullet RL 환경 (단순 URDF 사용)
Sim-to-Real 오차 보정 정책 학습
"""

import os
import sys
import math
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 모듈 레벨에서 경로 추가 및 임포트 (매 호출마다 반복 방지)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.ik_solver import (inverse_kinematics,
                              BOARD_ORIGIN_X, BOARD_ORIGIN_Y,
                              CELL_SIZE, PIECE_Z)
from utils.lagrange import required_torque, SERVO_LIMIT

# ─────────────────────────────────────────
# 하이퍼파라미터 / 상수
# ─────────────────────────────────────────
URDF_PATH    = os.path.join(os.path.dirname(__file__), "..", "setup", "urdf", "robot_simple.urdf")
L1, L2, L3  = 0.140, 0.155, 0.075   # 실측값 (m)
MAX_STEPS    = 150
DELTA_LIMIT  = 0.2        # 보정 델타 최대값 (rad)
REACH_FINE    = 0.020     # 종료 임계값 (m): 2cm 도달 시 에피소드 성공 종료
TORQUE_LIMIT  = 1.27      # N·m
TAU_OBS_LIMIT = 3.0       # 관측값 토크 클리핑 범위 (N·m) — observation_space 및 main.py 공유

# q_real 관측값 범위: q_ik(±π) + action(±0.2) + noise(±0.03) + friction(0.08) → ±π + 0.35 여유
OBS_Q_LIMIT  = math.pi + 0.4

# Sim-to-Real Gap 노이즈 범위 — MG996R 기준: 반복 정밀도 ±2° ≈ ±0.035rad
NOISE_LOW    = -0.03      # ±1.7° (MG996R 반복 정밀도 이내)
NOISE_HIGH   =  0.03
FRICTION_LOW =  0.01
FRICTION_HIGH=  0.08      # 데스크탑 서보 부하 마찰 현실적 상한


class ChessArmEnvSimple(gym.Env):
    """
    상태(12): [q1_ik, q2_ik, q3_ik,       -- 역기구학 이론 관절각
               q1_real, q2_real, q3_real,   -- 실제(노이즈 포함) 관절각
               tx, ty, tz,                  -- 목표 끝단 위치
               tau1, tau2, tau3]            -- 라그랑주 필요 토크 (N·m)
    행동(3): 각 관절 보정 델타 [-0.2, 0.2] rad
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    _gui_open = False  # 프로세스 내 GUI 창은 하나만 허용

    def __init__(self, render_mode=None, urdf_path=None):
        super().__init__()
        self.render_mode = render_mode
        self.urdf_path   = urdf_path or os.path.abspath(URDF_PATH)

        self.observation_space = spaces.Box(
            low  = np.array([-math.pi]*3 + [-OBS_Q_LIMIT]*3 + [-1.0, -1.0, 0.0] + [-TAU_OBS_LIMIT]*3, dtype=np.float32),
            high = np.array([ math.pi]*3 + [ OBS_Q_LIMIT]*3 + [ 1.0,  1.0, 1.0] + [ TAU_OBS_LIMIT]*3, dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low  = np.full(3, -DELTA_LIMIT, dtype=np.float32),
            high = np.full(3,  DELTA_LIMIT, dtype=np.float32),
        )

        self._pybullet_client = None
        self._robot_id        = None
        self._target_xyz      = None
        self._q_ik            = np.zeros(3)
        self._q_real          = np.zeros(3)
        self._tau             = np.zeros(3)
        self._step_count      = 0
        self._noise_scale     = 0.0
        self._friction        = 0.0
        self._prev_dist       = float("inf")
        self._slow_render     = True   # S키 토글 상태 (에피소드 간 유지)
        self._s_was_down      = False  # S키 엣지 감지용

        self._init_pybullet()

    # ─────────────────────────────────────────
    # PyBullet 초기화
    # ─────────────────────────────────────────
    def _init_pybullet(self):
        try:
            import pybullet as p
            import pybullet_data
        except ImportError:
            raise ImportError("pip install pybullet 을 먼저 실행하세요.")

        self._p = p
        if self.render_mode == "human" and not ChessArmEnvSimple._gui_open:
            self._pybullet_client = p.connect(p.GUI)
            ChessArmEnvSimple._gui_open = True
            self._is_gui = True   # 실제 GUI 창을 가진 env만 True
        else:
            self._pybullet_client = p.connect(p.DIRECT)
            self._is_gui = False

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        if os.path.exists(self.urdf_path):
            self._robot_id = p.loadURDF(self.urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
        else:
            from setup.stl_to_urdf import generate_simple_urdf
            joint_positions = [[0.0,0.0,0.05],[0.0,0.0,0.33],[0.0,0.0,0.55],[0.0,0.0,0.63]]
            os.makedirs(os.path.dirname(self.urdf_path), exist_ok=True)
            generate_simple_urdf(joint_positions, self.urdf_path)
            self._robot_id = p.loadURDF(self.urdf_path, basePosition=[0, 0, 0], useFixedBase=True)

        self._n_joints  = p.getNumJoints(self._robot_id)
        # 회전 관절 인덱스를 한 번만 계산해 캐싱 (매 스텝 getJointInfo 반복 방지)
        self._revolute_indices = [
            i for i in range(self._n_joints)
            if p.getJointInfo(self._robot_id, i)[2] == p.JOINT_REVOLUTE
        ][:3]
        self._target_body = None

        if self._is_gui:
            self._setup_visualization()

    # ─────────────────────────────────────────
    # 시각화 초기 설정 (GUI 모드 전용)
    # ─────────────────────────────────────────
    def _setup_visualization(self):
        p = self._p

        # 체스판 위를 내려다보는 시점
        p.resetDebugVisualizerCamera(
            cameraDistance    = 0.75,
            cameraYaw         = 30,
            cameraPitch       = -50,
            cameraTargetPosition = [0.2, 0.0, 0.05],
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # 좌측 패널 숨김

        # 체스판 64칸 그리기
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

        # 목표 위치 마커 (빨간 구)
        target_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.008, rgbaColor=[1, 0.1, 0.1, 0.9]
        )
        self._target_body = p.createMultiBody(0, -1, target_vis, [0, 0, 0])

        print("\n[조작법]")
        print("  ← → ↑ ↓  : 카메라 회전 (마우스 왼쪽 드래그도 가능)")
        print("  S         : 시뮬레이션 속도 ON/OFF (느림↔빠름)")
        print("  마우스 우클릭 드래그: 카메라 패닝")
        print("  마우스 스크롤: 줌인/아웃\n")

    # ─────────────────────────────────────────
    # DR 파라미터 설정 (에피소드마다 갱신)
    # env_full에서 오버라이드하여 더 넓은 범위 적용
    # ─────────────────────────────────────────
    def _set_dr_params(self):
        self._noise_scale = np.random.uniform(abs(NOISE_LOW), NOISE_HIGH)
        self._friction    = np.random.uniform(FRICTION_LOW, FRICTION_HIGH)

    # ─────────────────────────────────────────
    # 랜덤 목표 생성 (체스판 64칸 중 랜덤)
    # ─────────────────────────────────────────
    def _random_target(self):
        col = np.random.randint(0, 8)
        row = np.random.randint(0, 8)
        x = BOARD_ORIGIN_X + col * CELL_SIZE + CELL_SIZE / 2
        y = BOARD_ORIGIN_Y + row * CELL_SIZE + CELL_SIZE / 2
        z = PIECE_Z
        return np.array([x, y, z], dtype=np.float32)

    # ─────────────────────────────────────────
    # IK 계산
    # ─────────────────────────────────────────
    def _compute_ik(self, target):
        try:
            q1, q2, q3 = inverse_kinematics(target[0], target[1], target[2], L1, L2, L3)
            return np.array([q1, q2, q3], dtype=np.float32)
        except ValueError:
            return np.zeros(3, dtype=np.float32)

    # ─────────────────────────────────────────
    # 실제 관절각 시뮬레이션 (Sim-to-Real Gap)
    # self._noise_scale 사용 — 에피소드마다 다른 노이즈 크기 적용
    # ─────────────────────────────────────────
    def _simulate_real_joints(self, q_cmd):
        noise    = np.random.uniform(-self._noise_scale, self._noise_scale, size=3)
        friction = np.random.uniform(0, self._friction, size=3) * np.sign(q_cmd)
        q_real   = q_cmd + noise - friction
        return q_real.astype(np.float32)

    # ─────────────────────────────────────────
    # 순기구학으로 끝단 위치 계산
    # ─────────────────────────────────────────
    def _forward_kinematics(self, q):
        q1, q2, q3 = q
        r  = L1 * math.cos(q2) + L2 * math.cos(q2 + q3)
        x  = r * math.cos(q1)
        y  = r * math.sin(q1)
        z  = L1 * math.sin(q2) + L2 * math.sin(q2 + q3) - L3
        return np.array([x, y, z], dtype=np.float32)

    # ─────────────────────────────────────────
    # PyBullet 관절 위치 설정
    # slow 모드: 현재→목표 각도를 보간해 서보가 실제로 움직이는 것처럼 표현
    # ─────────────────────────────────────────
    def _set_joint_angles(self, q):
        p = self._p
        rev_indices = self._revolute_indices

        if self._is_gui and self._slow_render:
            # 현재 관절각 읽기
            curr = np.array([p.getJointState(self._robot_id, idx)[0]
                             for idx in rev_indices], dtype=np.float64)
            target = np.array(q, dtype=np.float64)
            # MG996R 6 rad/s 기준 → 최대 이동각 / 속도 = 소요 시간
            max_delta = float(np.max(np.abs(target - curr)))
            duration  = max(max_delta / 6.0, 0.05)   # 최소 50ms
            n_frames  = max(int(duration / 0.016), 3)  # ~60fps
            for k in range(1, n_frames + 1):
                t = k / n_frames
                q_interp = curr + t * (target - curr)
                for idx, angle in zip(rev_indices, q_interp):
                    p.resetJointState(self._robot_id, idx, float(angle))
                p.stepSimulation()
                time.sleep(0.016)
        else:
            for idx, angle in zip(rev_indices, q):
                p.resetJointState(self._robot_id, idx, angle)
            p.stepSimulation()

    # ─────────────────────────────────────────
    # 라그랑주 토크 계산
    # ─────────────────────────────────────────
    def _compute_tau(self, q_real) -> np.ndarray:
        try:
            tau = required_torque(q_real, np.zeros(3), np.array([0.1, 0.1, 0.1]))
            return np.clip(tau, -TAU_OBS_LIMIT, TAU_OBS_LIMIT).astype(np.float32)
        except Exception as e:
            print(f"[WARN] _compute_tau 실패 (q={np.round(q_real,3)}): {e}")
            return np.zeros(3, dtype=np.float32)

    def _get_obs(self):
        return np.concatenate([self._q_ik, self._q_real, self._target_xyz, self._tau]).astype(np.float32)

    # ─────────────────────────────────────────
    # reset
    # ─────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # DR 파라미터를 먼저 설정해야 _simulate_real_joints가 올바른 값 사용
        self._set_dr_params()

        self._target_xyz  = self._random_target()
        self._q_ik        = self._compute_ik(self._target_xyz)
        self._q_real      = self._simulate_real_joints(self._q_ik)
        self._tau         = self._compute_tau(self._q_real)
        self._step_count  = 0
        self._prev_dist   = float("inf")

        # 목표 마커를 새 위치로 이동 (GUI env만)
        if self._is_gui and self._target_body is not None:
            self._p.resetBasePositionAndOrientation(
                self._target_body, self._target_xyz.tolist(), [0, 0, 0, 1]
            )
        self._s_was_down = False  # S키 엣지 감지 초기화

        self._set_joint_angles(self._q_real)

        obs = self._get_obs()
        return obs, {}

    # ─────────────────────────────────────────
    # GUI 키보드 처리 (S: 속도토글 / 화살표: 카메라 회전)
    # ─────────────────────────────────────────
    def _handle_keys(self):
        p    = self._p
        keys = p.getKeyboardEvents()

        # S키: 속도 토글 (엣지 감지)
        s_down = bool(keys.get(ord('s'), 0) & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN))
        if s_down and not self._s_was_down:
            self._slow_render = not self._slow_render
            print(f"[렌더] 속도 제한: {'ON (느림)' if self._slow_render else 'OFF (빠름)'}")
        self._s_was_down = s_down

        # 화살표: 카메라 궤도 회전 (마우스 드래그 대체)
        cam   = p.getDebugVisualizerCamera()
        yaw, pitch, dist, target = cam[8], cam[9], cam[10], list(cam[11])
        step  = 3.0
        moved = False
        if keys.get(p.B3G_LEFT_ARROW,  0) & p.KEY_IS_DOWN: yaw   -= step; moved = True
        if keys.get(p.B3G_RIGHT_ARROW, 0) & p.KEY_IS_DOWN: yaw   += step; moved = True
        if keys.get(p.B3G_UP_ARROW,    0) & p.KEY_IS_DOWN: pitch  = min(pitch + step, -5);  moved = True
        if keys.get(p.B3G_DOWN_ARROW,  0) & p.KEY_IS_DOWN: pitch  = max(pitch - step, -89); moved = True
        if moved:
            p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

    # ─────────────────────────────────────────
    # step
    # ─────────────────────────────────────────
    def step(self, action):
        action = np.clip(action, -DELTA_LIMIT, DELTA_LIMIT)

        q_cmd   = self._q_ik + action
        q_real  = self._simulate_real_joints(q_cmd)
        ee_pos  = self._forward_kinematics(q_real)
        dist    = float(np.linalg.norm(ee_pos - self._target_xyz))

        self._q_real = q_real
        self._tau    = self._compute_tau(q_real)
        self._set_joint_angles(q_real)
        self._step_count += 1

        # 보상 계산
        reward = -dist * 20.0
        if dist < self._prev_dist:
            reward += 3.0
        self._prev_dist = dist
        reward -= 0.5  # 스텝 패널티 (맴돌기 방지)

        if np.any(np.abs(self._tau) > SERVO_LIMIT):
            excess = np.sum(np.maximum(np.abs(self._tau) - SERVO_LIMIT, 0))
            reward -= 30.0 + excess * 5.0

        terminated = dist < REACH_FINE
        truncated  = self._step_count >= MAX_STEPS

        if terminated:
            reward += 300.0 + (MAX_STEPS - self._step_count) * 3.0

        if self._is_gui:  # GUI 창이 있는 env[0]만 처리 (나머지는 풀속도 유지)
            self._handle_keys()

        info = {"dist_m": dist, "dist_cm": dist * 100, "target": self._target_xyz}
        obs  = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            p = self._p
            width, height = 640, 480
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.2, 0, 0.3],
                distance=0.8, yaw=45, pitch=-30, roll=0, upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=width/height, nearVal=0.1, farVal=10)
            _, _, rgb, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix)
            return np.array(rgb)[:, :, :3]

    def close(self):
        if self._pybullet_client is not None:
            if self._is_gui:
                ChessArmEnvSimple._gui_open = False  # 다음 env 생성 시 GUI 재사용 허용
            self._p.disconnect(self._pybullet_client)
            self._pybullet_client = None


# ─────────────────────────────────────────
# 단독 실행: 환경 동작 확인
# ─────────────────────────────────────────
if __name__ == "__main__":
    env = ChessArmEnvSimple()
    obs, _ = env.reset()
    print(f"obs shape: {obs.shape}")
    print(f"초기 관측: {np.round(obs, 3)}")

    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  step {step+1}: dist={info['dist_cm']:.2f}cm  reward={reward:.2f}")
        if terminated or truncated:
            break

    env.close()
    print(f"\n총 보상: {total_reward:.2f}")
    print("✅ env_simple 동작 확인 완료")
