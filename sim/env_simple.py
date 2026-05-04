"""
1단계 PyBullet RL 환경 (단순 URDF 사용)
Sim-to-Real 오차 보정 정책 학습
"""

import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ─────────────────────────────────────────
# 하이퍼파라미터 / 상수
# ─────────────────────────────────────────
URDF_PATH    = os.path.join(os.path.dirname(__file__), "..", "setup", "urdf", "robot_simple.urdf")
L1, L2, L3  = 0.140, 0.155, 0.075   # 실측값 (m)
MAX_STEPS    = 150
DELTA_LIMIT  = 0.2        # 보정 델타 최대값 (rad)
REACH_GOOD   = 0.02       # 도달 인정 거리 (m) → +100
REACH_FINE   = 0.005      # 정밀 도달 거리 (m) → +200
TORQUE_LIMIT = 1.27       # N·m

# Sim-to-Real Gap 노이즈 범위
NOISE_LOW    = -0.05
NOISE_HIGH   =  0.05
DELAY_LOW    =  0.0
DELAY_HIGH   =  0.03
FRICTION_LOW =  0.01
FRICTION_HIGH=  0.15

# 체스판 설정 (랜덤 목표 생성용)
BOARD_ORIGIN_X = 0.015
BOARD_ORIGIN_Y = -0.115
CELL_SIZE      = 0.028
PIECE_Z        = 0.015


class ChessArmEnvSimple(gym.Env):
    """
    상태(12): [q1_ik, q2_ik, q3_ik,       -- 역기구학 이론 관절각
               q1_real, q2_real, q3_real,   -- 실제(노이즈 포함) 관절각
               tx, ty, tz,                  -- 목표 끝단 위치
               tau1, tau2, tau3]            -- 라그랑주 필요 토크 (N·m)
    행동(3): 각 관절 보정 델타 [-0.2, 0.2] rad
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, urdf_path=None):
        super().__init__()
        self.render_mode = render_mode
        self.urdf_path   = urdf_path or os.path.abspath(URDF_PATH)

        # 토크 범위: MG996R 최대 1.27 N·m 기준으로 ±2배 여유
        TAU_LIMIT = 3.0
        self.observation_space = spaces.Box(
            low  = np.array([-math.pi]*3 + [-math.pi]*3 + [-1.0, -1.0, 0.0] + [-TAU_LIMIT]*3, dtype=np.float32),
            high = np.array([ math.pi]*3 + [ math.pi]*3 + [ 1.0,  1.0, 1.0] + [ TAU_LIMIT]*3, dtype=np.float32),
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
        self._tau             = np.zeros(3)   # 라그랑주 토크
        self._step_count      = 0

        # 노이즈 파라미터 (에피소드마다 갱신)
        self._noise_scale = 0.0
        self._friction    = 0.0

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
        if self.render_mode == "human":
            self._pybullet_client = p.connect(p.GUI)
        else:
            self._pybullet_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        if os.path.exists(self.urdf_path):
            self._robot_id = p.loadURDF(self.urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
        else:
            # URDF가 없으면 자동 생성
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from setup.stl_to_urdf import generate_simple_urdf
            joint_positions = [[0.0,0.0,0.05],[0.0,0.0,0.33],[0.0,0.0,0.55],[0.0,0.0,0.63]]
            os.makedirs(os.path.dirname(self.urdf_path), exist_ok=True)
            generate_simple_urdf(joint_positions, self.urdf_path)
            self._robot_id = p.loadURDF(self.urdf_path, basePosition=[0, 0, 0], useFixedBase=True)

        self._n_joints = p.getNumJoints(self._robot_id)

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
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from utils.ik_solver import inverse_kinematics
        try:
            q1, q2, q3 = inverse_kinematics(target[0], target[1], target[2], L1, L2, L3)
            return np.array([q1, q2, q3], dtype=np.float32)
        except ValueError:
            return np.zeros(3, dtype=np.float32)

    # ─────────────────────────────────────────
    # 실제 관절각 시뮬레이션 (Sim-to-Real Gap)
    # ─────────────────────────────────────────
    def _simulate_real_joints(self, q_cmd):
        noise    = np.random.uniform(NOISE_LOW, NOISE_HIGH, size=3)
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
    # ─────────────────────────────────────────
    def _set_joint_angles(self, q):
        p = self._p
        joint_indices = [i for i in range(self._n_joints)
                         if p.getJointInfo(self._robot_id, i)[2] == p.JOINT_REVOLUTE]
        for idx, angle in zip(joint_indices[:3], q):
            p.resetJointState(self._robot_id, idx, angle)
        p.stepSimulation()

    # ─────────────────────────────────────────
    # reset
    # ─────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 에피소드마다 Sim-to-Real 파라미터 랜덤화
        self._noise_scale = np.random.uniform(abs(NOISE_LOW), NOISE_HIGH)
        self._friction    = np.random.uniform(FRICTION_LOW, FRICTION_HIGH)

        self._target_xyz  = self._random_target()
        self._q_ik        = self._compute_ik(self._target_xyz)
        self._q_real      = self._simulate_real_joints(self._q_ik)
        self._tau         = self._compute_tau(self._q_real)
        self._step_count  = 0
        self._prev_dist   = float("inf")

        self._set_joint_angles(self._q_real)

        obs = self._get_obs()
        return obs, {}

    # ─────────────────────────────────────────
    # 라그랑주 토크 계산
    # ─────────────────────────────────────────
    def _compute_tau(self, q_real) -> np.ndarray:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from utils.lagrange import required_torque
        try:
            tau = required_torque(q_real, np.zeros(3), np.array([0.1, 0.1, 0.1]))
            return np.clip(tau, -3.0, 3.0).astype(np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32)

    def _get_obs(self):
        return np.concatenate([self._q_ik, self._q_real, self._target_xyz, self._tau]).astype(np.float32)

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
        self._tau    = self._compute_tau(q_real)   # 라그랑주 토크 갱신
        self._set_joint_angles(q_real)
        self._step_count += 1

        # 보상 계산 (거리 기반 연속 보상 + 단계별 보너스)
        reward = -dist * 20.0               # 거리 페널티 강화

        if dist < 0.10:                     # 10cm 이내
            reward += 10.0
        if dist < 0.05:                     # 5cm 이내
            reward += 30.0
        if dist < REACH_GOOD:               # 2cm 이내
            reward += 100.0
        if dist < REACH_FINE:               # 0.5cm 이내
            reward += 200.0

        # 이전 스텝보다 가까워졌으면 추가 보상 (방향 학습 유도)
        prev_dist = getattr(self, "_prev_dist", dist)
        if dist < prev_dist:
            reward += 5.0
        self._prev_dist = dist

        # 라그랑주 토크 한계 페널티
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from utils.lagrange import SERVO_LIMIT
        if np.any(np.abs(self._tau) > SERVO_LIMIT):
            excess = np.sum(np.maximum(np.abs(self._tau) - SERVO_LIMIT, 0))
            reward -= 30.0 + excess * 5.0

        terminated = dist < REACH_FINE
        truncated  = self._step_count >= MAX_STEPS

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
