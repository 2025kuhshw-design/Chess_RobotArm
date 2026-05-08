"""
2단계 PyBullet RL 환경 (robot_full.urdf 사용)
env_simple.py 상속 + Domain Randomization 확대
"""

import os
import numpy as np
from sim.env_simple import ChessArmEnvSimple

# ─────────────────────────────────────────
# 하이퍼파라미터 / 상수 (1단계보다 확대)
# ─────────────────────────────────────────
URDF_FULL_PATH  = os.path.join(os.path.dirname(__file__), "..", "setup", "urdf", "robot_simple.urdf")

NOISE_LOW_FULL    = -0.05      # 1단계(±0.03)보다 넓은 DR, 관절당 최대 ~3.3cm 오차
NOISE_HIGH_FULL   =  0.05
FRICTION_LOW_FULL =  0.01
FRICTION_HIGH_FULL=  0.12      # 1단계(0.08)보다 넓은 DR, REACH_FINE=2cm 달성 가능 상한
MASS_VARIATION    =  0.20   # ±20%

BASE_MASSES = [0.12, 0.08, 0.04]  # 링크 기본 질량 (kg) — lagrange.py와 동기화


class ChessArmEnvFull(ChessArmEnvSimple):
    """
    1단계 환경 상속 → URDF + Domain Randomization 범위만 교체.
    """

    def __init__(self, render_mode=None, urdf_path=None):
        full_path = urdf_path or os.path.abspath(URDF_FULL_PATH)
        super().__init__(render_mode=render_mode, urdf_path=full_path)

    # ─────────────────────────────────────────
    # DR 파라미터 오버라이드 (reset() 호출 전에 실행됨)
    # 링크 질량 랜덤화 포함
    # ─────────────────────────────────────────
    def _set_dr_params(self):
        self._noise_scale = np.random.uniform(abs(NOISE_LOW_FULL), NOISE_HIGH_FULL)
        self._friction    = np.random.uniform(FRICTION_LOW_FULL, FRICTION_HIGH_FULL)
        self._randomize_link_masses()

    # ─────────────────────────────────────────
    # Domain Randomization: 에피소드마다 질량 변경
    # ─────────────────────────────────────────
    def _randomize_link_masses(self):
        p = self._p
        if self._robot_id is None:
            return
        for i, link_idx in enumerate(self._revolute_indices):
            base_mass = BASE_MASSES[i]
            variation = np.random.uniform(-MASS_VARIATION, MASS_VARIATION)
            new_mass  = base_mass * (1.0 + variation)
            p.changeDynamics(self._robot_id, link_idx, mass=new_mass)

    # ─────────────────────────────────────────
    # 실제 관절각 시뮬레이션 (확대된 노이즈)
    # ─────────────────────────────────────────
    def _simulate_real_joints(self, q_cmd):
        noise    = np.random.uniform(-self._noise_scale, self._noise_scale, size=3)
        friction = np.random.uniform(0, self._friction, size=3) * np.sign(q_cmd)
        q_real   = q_cmd + noise - friction
        return q_real.astype(np.float32)


# ─────────────────────────────────────────
# 단독 실행: 환경 동작 확인
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    env = ChessArmEnvFull()
    obs, _ = env.reset()
    print(f"[env_full] obs shape: {obs.shape}")
    print(f"초기 관측: {np.round(obs, 3)}")

    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  step {step+1}: dist={info['dist_cm']:.2f}cm  reward={reward:.2f}")

    env.close()
    print("✅ env_full 동작 확인 완료")
