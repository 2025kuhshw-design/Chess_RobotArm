"""
라그랑주 방정식 기반 토크 계산 모듈
τ = M(q)·q̈ + C(q,q̇)·q̇ + G(q)
"""

import numpy as np
import warnings

# ─────────────────────────────────────────
# 하이퍼파라미터 / 상수
# ─────────────────────────────────────────
L1 = 0.140   # 상완 (m) — 실측 140mm
L2 = 0.155   # 전완 (m) — 실측 155mm
L3 = 0.075   # 손목~흡착기 (m) — 실측 75mm

m1 = 0.3     # 상완 질량 (kg)
m2 = 0.2     # 전완 질량 (kg)
m3 = 0.1     # 끝단 질량 (kg)

g  = 9.81    # 중력 가속도 (m/s²)

SERVO_LIMIT  = 1.27   # MG996R 최대 토크 (N·m)
MAX_RETRY    = 5      # 토크 초과 시 가속도 절반 최대 반복 횟수


# ─────────────────────────────────────────
# 함수 1: 관성 행렬 M(q)
# ─────────────────────────────────────────
def inertia_matrix(q: np.ndarray) -> np.ndarray:
    """
    3×3 관성 행렬 M(q)
    질량이 각 링크 끝단에 집중된다고 단순화.
    q = [q1, q2, q3]
    """
    q2, q3 = q[1], q[2]

    # 끝단 위치의 관성 기여 (링크 기준)
    # joint1 (베이스 Z축): 수평 평면 회전
    M11 = (m1 * L1**2
           + m2 * (L1**2 + 2*L1*L2*np.cos(q3) + L2**2)
           + m3 * (L1**2 + 2*L1*L2*np.cos(q3) + L2**2))

    M12 = 0.0
    M13 = 0.0

    # joint2 (어깨 Y축)
    M22 = (m1 * L1**2
           + m2 * (L1**2 + 2*L1*L2*np.cos(q3) + L2**2)
           + m3 * (L1**2 + 2*L1*L2*np.cos(q3) + L2**2))

    # joint2-joint3 교차 항
    M23 = (m2 * (L1*L2*np.cos(q3) + L2**2)
           + m3 * (L1*L2*np.cos(q3) + L2**2))

    # joint3 (팔꿈치 Y축)
    M33 = (m2 * L2**2 + m3 * L2**2)

    M = np.array([
        [M11, M12, M13],
        [M12, M22, M23],
        [M13, M23, M33],
    ])
    return M


# ─────────────────────────────────────────
# 함수 2: 코리올리·원심력 행렬 C(q, q_dot)
# ─────────────────────────────────────────
def coriolis_matrix(q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
    """
    3×3 코리올리·원심력 행렬 C(q, q̇)
    Christoffel 기호 기반 단순화 버전.
    """
    q3   = q[2]
    dq1, dq2, dq3 = q_dot

    h = -(m2 + m3) * L1 * L2 * np.sin(q3)

    C = np.array([
        [h*dq3,         h*dq3,         h*(dq2+dq3)],
        [-h*dq1,        0.0,           h*dq3       ],
        [-h*(dq1+dq2),  -h*dq2,        0.0         ],
    ])
    return C


# ─────────────────────────────────────────
# 함수 3: 중력 벡터 G(q)
# ─────────────────────────────────────────
def gravity_vector(q: np.ndarray) -> np.ndarray:
    """
    3×1 중력 항 G(q)
    joint1은 Z축 회전이므로 G[0] = 0.
    """
    q2, q3 = q[1], q[2]

    G1 = 0.0   # 베이스 Z축 회전 → 중력 영향 없음

    G2 = (g * L1 * np.cos(q2) * (m1 + m2 + m3)
          + g * L2 * np.cos(q2 + q3) * (m2 + m3))

    G3 = g * L2 * np.cos(q2 + q3) * (m2 + m3)

    return np.array([G1, G2, G3])


# ─────────────────────────────────────────
# 함수 4: 필요 토크 계산
# ─────────────────────────────────────────
def required_torque(q: np.ndarray,
                    q_dot: np.ndarray,
                    q_ddot: np.ndarray) -> np.ndarray:
    """
    τ = M(q)·q̈ + C(q,q̇)·q̇ + G(q)
    반환: tau 3×1 벡터 (N·m)
    """
    q      = np.asarray(q,      dtype=float)
    q_dot  = np.asarray(q_dot,  dtype=float)
    q_ddot = np.asarray(q_ddot, dtype=float)

    M = inertia_matrix(q)
    C = coriolis_matrix(q, q_dot)
    G = gravity_vector(q)

    tau = M @ q_ddot + C @ q_dot + G
    return tau


# ─────────────────────────────────────────
# 함수 5: 토크 한계 검증
# ─────────────────────────────────────────
def check_torque_feasibility(q, q_dot, q_ddot) -> tuple:
    """
    토크가 SERVO_LIMIT 이내인지 확인.
    초과 시 q_ddot를 절반으로 줄여 최대 MAX_RETRY회 재계산.
    반환: (feasible: bool, tau: ndarray, q_ddot_used: ndarray)
    """
    q      = np.asarray(q,      dtype=float)
    q_dot  = np.asarray(q_dot,  dtype=float)
    q_ddot = np.asarray(q_ddot, dtype=float)

    for attempt in range(MAX_RETRY):
        tau = required_torque(q, q_dot, q_ddot)
        if np.all(np.abs(tau) <= SERVO_LIMIT):
            return True, tau, q_ddot
        q_ddot = q_ddot * 0.5

    tau = required_torque(q, q_dot, q_ddot)
    warnings.warn(
        f"[Lagrange] 최대 {MAX_RETRY}회 재계산 후에도 토크 한계 초과: "
        f"tau={np.round(tau, 3)} N·m (한계={SERVO_LIMIT} N·m). "
        "속도를 더 낮추거나 궤적을 재계획하세요.",
        RuntimeWarning,
    )
    return False, tau, q_ddot


# ─────────────────────────────────────────
# 단독 실행: 토크 계산 검증
# ─────────────────────────────────────────
if __name__ == "__main__":
    import math

    print("=== 라그랑주 토크 계산 검증 ===")

    test_cases = [
        {"q": [0.0, 0.5, 0.8],   "q_dot": [0.1, 0.1, 0.1],  "q_ddot": [0.1, 0.1, 0.1]},
        {"q": [0.3, 0.8, 1.2],   "q_dot": [0.3, 0.2, 0.1],  "q_ddot": [0.5, 0.5, 0.5]},
        {"q": [1.0, 1.0, 1.3],   "q_dot": [0.5, 0.5, 0.5],  "q_ddot": [2.0, 2.0, 2.0]},  # 한계 초과 유도
    ]

    for i, tc in enumerate(test_cases):
        q      = np.array(tc["q"])
        q_dot  = np.array(tc["q_dot"])
        q_ddot = np.array(tc["q_ddot"])

        feasible, tau, q_ddot_used = check_torque_feasibility(q, q_dot, q_ddot)
        print(f"\n케이스 {i+1}:")
        print(f"  q      = {np.round(q,3)}")
        print(f"  q_dot  = {np.round(q_dot,3)}")
        print(f"  q_ddot = {np.round(q_ddot,3)} → 사용됨: {np.round(q_ddot_used,4)}")
        print(f"  tau    = {np.round(tau,4)} N·m")
        print(f"  한계({SERVO_LIMIT} N·m) 이내: {'✅ YES' if feasible else '❌ NO'}")

    print("\n✅ test_lagrange 완료")
