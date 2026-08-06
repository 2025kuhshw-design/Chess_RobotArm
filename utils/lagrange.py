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

m1 = 0.12    # 상완 질량 (kg) — 3D프린트 링크 ~60g + MG996R 서보 ~55g
m2 = 0.08    # 전완 질량 (kg) — 3D프린트 링크 ~50g + 기구부 ~30g
m3 = 0.04    # 끝단 질량 (kg) — 흡착기 브래킷 ~40g

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

    # 질량은 각 링크 끝에 모여 있다고 본다.
    #   m1 → 팔꿈치 (어깨에서 L1)
    #   m2, m3 → 손목/끝단 (어깨에서 L1, 거기서 L2)
    # joint1 (베이스, **연직축** 회전)
    #   연직축 둘레의 관성은 '수평 투영 거리'의 제곱으로 정해진다.
    #   ⚠️ 예전에는 팔 평면 안의 거리(L1²+2L1L2cos q3+L2²)를 그대로 썼다.
    #      그건 q2=0(팔이 수평)일 때만 맞고, 팔을 세울수록 과대평가된다.
    #      (연직 성분 L2²sin²q3 만큼이 실제로는 베이스 관성에 기여하지 않는다)
    r1 = L1 * np.cos(q2)                            # 팔꿈치의 수평 거리
    r2 = L1 * np.cos(q2) + L2 * np.cos(q2 + q3)     # 끝단의 수평 거리
    M11 = m1 * r1**2 + (m2 + m3) * r2**2

    M12 = 0.0    # 연직축 회전과 평면 내 회전은 서로 직교 → 결합 없음
    M13 = 0.0

    # joint2 (어깨, 팔 평면 안의 회전) — 여기는 평면 내 거리가 맞다
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
    Christoffel 기호 기반 **단순화** 버전 — 팔 평면 안의 항만 반영한다.

    ⚠️ 한계: M11 이 q2 에도 의존하므로 엄밀하게는 베이스와 관련된 교차항이
       더 있다. 이 프로젝트에서는 required_torque 를 항상 q̇=0(정지 상태)으로
       부르기 때문에 C·q̇ = 0 이라 결과에 영향이 없다.
       빠르게 움직이며 토크를 따지려면 이 항을 제대로 유도해야 한다.
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
