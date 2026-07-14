"""
정적분 발표용 운동 데이터 수집 (실측 계측 코드).

말단(end-effector)을 A→B 직선으로 **사다리꼴 속도 프로파일**
(가속 → 등속 → 감속)로 이동시키면서, 고정 제어주기 dt마다
    (시각 t, 관절각 θ, 말단 xyz, 명령 속도 v)
를 CSV로 기록한다. 이 CSV가 곧 "속도 로그"다.

⚠️ 정직성 주의:
  - 로그의 위치·속도는 **명령(commanded) 값**이다. MG996R은 엔코더가
    없어(개방루프) 서보가 실제 도달한 각도를 읽을 수 없다.
  - 따라서 "실제 이동 거리"는 이 로그가 아니라 **자로 직접 측정**해서
    예측값(속도 로그의 적분)과 비교한다.
  - t_real 열은 perf_counter로 측정한 **실제 경과 시간**이다(타이밍 지연 포함).
    t_plan 열은 계획된 이상적 시각이다. 둘을 구분해 쓸 것.

실행 예:
  python analysis/collect_motion.py --port COM5 \
      --from-sq e2 --to-sq e4 --vmax 0.05 --accel 0.10 --dt 0.05 \
      --out analysis/logs/run_dt50.csv

  # 하드웨어 없이 명령 궤적만 뽑아보기(로깅 파이프라인 점검):
  python analysis/collect_motion.py --sim --from-sq e2 --to-sq e4 \
      --dt 0.05 --out analysis/logs/sim_dt50.csv
"""

import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.ik_solver import inverse_kinematics, chess_square_to_xyz
from hardware.arm_controller import RealArm


# ─────────────────────────────────────────
# 체스 표기(e2) → (col, row)
# ─────────────────────────────────────────
def parse_square(sq: str):
    sq = sq.strip().lower()
    col = ord(sq[0]) - ord('a')     # a=0 .. h=7
    row = int(sq[1]) - 1            # 1=0 .. 8=7
    if not (0 <= col <= 7 and 0 <= row <= 7):
        raise ValueError(f"잘못된 칸: {sq}")
    return col, row


# ─────────────────────────────────────────
# 사다리꼴 속도 프로파일
#   가속도 accel로 vmax까지 가속 → 등속 → 감속.
#   거리 D가 짧아 vmax에 못 닿으면 삼각형(가속→감속) 프로파일.
#   반환: s(t)[이동거리], v(t)[속력], T[총시간], v_peak
# ─────────────────────────────────────────
def build_profile(D: float, vmax: float, accel: float):
    t_acc = vmax / accel
    d_acc = 0.5 * accel * t_acc ** 2

    if 2 * d_acc >= D:
        # 삼각형: vmax 도달 전에 감속 시작
        t_acc = math.sqrt(D / accel)
        v_peak = accel * t_acc
        T = 2 * t_acc

        def s(t):
            if t <= 0:      return 0.0
            if t < t_acc:   return 0.5 * accel * t * t
            if t < T:       return D - 0.5 * accel * (T - t) ** 2
            return D

        def v(t):
            if t <= 0 or t >= T: return 0.0
            if t < t_acc:        return accel * t
            return accel * (T - t)

        return s, v, T, v_peak

    # 사다리꼴: 가속 → 등속 → 감속
    d_cruise = D - 2 * d_acc
    t_cruise = d_cruise / vmax
    T = 2 * t_acc + t_cruise

    def s(t):
        if t <= 0:                    return 0.0
        if t < t_acc:                 return 0.5 * accel * t * t
        if t < t_acc + t_cruise:      return d_acc + vmax * (t - t_acc)
        if t < T:                     return D - 0.5 * accel * (T - t) ** 2
        return D

    def v(t):
        if t <= 0 or t >= T:          return 0.0
        if t < t_acc:                 return accel * t
        if t < t_acc + t_cruise:      return vmax
        return accel * (T - t)

    return s, v, T, vmax


def lerp(a, b, u):
    return tuple(ai + (bi - ai) * u for ai, bi in zip(a, b))


def main():
    ap = argparse.ArgumentParser(description="정적분 발표용 운동 로그 수집")
    ap.add_argument("--port", type=str, default="COM5")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--sim", action="store_true", help="하드웨어 없이 명령 궤적만 생성")
    ap.add_argument("--from-sq", type=str, default="e2")
    ap.add_argument("--to-sq",   type=str, default="e4")
    ap.add_argument("--ax", type=float, help="시작 x(m) — 칸 대신 좌표 직접 지정")
    ap.add_argument("--ay", type=float)
    ap.add_argument("--az", type=float)
    ap.add_argument("--bx", type=float, help="도착 x(m)")
    ap.add_argument("--by", type=float)
    ap.add_argument("--bz", type=float)
    ap.add_argument("--vmax",  type=float, default=0.05, help="등속 목표 속력 (m/s)")
    ap.add_argument("--accel", type=float, default=0.10, help="가감속 가속도 (m/s^2)")
    ap.add_argument("--dt",    type=float, default=0.05, help="제어주기=샘플링 간격 (s)")
    ap.add_argument("--out",   type=str, default="analysis/logs/run.csv")
    ap.add_argument("--settle", type=float, default=1.5, help="시작점 정지 대기 (s)")
    args = ap.parse_args()

    # A, B 좌표 결정
    if args.ax is not None and args.bx is not None:
        A = (args.ax, args.ay, args.az)
        B = (args.bx, args.by, args.bz)
        label = f"({A})->({B})"
    else:
        fa = parse_square(args.from_sq)
        fb = parse_square(args.to_sq)
        A = chess_square_to_xyz(*fa)
        B = chess_square_to_xyz(*fb)
        label = f"{args.from_sq}->{args.to_sq}"

    D = math.dist(A, B)
    if D < 1e-6:
        print("시작점과 도착점이 같습니다."); return

    s_func, v_func, T, v_peak = build_profile(D, args.vmax, args.accel)
    n_steps = int(math.ceil(T / args.dt))

    print(f"[collect] 이동 {label}")
    print(f"  직선거리 D = {D*100:.2f} cm")
    print(f"  프로파일: vmax={args.vmax} m/s, accel={args.accel} m/s^2, "
          f"v_peak={v_peak:.4f} m/s, 총시간 T={T:.3f} s")
    print(f"  dt = {args.dt*1000:.0f} ms → 단계 수 {n_steps} (+1)")
    print(f"  ⚠️ 로그 위치/속도는 '명령값'. 실제 거리는 자로 측정할 것.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # 팔 준비
    arm = RealArm(port=args.port, baudrate=args.baud, sim=args.sim)

    # 시작점 A로 이동 후 정지 대기 (정지 상태에서 로그 시작)
    qa = inverse_kinematics(*A)
    arm.move(*qa)
    time.sleep(args.settle)

    rows = []
    t0 = time.perf_counter()
    for i in range(n_steps + 1):
        t_plan = i * args.dt
        # 계획 시각까지 실시간 페이싱
        while time.perf_counter() - t0 < t_plan:
            pass
        s_cmd = s_func(t_plan)
        v_cmd = v_func(t_plan)
        pos = lerp(A, B, s_cmd / D)
        q1, q2, q3 = inverse_kinematics(*pos)

        t_real = time.perf_counter() - t0
        arm.move(q1, q2, q3)     # 명령 전송 (sim이면 print)

        rows.append([
            i, round(t_plan, 4), round(t_real, 4),
            round(s_cmd, 6), round(v_cmd, 6),
            round(math.degrees(q1), 3), round(math.degrees(q2), 3),
            round(math.degrees(q3), 3),
            round(pos[0], 6), round(pos[1], 6), round(pos[2], 6),
        ])

    # 홈 복귀
    arm.home()
    arm.close()

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "t_plan_s", "t_real_s", "s_cmd_m", "v_cmd_mps",
                    "q1_deg", "q2_deg", "q3_deg", "x_m", "y_m", "z_m"])
        w.writerows(rows)

    total_real = rows[-1][2]
    print(f"\n[collect] 저장: {args.out}  ({len(rows)} 행)")
    print(f"  명령 총 이동거리 = {D*100:.2f} cm  (직선)")
    print(f"  실제 경과시간 t_real = {total_real:.3f} s (계획 T={T:.3f} s)")
    print(f"\n다음: 자로 실제 이동거리를 재고 →")
    print(f"  python analysis/integrate.py --csv {args.out} --measured-cm <측정값>")


if __name__ == "__main__":
    main()
