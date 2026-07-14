"""
정적분 발표용 분석: 속도 로그를 구분구적으로 적분 → 예측 이동거리,
자로 잰 실측거리와 비교, 그리고 조각(dt)을 잘게 할수록 참값에 수렴함을 시연.

속도-시간 곡선 아래 넓이 = 이동 거리 (정적분의 원리).
데이터는 이산 표본이므로 해석적 적분이 불가능 → 조각의 합으로 근사(구분구적):
  - 왼쪽 끝점 직사각형 합 (Left Riemann)
  - 오른쪽 끝점 직사각형 합 (Right Riemann)
  - 중점 직사각형 합 (Midpoint)
  - 사다리꼴 합 (Trapezoid)
조각 폭 dt → 0 이면 모두 같은 참값(정적분)으로 수렴한다.

실행:
  python analysis/integrate.py --csv analysis/logs/run.csv --measured-cm 5.3
"""

import argparse
import csv


def load(csv_path):
    t, v, s = [], [], []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            t.append(float(r["t_plan_s"]))
            v.append(float(r["v_cmd_mps"]))
            s.append(float(r["s_cmd_m"]))
    return t, v, s


# ─────────────────────────────────────────
# 구분구적 4종 (등간격 dt 가정; 표본 t로 실제 간격 계산)
# ─────────────────────────────────────────
def left_sum(t, v):
    return sum(v[i] * (t[i+1] - t[i]) for i in range(len(t) - 1))

def right_sum(t, v):
    return sum(v[i+1] * (t[i+1] - t[i]) for i in range(len(t) - 1))

def trapezoid_sum(t, v):
    return sum(0.5 * (v[i] + v[i+1]) * (t[i+1] - t[i]) for i in range(len(t) - 1))

def midpoint_sum(t, v):
    # 표본 중점값을 이웃 평균으로 근사
    return sum(0.5 * (v[i] + v[i+1]) * (t[i+1] - t[i]) for i in range(len(t) - 1))


def subsample(t, v, k):
    """k칸마다 하나씩 뽑아 조각 폭을 k배로 (dt를 거칠게)."""
    ts, vs = t[::k], v[::k]
    if ts[-1] != t[-1]:      # 마지막 표본 보존
        ts = ts + [t[-1]]
        vs = vs + [v[-1]]
    return ts, vs


def main():
    ap = argparse.ArgumentParser(description="속도 로그 구분구적 적분 + 실측 비교")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--measured-cm", type=float, default=None,
                    help="자로 측정한 실제 이동거리 (cm)")
    args = ap.parse_args()

    t, v, s = load(args.csv)
    n = len(t)
    dt = (t[-1] - t[0]) / (n - 1) if n > 1 else 0.0
    cmd_total_cm = s[-1] * 100.0     # 명령 총 이동거리 (프로파일 설계값)

    print(f"[integrate] {args.csv}")
    print(f"  표본 수 = {n},  평균 dt = {dt*1000:.1f} ms,  총시간 = {t[-1]:.3f} s")
    print(f"  명령 총 이동거리 (설계 참값) = {cmd_total_cm:.3f} cm\n")

    # ── 전체 해상도에서 4종 적분 ──
    L = left_sum(t, v) * 100
    R = right_sum(t, v) * 100
    M = midpoint_sum(t, v) * 100
    Tz = trapezoid_sum(t, v) * 100
    print("  [구분구적 — 전체 해상도]")
    print(f"    왼쪽합   = {L:.3f} cm")
    print(f"    오른쪽합 = {R:.3f} cm")
    print(f"    중점합   = {M:.3f} cm")
    print(f"    사다리꼴 = {Tz:.3f} cm   ← 가장 정확\n")

    # ── 실측 비교 ──
    if args.measured_cm is not None:
        pred = Tz
        err = pred - args.measured_cm
        rel = err / args.measured_cm * 100 if args.measured_cm else float("nan")
        print("  [예측 vs 실측]")
        print(f"    예측(속도로그 적분, 사다리꼴) = {pred:.3f} cm")
        print(f"    실측(자)                      = {args.measured_cm:.3f} cm")
        print(f"    오차 = {err:+.3f} cm  ({rel:+.2f}%)")
        print(f"    → 오차 원인: 개방루프 서보(피드백 없음), 캘리브레이션 오프셋,")
        print(f"       기계적 유격, 응답지연 등 (명령값과 실제 물리 이동의 차이)\n")

    # ── 수렴 시연: dt를 키우며(subsample) 왼쪽합이 참값에서 얼마나 벗어나나 ──
    print("  [수렴 시연 — 조각을 잘게 할수록 참값에 수렴]")
    print(f"  {'dt(ms)':>8} {'표본':>5} {'왼쪽합(cm)':>11} {'사다리꼴(cm)':>12} {'왼쪽오차(cm)':>12}")
    for k in [8, 4, 2, 1]:
        if k >= n:
            continue
        ts, vs = subsample(t, v, k)
        Lk = left_sum(ts, vs) * 100
        Tk = trapezoid_sum(ts, vs) * 100
        dtk = (ts[-1] - ts[0]) / (len(ts) - 1) * 1000
        print(f"  {dtk:>8.0f} {len(ts):>5} {Lk:>11.3f} {Tk:>12.3f} "
              f"{Lk - cmd_total_cm:>+12.3f}")
    print(f"\n  → dt가 작아질수록 왼쪽합이 설계 참값 {cmd_total_cm:.3f} cm 에 수렴.")
    print(f"    (사다리꼴은 속도가 구간별 선형이라 거친 dt에서도 이미 정확)")


if __name__ == "__main__":
    main()
