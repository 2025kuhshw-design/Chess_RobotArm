"""
보드 위치 실측 보정 — 로봇이 실제로 어디를 짚는지 재서 오차를 없앤다.

카메라를 쓰지 않는다. 체스판에 인쇄된 격자 자체를 자로 쓴다(한 칸 2.91cm라
눈으로 2~3mm까지 읽힌다). 카메라로 재면 카메라 캘리브레이션 오차가 측정에
섞이므로 오히려 부정확하다.

절차:
  1) 팔을 여러 칸으로 보낸다 (기물 없이, 흡착컵만 내려감)
  2) 흡착컵이 그 칸 중심에서 얼마나 벗어났는지 mm로 입력한다
  3) 최소제곱으로 보정식(어파인)을 구해 파일로 저장한다

구해지는 것: 전체 밀림(offset) · 축척(scale) · 회전/기울어짐(skew).
이 세 가지가 계통 오차의 대부분이라 6~9점이면 충분하다.

실행:
  python hardware/calibrate_board.py --port COM5 --step 1 --step-delay 0.1
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import hardware.arm_controller as ac
from hardware.arm_controller import RealArm, _safe_lift, interactive_startup
from utils.ik_solver import inverse_kinematics, chess_square_to_xyz, safe_approach_xyz

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "vision", "board_fit.json")

# 측정할 칸 — 도달 가능 영역(랭크3~8)에 고루 퍼지도록
DEFAULT_SQUARES = ["a8", "d8", "h8", "a5", "d5", "h5", "b3", "e3", "f6"]


def parse_square(sq):
    sq = sq.strip().lower()
    if len(sq) != 2 or not ('a' <= sq[0] <= 'h') or not ('1' <= sq[1] <= '8'):
        return None
    return (ord(sq[0]) - ord('a'), int(sq[1]) - 1)


def fit_affine(samples):
    """samples = [(x_target, y_target, dx_err, dy_err), ...] (m 단위)

    오차를 목표 좌표의 1차식으로 모델링한다:
        dx = a*x + b*y + c
        dy = d*x + e*y + f
    최소제곱으로 6개 계수를 구한다. 점이 3개 미만이면 평균 오프셋만.
    """
    n = len(samples)
    if n == 0:
        return None
    if n < 3:
        cx = sum(s[2] for s in samples) / n
        cy = sum(s[3] for s in samples) / n
        return {"mode": "offset", "coef_x": [0.0, 0.0, cx], "coef_y": [0.0, 0.0, cy],
                "n_samples": n}
    A = np.array([[s[0], s[1], 1.0] for s in samples])
    bx = np.array([s[2] for s in samples])
    by = np.array([s[3] for s in samples])
    cx, *_ = np.linalg.lstsq(A, bx, rcond=None)
    cy, *_ = np.linalg.lstsq(A, by, rcond=None)
    return {"mode": "affine", "coef_x": list(map(float, cx)),
            "coef_y": list(map(float, cy)), "n_samples": n}


def residuals(fit, samples):
    """보정 후 남는 오차 (mm)."""
    out = []
    for x, y, dx, dy in samples:
        px = fit["coef_x"][0]*x + fit["coef_x"][1]*y + fit["coef_x"][2]
        py = fit["coef_y"][0]*x + fit["coef_y"][1]*y + fit["coef_y"][2]
        out.append(((dx - px) * 1000, (dy - py) * 1000))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=str, default="COM5")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--step-delay", type=float, default=None)
    ap.add_argument("--squares", nargs="*", default=DEFAULT_SQUARES)
    ap.add_argument("--no-startup", action="store_true")
    args = ap.parse_args()

    arm = RealArm(port=args.port, sim=args.sim, auto_home=False,
                  ramp_step=args.step, ramp_delay=args.step_delay)
    if not args.sim and not args.no_startup:
        if not interactive_startup(arm):
            arm.close(); print("[calib] 기동 중단."); return

    print("\n" + "=" * 62)
    print(" 보드 위치 실측 보정")
    print("=" * 62)
    print(" 각 칸으로 팔을 보냅니다. 흡착컵 끝이 그 칸 '중심'에서")
    print(" 얼마나 벗어났는지 mm로 입력하세요. 한 칸 = 29.1mm 이므로")
    print(" 격자를 눈금 삼아 읽으면 됩니다.")
    print()
    print("  부호 규칙 (로봇에서 보드를 바라본 기준):")
    print("    앞(로봇에서 멀어짐) = +x    뒤(로봇 쪽) = -x")
    print("    왼쪽                = +y    오른쪽      = -y")
    print("  예) 흡착컵이 칸 중심보다 5mm 앞, 3mm 오른쪽 → '5 -3'")
    print()
    print("  s = 이 칸 건너뛰기,  q = 측정 종료하고 피팅")
    print(" 🛑 이상 동작 시 즉시 Ctrl+C\n")

    samples = []
    for name in args.squares:
        sq = parse_square(name)
        if not sq:
            print(f"  '{name}' 칸 이름 오류 — 건너뜀"); continue
        try:
            x, y, _ = chess_square_to_xyz(*sq)
            arm.move(*inverse_kinematics(*safe_approach_xyz(*sq, _safe_lift(*sq))))
            arm.move(*inverse_kinematics(*ac.touch_xyz(*sq)))
        except ValueError as e:
            print(f"  {name}: 도달 불가 — 건너뜀 ({e})"); continue

        print(f"\n  [{name}] 목표 x={x*100:.1f}cm y={y*100:+.1f}cm 로 이동 완료")
        try:
            ans = input(f"    벗어난 양 (앞뒤 좌우, mm) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if ans == "q":
            break
        if ans == "s" or not ans:
            print("    건너뜀"); continue
        try:
            parts = ans.split()
            dx = float(parts[0]) / 1000.0
            dy = float(parts[1]) / 1000.0
        except (ValueError, IndexError):
            print("    입력 오류 — 건너뜀"); continue
        samples.append((x, y, dx, dy))
        print(f"    기록: dx={dx*1000:+.0f}mm dy={dy*1000:+.0f}mm  (총 {len(samples)}점)")

    # 안전하게 park로
    try:
        arm.home()
    except Exception:
        pass
    arm.close()

    if not samples:
        print("\n[calib] 측정값이 없어 저장하지 않습니다."); return

    fit = fit_affine(samples)
    res = residuals(fit, samples)
    import math
    before = math.sqrt(sum((s[2]*1000)**2 + (s[3]*1000)**2 for s in samples) / len(samples))
    after = math.sqrt(sum(rx**2 + ry**2 for rx, ry in res) / len(res))

    print("\n" + "=" * 62)
    print(f"  측정 {len(samples)}점 → {fit['mode']} 보정식")
    print(f"  보정 전 평균 오차: {before:5.1f} mm")
    print(f"  보정 후 잔차     : {after:5.1f} mm")
    if after < before:
        print(f"  → {(1-after/before)*100:.0f}% 감소")
    else:
        print("  ⚠️ 개선되지 않음 — 측정 부호나 점 분포를 확인하세요")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(fit, f, indent=2)
    print(f"\n  저장: {os.path.relpath(OUT_PATH)}")
    print("  다음 실행부터 자동으로 적용됩니다 (utils/ik_solver.py가 읽음).")

    # 보정식이 가장자리 칸을 서보 범위 밖으로 밀어낼 수 있으므로 확인
    import importlib
    import utils.ik_solver as iks
    importlib.reload(iks)
    from hardware.arm_controller import RealArm as _RA, _safe_lift as _sl
    import hardware.arm_controller as _ac
    lost = []
    for c in range(8):
        for r in range(8):
            try:
                _RA._rad_to_servo(*iks.inverse_kinematics(*_ac.touch_xyz(c, r)))
                _RA._rad_to_servo(*iks.inverse_kinematics(
                    *iks.safe_approach_xyz(c, r, _sl(c, r))))
            except ValueError:
                lost.append(f"{chr(97+c)}{r+1}")
    print(f"\n  보정 적용 후 도달 가능: {64-len(lost)}/64 칸")
    if lost:
        print(f"  ⚠️ 도달 불가: {' '.join(lost)}")
        print("     보정식이 가장자리 칸을 서보 범위 밖으로 밀어낸 것입니다.")
        print("     측정 점을 보드 가장자리 쪽으로 더 넓게 잡으면 개선됩니다.")
        print(f"     되돌리려면 {os.path.relpath(OUT_PATH)} 를 지우세요.")


if __name__ == "__main__":
    main()
