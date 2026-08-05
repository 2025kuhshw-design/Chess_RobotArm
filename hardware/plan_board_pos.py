"""
체스판을 어디에 놓아야 가장 많은 칸에 닿는지 계산한다.

테이프를 뜯기 전에 먼저 돌려볼 것. 보드를 옮기면 카메라 캘리브레이션을
다시 해야 하므로, 얻는 칸 수가 그만한 값어치가 있는지 숫자로 확인한다.

실행:
  python hardware/plan_board_pos.py                 # 현재 설정으로 스윕
  python hardware/plan_board_pos.py --smax 195      # 관절 상한을 올렸다고 가정
  python hardware/plan_board_pos.py --show 0.06     # 특정 배치의 칸별 지도
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import hardware.arm_controller as ac
from utils.ik_solver import (inverse_kinematics, CELL_SIZE, PIECE_Z,
                             L1_DEFAULT, L2_DEFAULT)

FILES = "abcdefgh"


def square_xy(ox, oy, col, row):
    """chess_square_to_xyz 와 같은 규칙 — 단, 원점을 인자로 받는다."""
    x = ox + (7 - row) * CELL_SIZE + CELL_SIZE / 2
    y = oy + col * CELL_SIZE + CELL_SIZE / 2
    return x, y


def survey(ox, oy, lift=0.0):
    """(도달칸수, 실패칸목록, 최대서보각) — 접근높이 lift 에서 평가."""
    ok = 0
    bad = []
    peak = 0
    for row in range(8):
        for col in range(8):
            x, y = square_xy(ox, oy, col, row)
            try:
                s = ac.RealArm._rad_to_servo(
                    *inverse_kinematics(x, y, PIECE_Z + lift))
                ok += 1
                peak = max(peak, max(s))
            except ValueError:
                bad.append(FILES[col] + str(row + 1))
    return ok, bad, peak


def show_map(ox, oy, lift=0.0):
    """칸별 베이스 각도 / 실패 사유 지도."""
    print(f"  (숫자=베이스 s1 각도, 멀=팔이 짧음, 가=너무 가까움, 범=관절한계)")
    for row in range(7, -1, -1):
        cells = []
        for col in range(8):
            x, y = square_xy(ox, oy, col, row)
            try:
                s = ac.RealArm._rad_to_servo(
                    *inverse_kinematics(x, y, PIECE_Z + lift))
                cells.append(f"{s[0]:3d}")
            except ValueError as e:
                m = str(e)
                cells.append(" 멀" if "팔 길이" in m
                             else " 가" if "L1-L2" in m else " 범")
        print(f"  rank{row+1}  " + " ".join(cells))
    print("          " + "   ".join(FILES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smax", type=int, default=None,
                    help="관절 상한을 이 값으로 가정 (기본: 현재 SERVO_SAFE_MAX)")
    ap.add_argument("--oy", type=float, default=None,
                    help="보드 좌우 위치 y (m). 기본: 현재 BOARD_ORIGIN_Y")
    ap.add_argument("--show", type=float, default=None,
                    help="이 x 값 배치의 칸별 지도를 출력 (m)")
    ap.add_argument("--lift", type=float, default=0.0,
                    help="평가 높이 (m). 0=기물 표면, 0.04=접근높이")
    args = ap.parse_args()

    from utils.ik_solver import BOARD_ORIGIN_X, BOARD_ORIGIN_Y
    oy = args.oy if args.oy is not None else BOARD_ORIGIN_Y

    if args.smax is not None:
        ac.SERVO_SAFE_MAX = [args.smax] * 3

    hi = min(ac.SERVO_MAX, min(ac.SERVO_SAFE_MAX))
    print(f"팔 길이 L1+L2 = {(L1_DEFAULT+L2_DEFAULT)*100:.1f} cm")
    print(f"체스판 대각선 = {8*CELL_SIZE*math.sqrt(2)*100:.1f} cm")
    print(f"관절 상한     = {hi}°   (SERVO_MAX={ac.SERVO_MAX}, "
          f"SERVO_SAFE_MAX={ac.SERVO_SAFE_MAX})")
    print(f"보드 좌우 y   = {oy*100:+.1f} cm\n")

    if args.show is not None:
        print(f"=== 배치 x={args.show:.3f} m 의 도달 지도 ===")
        show_map(args.show, oy, args.lift)
        n, bad, peak = survey(args.show, oy, args.lift)
        print(f"\n  도달 {n}/64,  최대 서보각 {peak}°")
        if bad:
            print(f"  실패: {' '.join(bad)}")
        return

    cur_n, _, _ = survey(BOARD_ORIGIN_X, oy, args.lift)
    print(f"현재 배치 x={BOARD_ORIGIN_X:.3f} m → {cur_n}/64\n")
    print("x(cm)  도달   최대각   실패 랭크")
    best = None
    for mm in range(40, 141, 5):
        ox = mm / 1000
        n, bad, peak = survey(ox, oy, args.lift)
        ranks = sorted({int(s[1]) for s in bad})
        mark = ""
        if best is None or n > best[0]:
            best = (n, ox)
        print(f"{ox*100:5.1f}  {n:2d}/64   {peak:3d}°    {ranks}{mark}")
    print(f"\n최선: x={best[1]*100:.1f} cm → {best[0]}/64  "
          f"(현재 {cur_n}/64, {best[0]-cur_n:+d}칸)")
    print("\n※ 보드를 옮기면 vision/calibration.json 과 board_fit.json 을"
          " 다시 만들어야 한다.")


if __name__ == "__main__":
    main()
