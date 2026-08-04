"""
칸 단위 동작 테스트 — 서보 캘리브레이션과 실전 게임 사이의 검증 단계.

체스 칸 이름(e4 등)을 입력하면 IK로 계산해 팔을 그 칸으로 보낸다.
servo_jog.py가 '서보 각도'를 다룬다면, 이 도구는 '체스 좌표'를 다룬다.
→ SERVO*_HOME 캘리브레이션과 체스판 물리 배치가 맞는지 확인하는 용도.

실행:
  python hardware/test_square.py --port COM5 --start 90 45 135

명령:
  e4          e4 칸 위 안전높이로 이동 (기물 안 건드림)
  down        현재 칸으로 하강 (기물 높이)
  up          다시 안전높이로 상승
  pick e2     e2 기물 집기 (위→하강→흡착ON→상승)
  place e4    e4에 놓기 (위→하강→흡착OFF→상승)
  move e2 e4  e2에서 e4로 한 번에 (게임과 동일한 시퀀스)
  home        park(Z자) 자세로 복귀
  q           종료

🛑 첫 실행은 기물 없이, 전원 스위치를 손에 두고 할 것.
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hardware.arm_controller import RealArm, _safe_lift
from utils.ik_solver import inverse_kinematics, chess_square_to_xyz, safe_approach_xyz


def parse_square(sq: str):
    """'e4' → (col, row). 잘못되면 None."""
    sq = sq.strip().lower()
    if len(sq) != 2 or not ('a' <= sq[0] <= 'h') or not ('1' <= sq[1] <= '8'):
        return None
    return (ord(sq[0]) - ord('a'), int(sq[1]) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=str, default="COM5")
    ap.add_argument("--sim", action="store_true", help="하드웨어 없이 명령만 출력")
    ap.add_argument("--start", type=int, nargs=3, metavar=("S1", "S2", "S3"),
                    help="현재 팔의 실제 서보 각도 (전원 끄고 옮겼다면 필수)")
    args = ap.parse_args()

    arm = RealArm(port=args.port, sim=args.sim,
                  start_pose=tuple(args.start) if args.start else None)

    cur = None      # 현재 대상 칸 (col,row)

    def goto(col, row, lift):
        x, y, z = (safe_approach_xyz(col, row, _safe_lift(col, row))
                   if lift else chess_square_to_xyz(col, row))
        q = inverse_kinematics(x, y, z)
        arm.move(*q)
        print(f"    → x={x*100:.1f} y={y*100:.1f} z={z*100:.1f} cm  "
              f"(서보 {arm._rad_to_servo(*q)})")

    print("\n명령: e4 | down | up | pick e2 | place e4 | move e2 e4 | home | q")
    print("🛑 긁는 소리·이상 동작 시 즉시 Ctrl+C → 전원 차단\n")

    while True:
        try:
            line = input("square> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        p = line.split()

        try:
            if p[0] == "q":
                break

            elif p[0] == "home":
                print("  park 자세로 복귀")
                arm.home()

            elif p[0] in ("down", "up") and cur:
                print(f"  {'하강' if p[0]=='down' else '상승'}")
                goto(cur[0], cur[1], lift=(p[0] == "up"))

            elif p[0] == "pick" and len(p) == 2:
                sq = parse_square(p[1])
                if not sq:
                    print("  칸 이름 오류 (예: e2)"); continue
                cur = sq
                print(f"  {p[1]} 집기")
                goto(*sq, lift=True)
                goto(*sq, lift=False)
                arm.move(*inverse_kinematics(*chess_square_to_xyz(*sq)), suction=True)
                time.sleep(0.6)
                goto(*sq, lift=True)

            elif p[0] == "place" and len(p) == 2:
                sq = parse_square(p[1])
                if not sq:
                    print("  칸 이름 오류 (예: e4)"); continue
                cur = sq
                print(f"  {p[1]}에 놓기")
                goto(*sq, lift=True)
                goto(*sq, lift=False)
                arm.move(*inverse_kinematics(*chess_square_to_xyz(*sq)), suction=False)
                time.sleep(0.4)
                goto(*sq, lift=True)

            elif p[0] == "move" and len(p) == 3:
                a, b = parse_square(p[1]), parse_square(p[2])
                if not a or not b:
                    print("  칸 이름 오류 (예: move e2 e4)"); continue
                print(f"  {p[1]} → {p[2]} (게임과 동일 시퀀스)")
                arm.execute_move(a, b, is_capture=False)
                cur = b

            else:
                sq = parse_square(p[0])
                if sq:
                    cur = sq
                    print(f"  {p[0]} 위 안전높이로 이동")
                    goto(*sq, lift=True)
                else:
                    print("  명령: e4 | down | up | pick e2 | place e4 | "
                          "move e2 e4 | home | q")

        except ValueError as e:
            print(f"  [실패] {e}")
        except KeyboardInterrupt:
            print("\n  [중단] 현재 위치에서 멈춤. 전원 확인하세요.")

    arm.close()
    print("[test_square] 종료 (팔은 현재 위치 유지)")


if __name__ == "__main__":
    main()
