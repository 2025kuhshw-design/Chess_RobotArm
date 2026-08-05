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
import hardware.arm_controller as ac
from hardware.arm_controller import RealArm, _safe_lift, interactive_startup
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
                    help="현재 팔의 실제 서보 각도. 생략하면 PARK_POSE에 "
                         "있다고 가정 (전원 끄고 옮겼다면 반드시 지정)")
    ap.add_argument("--step", type=int, default=None,
                    help="한 스텝당 각도(도). 작을수록 느리고 안전 (기본 2)")
    ap.add_argument("--step-delay", type=float, default=None,
                    help="스텝 간 대기(s). 클수록 느리고 안전 (기본 0.05)")
    ap.add_argument("--no-startup", action="store_true",
                    help="기동 절차를 건너뛴다 (팔이 이미 PARK에 잡혀 있을 때)")
    args = ap.parse_args()

    # auto_home=False: 시작하자마자 팔을 움직이지 않는다.
    # (부팅 직후 서보 무부하 → 팔이 처져 있고, 첫 명령은 램프가 안 먹는다)
    arm = RealArm(port=args.port, sim=args.sim,
                  start_pose=tuple(args.start) if args.start else None,
                  ramp_step=args.step, ramp_delay=args.step_delay,
                  auto_home=False)

    # 기동 절차 (팔이 처진 상태에서 안전하게 시작)
    if not args.sim and not args.no_startup:
        if not interactive_startup(arm):
            arm.close()
            print("[test_square] 기동 중단."); return

    cur = None       # 현재 대상 칸 (col,row)
    holding = False  # 기물을 흡착해 들고 있는 중인가

    def goto(col, row, lift, suction=False):
        """suction을 반드시 넘길 것 — 기본값으로 두면 기물을 든 채 이동하는
        구간에서 흡착이 풀려 기물을 떨어뜨린다."""
        x, y, z = (safe_approach_xyz(col, row, _safe_lift(col, row))
                   if lift else ac.touch_xyz(col, row))
        q = inverse_kinematics(x, y, z)
        arm.move(*q, suction=suction)
        print(f"    → x={x*100:.1f} y={y*100:.1f} z={z*100:.1f} cm  "
              f"(서보 {arm._rad_to_servo(*q)}){'  [흡착 유지]' if suction else ''}")

    print("\n명령: e4 | down | up | pick e2 | place e4 | move e2 e4 | "
          "set 38 140 180 | home | q")
    print(f"  현재 위치 가정: A{arm._cur[0]},{arm._cur[1]},{arm._cur[2]}"
          "  (실제와 다르면 'set'으로 교정)")
    print("  ⚠️ 시작 시 아무 명령도 보내지 않습니다. 첫 이동 명령부터 제어 시작.")
    print("     팔이 처져 있으면 손으로 Z자 자세로 받쳐준 뒤 첫 명령을 주세요.")
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

            elif p[0] == "set" and len(p) == 4:
                # 움직이지 않고 '현재 위치 가정'만 교정 (첫 명령 슬램 방지)
                arm._cur = [int(v) for v in p[1:]]
                print(f"  현재 위치 가정을 A{arm._cur[0]},{arm._cur[1]},"
                      f"{arm._cur[2]} 로 교정 (전송 안 함)")

            elif p[0] in ("down", "up"):
                if cur is None:
                    print("  먼저 칸을 지정하세요 (예: e5)")
                    continue
                print(f"  {'하강' if p[0]=='down' else '상승'}")
                goto(cur[0], cur[1], lift=(p[0] == "up"), suction=holding)

            elif p[0] == "pick" and len(p) == 2:
                sq = parse_square(p[1])
                if not sq:
                    print("  칸 이름 오류 (예: e2)"); continue
                cur = sq
                print(f"  {p[1]} 집기 (이후 흡착 유지 — place로 놓을 때까지)")
                goto(*sq, lift=True)
                time.sleep(ac.SETTLE_WAIT)
                goto(*sq, lift=False)
                time.sleep(ac.SETTLE_WAIT)   # 흔들림 가라앉힌 뒤 흡착
                arm.move(*inverse_kinematics(*ac.touch_xyz(*sq)), suction=True)
                time.sleep(0.6)
                goto(*sq, lift=True, suction=True)   # 든 채로 상승
                holding = True

            elif p[0] == "place" and len(p) == 2:
                sq = parse_square(p[1])
                if not sq:
                    print("  칸 이름 오류 (예: e4)"); continue
                cur = sq
                print(f"  {p[1]}에 놓기")
                goto(*sq, lift=True,  suction=holding)   # 든 채로 이동
                time.sleep(ac.SETTLE_WAIT)
                goto(*sq, lift=False, suction=holding)   # 든 채로 하강
                time.sleep(ac.SETTLE_WAIT)   # 흔들림 가라앉힌 뒤 놓기
                arm.move(*inverse_kinematics(*ac.touch_xyz(*sq)), suction=False)
                time.sleep(0.4)
                holding = False
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
                    goto(*sq, lift=True, suction=holding)
                elif p[0] == "zoff":
                    # 하강 깊이를 mm 단위로 실시간 조정 (값 찾기용)
                    if len(p) == 2:
                        ac.TOUCH_PRESS = float(p[1]) / 1000.0
                    print(f"  하강 깊이 = 기물 윗면보다 "
                          f"{ac.TOUCH_PRESS*1000:.1f}mm 더 아래")
                    print("     (더 내려가려면 'zoff 8', 덜 내려가려면 'zoff 2')")
                elif p[0] == "suction" and len(p) == 2:
                    # 현재 서보 각도 그대로 두고 흡착 비트만 바꾼다 (움직이지 않음)
                    holding = (p[1] == "1")
                    arm._send_cmd(arm._cur[0], arm._cur[1], arm._cur[2], holding)
                    print(f"  흡착 {'ON (유지)' if holding else 'OFF'} "
                          f"— 위치 A{arm._cur[0]},{arm._cur[1]},{arm._cur[2]} 유지")
                else:
                    print("  명령: e4 | down | up | pick e2 | place e4 | "
                          "move e2 e4 | home | q")

        except ValueError as e:
            print(f"  [실패] {e}")
            print("     ※ 랭크1~2(사람 진영)는 현재 배치에서 팔이 닿지 않습니다."
                  " 랭크3~8로 시도하세요.")
        except KeyboardInterrupt:
            print("\n  [중단] 현재 위치에서 멈춤. 전원 확인하세요.")

    arm.close()
    print("[test_square] 종료 (팔은 현재 위치 유지)")


if __name__ == "__main__":
    main()
