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
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import hardware.arm_controller as ac
from hardware.arm_controller import RealArm, _safe_lift, interactive_startup
from utils.ik_solver import inverse_kinematics, chess_square_to_xyz, safe_approach_xyz
import utils.ik_solver as iks
from hardware.visual_align import align_over_square


def ac_fit_active() -> bool:
    """실측 보정(board_fit.json)이 로드돼 있는지."""
    return getattr(iks, "_BOARD_FIT", None) is not None


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
    ap.add_argument("--no-confirm", action="store_true",
                    help="pick/place에서 칸 위 확인 단계를 생략하고 바로 하강")
    ap.add_argument("--camera", type=int, default=None,
                    help="카메라 번호를 주면 하강 전 카메라로 위치를 보고 "
                         "오차만큼 다시 움직여 정렬한다(폐루프 보정)")
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

    # 카메라 폐루프 보정 (선택)
    detector = None
    if args.camera is not None and not args.sim:
        try:
            from vision.detect import ChessBoardDetector
            detector = ChessBoardDetector(camera_index=args.camera)
            print("  [카메라] 폐루프 보정 활성 — 하강 전 마커를 보고 정렬합니다")
        except Exception as e:
            print(f"  [카메라] 초기화 실패: {e} → 보정 없이 진행")

    cur = None       # 현재 대상 칸 (col,row)
    holding = False  # 기물을 흡착해 들고 있는 중인가

    def show(win_name="test_square view"):
        """카메라가 있으면 현재 프레임(마커 표시 포함)을 창으로 띄운다.
        --mode vision과 달리 이 프로그램 자신이 카메라를 이미 열고 있으므로
        별도 프로세스 충돌 없이 같이 볼 수 있다."""
        if detector is None:
            return
        try:
            import cv2
            ret, frame = detector.cap.read()
            if not ret:
                return
            debug = detector.overlay_debug(frame)
            cv2.imshow(win_name, debug)
            cv2.waitKey(1)   # 창을 그리고 넘어감 (여기서 멈추지 않음)
        except Exception:
            pass   # 화면 표시 실패는 무시 — 보정 자체엔 지장 없음

    def align(sq):
        """카메라가 있으면 하강 전에 정렬(하며 매 반복마다 화면 갱신)."""
        if detector is None:
            return
        show()
        align_over_square(arm, detector, sq[0], sq[1], _safe_lift(*sq))
        show()

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
    if detector is not None:
        print("  카메라 명령: show(=view) 화면보기 | mark 마커위치 | "
              "markcal <칸> 오프셋측정")
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
                print(f"  {p[1]} 칸 위로 이동 — 하강 전 확인 단계")
                goto(*sq, lift=True)
                time.sleep(ac.SETTLE_WAIT)
                align(sq)
                if not args.no_confirm:
                    # 칸 위에 멈춰서 흡착컵이 기물 바로 위에 있는지 눈/카메라로 확인
                    ans = input("    흡착컵이 기물 바로 위인가요? "
                                "[Enter=하강] [x=취소] [칸이름=그 칸으로 다시] > ").strip().lower()
                    if ans in ("x", "q", "n"):
                        print("    취소 — 하강하지 않음"); continue
                    if ans:
                        sq2 = parse_square(ans)
                        if sq2:
                            cur = sq2; sq = sq2
                            print(f"    {ans} 위로 다시 이동")
                            goto(*sq, lift=True); time.sleep(ac.SETTLE_WAIT)
                        else:
                            print("    입력 무시 — 그대로 하강")
                print("    하강 → 흡착")
                goto(*sq, lift=False)
                time.sleep(ac.SETTLE_WAIT)   # 흔들림 가라앉힌 뒤 흡착
                arm.move(*inverse_kinematics(*ac.touch_xyz(*sq)), suction=True)
                time.sleep(0.6)
                goto(*sq, lift=True, suction=True)   # 든 채로 상승
                holding = True
                print("    집기 완료 (흡착 유지 — place로 놓을 때까지)")

            elif p[0] == "place" and len(p) == 2:
                sq = parse_square(p[1])
                if not sq:
                    print("  칸 이름 오류 (예: e4)"); continue
                cur = sq
                print(f"  {p[1]} 칸 위로 이동 — 하강 전 확인 단계")
                goto(*sq, lift=True,  suction=holding)   # 든 채로 이동
                time.sleep(ac.SETTLE_WAIT)
                align(sq)
                if not args.no_confirm:
                    ans = input("    이 칸에 놓을까요? [Enter=하강] [x=취소] > ").strip().lower()
                    if ans in ("x", "q", "n"):
                        print("    취소 — 하강하지 않음 (기물은 계속 들고 있음)"); continue
                print("    하강 → 놓기")
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
                elif p[0] == "markcal" and len(p) == 2:
                    # 흡착컵을 그 칸 중심에 정확히 맞춰 둔 상태에서 실행.
                    # 마커가 보이는 위치와의 차이를 오프셋으로 저장한다.
                    sq = parse_square(p[1])
                    if not sq:
                        print("  칸 이름 오류 (예: markcal e5)"); continue
                    if detector is None:
                        print("  --camera 옵션으로 실행해야 합니다"); continue
                    import vision.detect as vd
                    vd.MARKER_OFFSET_COL = vd.MARKER_OFFSET_ROW = 0.0
                    mk = detector.find_marker()
                    if mk is None:
                        print("  마커를 못 찾음 — MARKER_HSV_RANGES 확인"); continue
                    oc, orow = mk[0] - sq[0], mk[1] - sq[1]
                    vd.MARKER_OFFSET_COL, vd.MARKER_OFFSET_ROW = oc, orow
                    print(f"  마커 오프셋 = 파일 {oc:+.2f}칸, 랭크 {orow:+.2f}칸 "
                          f"({math.hypot(oc,orow)*2.91:.1f}mm)")
                    print(f"  영구 적용하려면 vision/detect.py 에:")
                    print(f"    MARKER_OFFSET_COL = {oc:.3f}")
                    print(f"    MARKER_OFFSET_ROW = {orow:.3f}")
                elif p[0] in ("show", "view"):
                    if detector is None:
                        print("  --camera 옵션으로 실행해야 합니다"); continue
                    show()
                    print("  창을 띄웠습니다 (다른 명령을 입력하면 계속 갱신됩니다)")
                elif p[0] == "mark":
                    if detector is None:
                        print("  --camera 옵션으로 실행해야 합니다"); continue
                    show()
                    mk = detector.find_marker()
                    if mk is None:
                        import vision.detect as vd
                        a = detector._last_marker_area
                        print("  마커를 못 찾음")
                        if a > vd.MARKER_MAX_AREA:
                            print(f"    → 너무 큰 덩어리({a:.0f}px)를 잡음. 체스판이나 반사를"
                                  f" 마커로 오인한 것입니다.")
                            print(f"       MARKER_HSV_RANGES의 채도(S) 최소값을 올리세요.")
                        else:
                            print("    → --mode vision 창이 닫혀 있는지, 색 범위가 맞는지 확인")
                    else:
                        import vision.detect as vd
                        print(f"  마커 위치: 파일 {mk[0]:+.2f}, 랭크 {mk[1]:+.2f} "
                              f"(가장 가까운 칸 {chr(97+int(round(mk[0])))}{int(round(mk[1]))+1})")
                        print(f"  검출 크기: {detector._last_marker_area:.0f}px "
                              f"(정상 범위 {vd.MARKER_MIN_AREA}~{vd.MARKER_MAX_AREA})")
                elif p[0] == "check":
                    # 현재 캘리브레이션·보정 기준으로 어느 칸에 닿는지 지도 출력
                    print("  도달 지도 (O=닿음, .=서보범위 밖)   랭크8=로봇쪽")
                    print("          " + "  ".join("abcdefgh"))
                    n = 0
                    for r in range(7, -1, -1):
                        row = []
                        for c in range(8):
                            try:
                                arm._rad_to_servo(*inverse_kinematics(*ac.touch_xyz(c, r)))
                                arm._rad_to_servo(*inverse_kinematics(
                                    *safe_approach_xyz(c, r, _safe_lift(c, r))))
                                row.append("O"); n += 1
                            except ValueError:
                                row.append(".")
                        print(f"   랭크{r+1}  " + "  ".join(row))
                    print(f"  → {n}/64 칸 도달 가능")
                    if ac_fit_active():
                        print("  ※ 실측 보정(board_fit.json)이 적용된 상태입니다.")
                        print("    가장자리 칸이 범위 밖이면 보정식 외삽 때문일 수 있습니다.")
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
                    print("  명령: e4 | down | up | pick e2 | place e4 | move e2 e4")
                    print("        check | zoff 8 | suction 1 | set 38 140 180 | home | q")

        except ValueError as e:
            print(f"  [실패] {e}")
            print("     ※ 랭크1~2(사람 진영)는 현재 배치에서 팔이 닿지 않습니다."
                  " 랭크3~8로 시도하세요.")
        except KeyboardInterrupt:
            print("\n  [중단] 현재 위치에서 멈춤. 전원 확인하세요.")

    if detector is not None:
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        detector.close()
    arm.close()
    print("[test_square] 종료 (팔은 현재 위치 유지)")


if __name__ == "__main__":
    main()
