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
from vision.live_view import LiveView, DetectorProxy


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
                    help="카메라 번호(이 PC는 보통 1). 주면 하강 전 카메라로 "
                         "위치를 보고 오차만큼 다시 움직여 정렬한다(폐루프 보정)")
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

    markcal_pts = []  # markcal 로 모은 (마커col, 마커row, 실제col, 실제row)
    nudge_off = [0.0, 0.0]   # nudge 로 밀어놓은 누적량 (칸 단위)
    cur = None       # 현재 대상 칸 (col,row)
    holding = False  # 기물을 흡착해 들고 있는 중인가

    WIN = "test_square view"

    # 라이브 뷰 / 프레임 공유 프록시는 main.py 와 공유한다 (vision/live_view.py)
    live = LiveView(detector, WIN) if detector is not None else None
    det_src = DetectorProxy(detector, live) if detector is not None else None


    def show_once():
        """프레임 한 장을 창에 갱신. 이동/정렬 중 호출용.
        라이브 뷰 스레드가 돌고 있으면 그쪽이 알아서 갱신하므로 아무것도 안 한다."""
        if detector is None or (live is not None and live.running):
            return
        try:
            import cv2
            ret, frame = detector.cap.read()
            if not ret:
                return
            cv2.imshow(WIN, detector.overlay_debug(frame))
            for _ in range(3):      # 이벤트 루프를 몇 번 돌려 실제로 그려지게
                cv2.waitKey(1)
        except Exception:
            pass   # 화면 표시 실패는 무시 — 보정 자체엔 지장 없음

    def show_live(seconds=8.0):
        """지정 시간 동안 라이브로 보여준다. 아무 키나 누르면 종료.

        input()은 블로킹이라 그 동안 GUI 이벤트가 처리되지 않는다. 그래서
        '창을 띄워두고 프롬프트로 돌아가는' 방식은 Windows에서 응답 없음으로
        보인다. 대신 여기서 정해진 시간만큼 이벤트 루프를 직접 돌린다."""
        if detector is None:
            print("  --camera 옵션으로 실행해야 합니다"); return
        if live is not None and live.running:
            print("  라이브 뷰가 이미 켜져 있습니다 ('show off'로 끄기)"); return
        try:
            import cv2
        except Exception as e:
            print(f"  창을 열 수 없습니다: {e}"); return
        print(f"  라이브 뷰 {seconds:.0f}초 — 아무 키나 누르면 종료")
        t0 = time.time()
        while time.time() - t0 < seconds:
            ret, frame = detector.cap.read()
            if not ret:
                continue
            cv2.imshow(WIN, detector.overlay_debug(frame))
            if cv2.waitKey(30) != -1:      # 키 입력 시 즉시 종료
                break
        print("  라이브 뷰 종료")

    show = show_once

    def align(sq, suction=False, on_piece=False):
        """카메라가 있으면 하강 전에 정렬(하며 매 반복마다 화면 갱신).
        ⚠️ 기물을 들고 있으면 suction=True — 안 넘기면 정렬 중 떨어뜨린다."""
        if detector is None:
            return
        from hardware.visual_align import ALIGN_WORK_LIFT
        target = det_src.piece_center(sq[0], sq[1]) if on_piece else None
        align_over_square(arm, det_src, sq[0], sq[1], ALIGN_WORK_LIFT,
                          view_cb=show_once, suction=suction, target=target)
        show_once()

    def aligner_cb(col, row, lift, suction, on_piece=False):
        """execute_move가 하강 직전마다 부르는 콜백.
        ⚠️ 정렬이 돌려준 보정 오프셋을 그대로 반환해야 한다. 삼키면
           execute_move 가 하강할 때 원래 칸 좌표로 되돌아간다.
        on_piece=True 면 칸 중심이 아니라 기물의 실제 중심을 목표로 삼는다.
        ⚠️ 팔이 아직 안전높이에 있을 때 재야 한다 — 내려간 뒤엔 팔이 기물을
           가려서 무게중심이 엉뚱하게 나온다."""
        if detector is None:
            return None
        target = det_src.piece_center(col, row) if on_piece else None
        off = align_over_square(arm, det_src, col, row, lift,
                                view_cb=show_once, suction=suction,
                                target=target)
        show_once()
        return off

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
          "set 85 140 180 | home | q")
    print("  진단: a 110 140 180  서보 각도로 직접 이동 (한 축만 바꿔 시험)")
    print("        limit            소프트 제한 보기/바꾸기 (limit max s3 200)")
    print("        probe s3 180 200 그 관절의 실제 기계 한계를 5도씩 찾아본다")
    if detector is not None:
        print("  카메라 명령: show(켜기) / show off(끄기) | mark 마커위치 | "
              "markcal auto 시차보정")
        print("              board  기물 인식·판 방향 점검 (시작 배치와 대조)")
        print("              margin 판 바깥을 카메라가 몇 칸까지 보는지 "
              "(외곽 칸에서 마커가 안 보일 때)")
        print("              refcap ⭐ 빈 판 기준 영상 찍기 (기물 다 치우고)")
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

            elif p[0] == "a" and len(p) == 4:
                # 서보 각도로 직접 이동 (램프 적용). 관절 하나만 바꿔서
                # '그 모터가 실제로 도는지' 격리 진단할 때 쓴다.
                tgt = [int(v) for v in p[1:]]
                print(f"  A{tgt[0]},{tgt[1]},{tgt[2]} 로 이동 "
                      f"(현재 가정 A{arm._cur[0]},{arm._cur[1]},{arm._cur[2]})")
                arm._ramp_send(tgt[0], tgt[1], tgt[2], holding)
                print(f"  → 완료. 현재 A{arm._cur[0]},{arm._cur[1]},{arm._cur[2]}")

            elif p[0] == "limit":
                # 소프트 제한(SERVO_SAFE_MIN/MAX)을 실행 중에 조정한다.
                # 한계를 '재는' 동안에는 제한이 방해가 되므로 여기서 풀고,
                # 확인이 끝나면 그 값을 arm_controller.py 에 적어 넣는다.
                if len(p) == 1:
                    print(f"  SERVO_SAFE_MIN = {ac.SERVO_SAFE_MIN}")
                    print(f"  SERVO_SAFE_MAX = {ac.SERVO_SAFE_MAX}")
                    print(f"  (펌웨어 절대 한계 SERVO_MAX = {ac.SERVO_MAX})")
                    print("  사용법: limit max 200 | limit max s3 195 | "
                          "limit min s2 20")
                    continue
                if p[1] not in ("max", "min"):
                    print("  사용법: limit max 200 | limit max s3 195"); continue
                arr = ac.SERVO_SAFE_MAX if p[1] == "max" else ac.SERVO_SAFE_MIN
                if len(p) == 3:                       # 세 관절 모두
                    arr[:] = [int(p[2])] * 3
                elif len(p) == 4 and p[2] in ("s1", "s2", "s3"):
                    arr[int(p[2][1]) - 1] = int(p[3])
                else:
                    print("  사용법: limit max 200 | limit max s3 195"); continue
                print(f"  SERVO_SAFE_MIN = {ac.SERVO_SAFE_MIN}")
                print(f"  SERVO_SAFE_MAX = {ac.SERVO_SAFE_MAX}")
                print("  ⚠️ 이 값은 이 실행에서만 유효합니다. 확정되면")
                print("     hardware/arm_controller.py 에 직접 적어 넣으세요.")

            elif p[0] == "probe" and len(p) in (3, 4):
                # 관절 하나의 '실제 기계 한계'를 5°씩 올려가며 찾는다.
                #   probe s3 180 200
                # 각 단계에서 사람이 눈으로 보고 '더 움직였는지' 판단한다.
                # 서보에는 위치 센서가 없어 코드로는 알 수 없기 때문이다.
                jn = p[1]
                if jn not in ("s1", "s2", "s3"):
                    print("  관절은 s1/s2/s3"); continue
                j = int(jn[1]) - 1
                start = int(p[2])
                end = int(p[3]) if len(p) == 4 else ac.SERVO_MAX
                stepd = 5 if end >= start else -5

                saved = list(ac.SERVO_SAFE_MAX), list(ac.SERVO_SAFE_MIN)
                # 재는 동안만 그 관절의 제한을 펌웨어 한계까지 열어 둔다
                ac.SERVO_SAFE_MAX[j] = max(ac.SERVO_SAFE_MAX[j], start, end)
                ac.SERVO_SAFE_MIN[j] = min(ac.SERVO_SAFE_MIN[j], start, end)

                print(f"  {jn} 를 {start}° → {end}° 까지 {abs(stepd)}°씩 움직입니다.")
                print("  각 단계에서 팔이 '실제로 더 움직였는지' 눈으로 보세요.")
                print("  🛑 끼익·드르륵 소리가 나면 즉시 x 를 누르세요 (기어 손상 중).")
                last_ok = None
                try:
                    for tgt in range(start, end + stepd, stepd):
                        cmd = list(arm._cur)
                        cmd[j] = tgt
                        arm._ramp_send(cmd[0], cmd[1], cmd[2], holding)
                        ans = input(f"    {jn}={tgt}° → 더 움직였나요? "
                                    "[Enter=예, 계속] [x=아니오/소리남 → 여기가 한계] > "
                                    ).strip().lower()
                        if ans in ("x", "n", "q"):
                            break
                        last_ok = tgt
                finally:
                    ac.SERVO_SAFE_MAX[:], ac.SERVO_SAFE_MIN[:] = saved[0], saved[1]

                if last_ok is None:
                    print(f"  → {start}° 에서 이미 안 움직였습니다. "
                          "시작값을 더 낮춰서 다시 재보세요.")
                else:
                    print(f"  → {jn} 의 실제 한계 = {last_ok}°")
                    print(f"     되돌립니다: {jn}={last_ok}")
                    cmd = list(arm._cur); cmd[j] = last_ok
                    ac.SERVO_SAFE_MAX[j] = max(ac.SERVO_SAFE_MAX[j], last_ok)
                    arm._ramp_send(cmd[0], cmd[1], cmd[2], holding)
                    tag = "MAX" if stepd > 0 else "MIN"
                    print(f"\n     hardware/arm_controller.py 에 적어 넣으세요:")
                    print(f"       SERVO_SAFE_{tag}[{j}] = {last_ok}   # {jn}")

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
                nudge_off[0] = nudge_off[1] = 0.0
                print(f"  {p[1]} 칸 위로 이동 — 하강 전 확인 단계")
                goto(*sq, lift=True)
                time.sleep(ac.SETTLE_WAIT)
                align(sq, on_piece=True)   # 기물 실제 중심에 맞춘다
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
                            nudge_off[0] = nudge_off[1] = 0.0
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
                nudge_off[0] = nudge_off[1] = 0.0
                print(f"  {p[1]} 칸 위로 이동 — 하강 전 확인 단계")
                goto(*sq, lift=True,  suction=holding)   # 든 채로 이동
                time.sleep(ac.SETTLE_WAIT)
                align(sq, suction=holding)   # 든 채로 정렬 (흡착 유지)
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
                arm.execute_move(a, b, is_capture=False,
                                 aligner=aligner_cb if detector else None)
                cur = b

            else:
                sq = parse_square(p[0])
                if sq:
                    cur = sq
                    nudge_off[0] = nudge_off[1] = 0.0
                    print(f"  {p[0]} 위 안전높이로 이동")
                    goto(*sq, lift=True, suction=holding)
                elif p[0] == "nudge" and len(p) == 3:
                    # 지금 높이 그대로 판 좌표계로 mm 만큼 옆으로 민다.
                    # markcal add 를 쓸 때 흡착컵을 칸 중심에 맞추는 용도.
                    if cur is None:
                        print("  먼저 칸을 지정하세요 (예: e5)"); continue
                    try:
                        df, dr = float(p[1]), float(p[2])
                    except ValueError:
                        print("  사용법: nudge <파일mm> <랭크mm>  (예: nudge 2 -3)"); continue
                    nudge_off[0] += df / 29.125
                    nudge_off[1] += dr / 29.125
                    from hardware.visual_align import offset_xy, ALIGN_WORK_LIFT
                    x, y = offset_xy(cur[0], cur[1], nudge_off[0], nudge_off[1])
                    try:
                        arm.move(*inverse_kinematics(x, y, ac.PIECE_Z + ALIGN_WORK_LIFT),
                                 suction=holding)
                    except ValueError as e:
                        print(f"  도달 불가 ({e})"); continue
                    print(f"  누적 이동: 파일 {nudge_off[0]*29.125:+.1f}mm, "
                          f"랭크 {nudge_off[1]*29.125:+.1f}mm")

                elif p[0] == "markcal":
                    # ── 마커 위치 → 흡착컵 실제 위치 관계를 재는 명령 ──
                    # ⚠️ 왜 필요한가: 정렬은 '마커'를 칸 중심에 놓는다. 그런데
                    #    카메라가 완전한 수직이 아니면 마커는 시차로 밀려 보여서,
                    #    마커를 중심에 맞추면 **흡착컵은 반대편으로 밀린다**.
                    #    25mm 높이·판 가장자리에서 10mm 넘게 벌어진다.
                    #    이 밀림은 판 위치에 따라 선형으로 커지므로 상수 하나로는
                    #    못 잡는다 → 여러 칸에서 재서 어파인 변환을 구한다.
                    if detector is None:
                        print("  --camera 옵션으로 실행해야 합니다"); continue
                    import vision.detect as vd
                    sub = p[1] if len(p) >= 2 else "help"

                    if sub == "auto":
                        # ⭐ 팔이 스스로 여러 칸을 돌며 잰다 — 손으로 옮길 필요 없음.
                        # 논리: 우리가 없애려는 건 '시차'인데, 이건 위치에 비례하는
                        #   **계통 오차**다. 관절 유격은 매번 방향이 달라지는
                        #   **무작위 오차**라 여러 점을 최소제곱하면 상쇄된다.
                        #   그래서 "명령한 칸 = 실제 흡착컵 위치"로 놓고 재도
                        #   계통 성분(시차)은 제대로 뽑힌다.
                        from hardware.visual_align import (ALIGN_WORK_LIFT,
                                                           ALIGN_CLEAR_LIFT)
                        # 점이 많을수록 유격(무작위 오차)이 상쇄돼 시차(계통 오차)만
                        # 남는다. 모의실험: 유격 ±5mm 일 때 보정 후 평균 오차가
                        #   6칸 4.0mm / 12칸 3.2mm / 20칸 2.5mm  (보정 전 4.9mm)
                        SQUARES = [f"{'abcdefgh'[c]}{r+1}"
                                   for c in (0, 2, 5, 7) for r in (3, 4, 5, 6, 7)]
                        print(f"  {len(SQUARES)}칸을 돌며 잽니다 "
                              f"(약 {len(SQUARES)*4}초). 판 위 기물을 모두 치우세요.")
                        if input("  준비됐으면 Enter (취소는 x) > ").strip().lower() == "x":
                            continue
                        markcal_pts.clear()
                        sf, so = vd.MARKER_FIT, (vd.MARKER_OFFSET_COL, vd.MARKER_OFFSET_ROW)
                        try:
                            for name in SQUARES:
                                sq = parse_square(name)
                                x, y, _ = chess_square_to_xyz(*sq)
                                try:
                                    arm.move(*inverse_kinematics(x, y, ac.PIECE_Z + ALIGN_CLEAR_LIFT))
                                    arm.move(*inverse_kinematics(x, y, ac.PIECE_Z + ALIGN_WORK_LIFT))
                                except ValueError as e:
                                    print(f"    {name}: 도달 불가 — 건너뜀 ({e})"); continue
                                time.sleep(ac.SETTLE_WAIT)
                                show_once()
                                vd.MARKER_FIT = None
                                vd.MARKER_OFFSET_COL = vd.MARKER_OFFSET_ROW = 0.0
                                mk = det_src.find_marker()
                                vd.MARKER_FIT = sf
                                vd.MARKER_OFFSET_COL, vd.MARKER_OFFSET_ROW = so
                                if mk is None:
                                    print(f"    {name}: 마커 못 찾음 — 건너뜀"); continue
                                markcal_pts.append((mk[0], mk[1], float(sq[0]), float(sq[1])))
                                d = math.hypot(mk[0]-sq[0], mk[1]-sq[1]) * 29.125
                                print(f"    {name}: 마커 ({mk[0]:.2f},{mk[1]:.2f}) "
                                      f"차이 {d:.1f}mm")
                        finally:
                            vd.MARKER_FIT = sf
                            vd.MARKER_OFFSET_COL, vd.MARKER_OFFSET_ROW = so
                        print(f"  {len(markcal_pts)}점 수집 완료.")
                        if len(markcal_pts) >= 3:
                            print("  → 'markcal fit' 으로 계산·저장하세요")
                        else:
                            print("  ⚠️ 점이 부족합니다. 마커가 잘 잡히는지 'mark' 로 확인")

                    elif sub == "add" and len(p) == 3:
                        sq = parse_square(p[2])
                        if not sq:
                            print("  칸 이름 오류 (예: markcal add e5)"); continue
                        # 보정을 끈 '날것' 측정값을 모은다
                        sf, so = vd.MARKER_FIT, (vd.MARKER_OFFSET_COL, vd.MARKER_OFFSET_ROW)
                        vd.MARKER_FIT = None
                        vd.MARKER_OFFSET_COL = vd.MARKER_OFFSET_ROW = 0.0
                        mk = det_src.find_marker()
                        vd.MARKER_FIT = sf
                        vd.MARKER_OFFSET_COL, vd.MARKER_OFFSET_ROW = so
                        if mk is None:
                            print("  마커를 못 찾음 — 색 범위 확인"); continue
                        markcal_pts.append((mk[0], mk[1], float(sq[0]), float(sq[1])))
                        d = math.hypot(mk[0]-sq[0], mk[1]-sq[1]) * 29.125
                        print(f"  기록 {len(markcal_pts)}번째: {p[2]} — "
                              f"마커 ({mk[0]:.2f},{mk[1]:.2f}), 차이 {d:.1f}mm")
                        if len(markcal_pts) < 3:
                            print(f"  → {3-len(markcal_pts)}개 더 필요합니다 "
                                  "(판의 서로 다른 구석에서 재세요)")
                        else:
                            print("  → 'markcal fit' 으로 계산·저장")

                    elif sub == "fit":
                        if len(markcal_pts) < 3:
                            print(f"  점이 {len(markcal_pts)}개뿐입니다. "
                                  "3개 이상 필요 (markcal add <칸>)"); continue
                        import numpy as np
                        A = np.array([[m[0], m[1], 1.0] for m in markcal_pts])
                        tc = np.array([m[2] for m in markcal_pts])
                        tr = np.array([m[3] for m in markcal_pts])
                        try:
                            cc, *_ = np.linalg.lstsq(A, tc, rcond=None)
                            cr, *_ = np.linalg.lstsq(A, tr, rcond=None)
                        except Exception as e:
                            print(f"  계산 실패: {e}"); continue
                        fit = [float(v) for v in (*cc, *cr)]
                        # 잔차 확인 — 점이 한 줄로 늘어서 있으면 값이 이상해진다
                        res = []
                        for mc, mr, ttc, ttr in markcal_pts:
                            pc = fit[0]*mc + fit[1]*mr + fit[2]
                            pr = fit[3]*mc + fit[4]*mr + fit[5]
                            res.append(math.hypot(pc-ttc, pr-ttr) * 29.125)
                        print(f"  {len(markcal_pts)}점으로 계산: 잔차 평균 "
                              f"{sum(res)/len(res):.1f}mm, 최대 {max(res):.1f}mm")
                        # ⚠️ 평균·최대만 보면 '어느 칸이 나쁜지'를 모른다.
                        #    실제로 랭크 7·8 에서만 크게 틀어지는 일이 있었다.
                        named = [(f"{'abcdefgh'[int(m[2])]}{int(m[3])+1}", r,
                                  int(m[3]) + 1)
                                 for m, r in zip(markcal_pts, res)]
                        worst = sorted(named, key=lambda t: -t[1])[:5]
                        print("  잔차가 큰 칸: " +
                              ", ".join(f"{n} {r:.1f}mm" for n, r, _ in worst))
                        near = [r for _, r, rk in named if rk >= 7]
                        far = [r for _, r, rk in named if rk <= 6]
                        if near and far:
                            mn, mf = sum(near)/len(near), sum(far)/len(far)
                            print(f"  랭크 7~8 평균 {mn:.1f}mm  vs  랭크 1~6 평균 "
                                  f"{mf:.1f}mm")
                            if mn > mf * 1.8 and mn > 4:
                                print("  ⚠️ 로봇 쪽(랭크 7~8)에서만 크게 틀어집니다.")
                                print("     이건 시차가 아니라 **팔 자체의 위치 오차**일")
                                print("     가능성이 큽니다. markcal 은 카메라 보정이라")
                                print("     팔의 계통 오차는 못 고칩니다(오히려 그대로")
                                print("     남습니다). 자로 재는 보정을 쓰세요:")
                                print("       python hardware/calibrate_board.py --port <포트>")
                        if max(res) > 8:
                            print("  ⚠️ 잔차가 큽니다. 점들이 한 줄로 늘어서 있지"
                                  " 않은지, 마커를 제대로 잡았는지 확인하세요")
                        vd.save_colors(marker_fit=fit, marker_offset=(0.0, 0.0))
                        print(f"  저장 완료 → {vd.COLORS_PATH} (바로 적용됨)")
                        print("  확인: 'mark' 로 흡착컵 실제 위치와 맞는지 보세요")

                    elif sub == "clear":
                        markcal_pts.clear()
                        vd.save_colors(marker_fit=None, marker_offset=(0.0, 0.0))
                        print("  기록·저장된 보정을 모두 지웠습니다")

                    elif len(p) == 2 and parse_square(p[1]):
                        # 예전 방식: 한 칸에서 상수 오프셋만 (급할 때만)
                        sq = parse_square(p[1])
                        sf = vd.MARKER_FIT
                        vd.MARKER_FIT = None
                        vd.MARKER_OFFSET_COL = vd.MARKER_OFFSET_ROW = 0.0
                        mk = det_src.find_marker()
                        vd.MARKER_FIT = sf
                        if mk is None:
                            print("  마커를 못 찾음"); continue
                        oc, orow = mk[0] - sq[0], mk[1] - sq[1]
                        print(f"  마커 오프셋 = 파일 {oc:+.2f}칸, 랭크 {orow:+.2f}칸 "
                              f"({math.hypot(oc,orow)*29.125:.1f}mm)")
                        vd.save_colors(marker_offset=(oc, orow), marker_fit=None)
                        print(f"  저장 완료 → {vd.COLORS_PATH}")
                        print("  ※ 이건 그 칸에서만 정확합니다. 판 전체를 맞추려면")
                        print("    'markcal add' 를 3칸 이상에서 한 뒤 'markcal fit'")
                    else:
                        print("  markcal auto       ⭐ 팔이 6칸을 돌며 자동 측정 (권장)")
                        print("  markcal add <칸>   흡착컵을 그 칸 중심에 맞춘 뒤 기록")
                        print("                     (nudge 로 맞춘다: nudge 2 -3)")
                        print("  markcal fit        3점 이상 모이면 계산·저장")
                        print("  markcal clear      기록·보정 초기화")
                        print("  markcal <칸>       한 칸만 보는 간이 방식")
                        print(f"  (현재 기록 {len(markcal_pts)}점)")

                elif p[0] in ("show", "view"):
                    if detector is None:
                        print("  --camera 옵션으로 실행해야 합니다"); continue
                    arg = p[1] if len(p) == 2 else "on"
                    if arg in ("off", "0", "stop"):
                        live.stop(); print("  라이브 뷰 종료")
                    elif arg in ("on", "1"):
                        live.start()
                        print("  라이브 뷰 시작 — 창을 띄운 채로 계속 명령할 수 있습니다")
                        print("  ('show off'로 종료. 창의 X 버튼으로는 닫지 마세요)")
                    else:
                        try: show_live(float(arg))       # 'show 10' = 10초만
                        except ValueError: print("  show | show off | show <초>")
                elif p[0] == "refcap":
                    # 빈 판 기준 영상 찍기 — 밝기 방식으로는 검은 기물과
                    # 어두운 칸을 구분할 수 없어서 반드시 필요하다.
                    if detector is None:
                        print("  --camera 옵션으로 실행해야 합니다"); continue
                    print("  ⚠️ 체스판에서 기물을 전부 치우세요.")
                    print("     팔은 park(Z자)로 — 그림자까지 같이 기록됩니다.")
                    if input("     준비됐으면 Enter (취소는 x) > ").strip().lower() == "x":
                        continue
                    if live is not None and live.running:
                        live.stop(); print("  (라이브 뷰 잠시 종료)")
                    detector.capture_empty_reference()
                    print("  이제 기물을 놓고 'board' 로 확인하세요.")

                elif p[0] == "board":
                    # 기물 인식이 되는지 / 판 방향이 맞는지 그 자리에서 확인
                    if detector is None:
                        print("  --camera 옵션으로 실행해야 합니다"); continue
                    import vision.detect as vd
                    from vision.detect import format_state
                    expect = [["empty"] * 8 for _ in range(8)]
                    for c in range(8):
                        expect[0][c] = expect[1][c] = "white"
                        expect[6][c] = expect[7][c] = "black"
                    obs = det_src.get_board_state()
                    n = sum(obs[r][c] == expect[r][c]
                            for r in range(8) for c in range(8))
                    print(f"  ROBOT_SIDE=\"{vd.ROBOT_SIDE}\"  "
                          f"OCC_DIFF_THRESH={vd.OCC_DIFF_THRESH}")
                    print("  (대문자 = 시작 배치와 다른 칸)")
                    print(format_state(obs, expect))
                    print(f"  시작 배치와 {n}/64 칸 일치")
                    print(detector.explain_board_state())
                    if n < 56:
                        print("  → python vision/check_board.py --camera <번호> "
                              "--all-sides 로 방향까지 확인해 보세요")

                elif p[0] == "margin":
                    if detector is None:
                        print("  --camera 옵션으로 실행해야 합니다"); continue
                    # 판 바깥을 카메라가 몇 칸까지 보는지. 외곽 칸에서 마커가
                    # 아예 화면에 안 나올 때 원인이 여기다.
                    print("  " + detector.margin_report().replace("\n", "\n  "))

                elif p[0] == "mark":
                    if detector is None:
                        print("  --camera 옵션으로 실행해야 합니다"); continue
                    show()
                    mk = det_src.find_marker()
                    if mk is None:
                        import vision.detect as vd
                        a = detector._last_marker_area
                        print("  마커를 못 찾음")
                        if a > vd.MARKER_MAX_AREA:
                            print(f"    → 너무 큰 덩어리({a:.0f}px)를 잡음. 체스판이나 반사를"
                                  f" 마커로 오인한 것입니다.")
                            print(f"       MARKER_HSV_RANGES의 채도(S) 최소값을 올리세요.")
                        # 원본 화면까지 뒤져서 '왜' 안 보이는지 가른다.
                        # 색 문제인지, 탑뷰 밖으로 나간 건지, 카메라 화각 밖인지는
                        # 고치는 방법이 완전히 다르다.
                        print("  " + detector.marker_debug())
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
                    print("        check | zoff 8 | suction 1 | set 85 140 180 | home | q")
                    print("        a 85 140 185 | limit | probe s3 180 200")

        except ValueError as e:
            print(f"  [실패] {e}")
            print("     ※ 랭크1~2(사람 진영)는 현재 배치에서 팔이 닿지 않습니다."
                  " 랭크3~8로 시도하세요.")
        except KeyboardInterrupt:
            print("\n  [중단] 현재 위치에서 멈춤. 전원 확인하세요.")

    if detector is not None:
        if live is not None:
            live.stop()
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
