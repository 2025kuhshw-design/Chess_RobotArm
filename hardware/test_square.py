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

    cur = None       # 현재 대상 칸 (col,row)
    holding = False  # 기물을 흡착해 들고 있는 중인가

    WIN = "test_square view"

    class LiveView:
        """백그라운드 스레드로 카메라 창을 계속 띄운다.

        input()은 블로킹이라 메인 스레드에서는 GUI 이벤트를 돌릴 수 없다.
        그래서 별도 스레드가 read→그리기→waitKey 를 반복한다.
        ⚠️ 카메라를 두 곳에서 읽으면 프레임을 서로 뺏는다. 그래서 이 스레드가
           읽기를 독점하고, 마커 검출은 여기 저장된 최신 프레임을 쓴다.
        """
        def __init__(self, det):
            import threading
            self.det = det
            self._frame = None
            self._lock = threading.Lock()
            self._stop = threading.Event()
            self._th = None

        def start(self):
            import threading
            if self._th is not None:
                return
            self._stop.clear()
            self._th = threading.Thread(target=self._loop, daemon=True)
            self._th.start()

        def stop(self):
            if self._th is None:
                return
            self._stop.set()
            self._th.join(timeout=2)
            self._th = None
            try:
                import cv2
                cv2.destroyWindow(WIN)
                cv2.waitKey(1)
            except Exception:
                pass

        @property
        def running(self):
            return self._th is not None

        def latest(self):
            with self._lock:
                return None if self._frame is None else self._frame.copy()

        def _loop(self):
            import cv2
            while not self._stop.is_set():
                ok, f = self.det.cap.read()
                if not ok:
                    continue
                with self._lock:
                    self._frame = f
                try:
                    cv2.imshow(WIN, self.det.overlay_debug(f))
                    cv2.waitKey(30)
                except Exception:
                    break     # GUI를 못 쓰는 환경이면 조용히 종료

    live = LiveView(detector) if detector is not None else None

    class _DetProxy:
        """정렬 루틴이 라이브 뷰의 최신 프레임을 쓰도록 감싼다.
        뷰가 꺼져 있으면 원래대로 detector가 직접 읽는다."""
        def __init__(self, det, view):
            self.det, self.view = det, view

        def find_marker(self):
            if self.view is not None and self.view.running:
                for _ in range(5):          # 최신 프레임으로 몇 번 재시도
                    f = self.view.latest()
                    if f is not None:
                        r = self.det.find_marker(f)
                        if r is not None:
                            return r
                    time.sleep(0.05)
                return None
            return self.det.find_marker()

        def __getattr__(self, k):
            return getattr(self.det, k)

    det_src = _DetProxy(detector, live) if detector is not None else None

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

    def align(sq, suction=False):
        """카메라가 있으면 하강 전에 정렬(하며 매 반복마다 화면 갱신).
        ⚠️ 기물을 들고 있으면 suction=True — 안 넘기면 정렬 중 떨어뜨린다."""
        if detector is None:
            return
        align_over_square(arm, det_src, sq[0], sq[1], _safe_lift(*sq),
                          view_cb=show_once, suction=suction)
        show_once()

    def aligner_cb(col, row, lift, suction):
        """execute_move가 하강 직전마다 부르는 콜백."""
        if detector is None:
            return
        align_over_square(arm, det_src, col, row, lift,
                          view_cb=show_once, suction=suction)
        show_once()

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
    print("  진단: a 110 140 180 ← 서보 각도로 직접 이동 (한 축만 바꿔서 시험)")
    if detector is not None:
        print("  카메라 명령: show(켜기) / show off(끄기) | mark 마커위치 | "
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

            elif p[0] == "a" and len(p) == 4:
                # 서보 각도로 직접 이동 (램프 적용). 관절 하나만 바꿔서
                # '그 모터가 실제로 도는지' 격리 진단할 때 쓴다.
                tgt = [int(v) for v in p[1:]]
                print(f"  A{tgt[0]},{tgt[1]},{tgt[2]} 로 이동 "
                      f"(현재 가정 A{arm._cur[0]},{arm._cur[1]},{arm._cur[2]})")
                arm._ramp_send(tgt[0], tgt[1], tgt[2], holding)
                print(f"  → 완료. 현재 A{arm._cur[0]},{arm._cur[1]},{arm._cur[2]}")

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
                    mk = det_src.find_marker()
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
                    print("        check | zoff 8 | suction 1 | set 85 140 180 | home | q")

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
