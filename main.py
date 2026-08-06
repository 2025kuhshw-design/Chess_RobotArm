"""
체스 로봇팔 통합 실행 엔트리포인트

실행 모드:
  python main.py --mode sim        아두이노 없이 시뮬레이션
  python main.py --mode real       실제 하드웨어 연결
  python main.py --mode vision     카메라 인식 테스트
  python main.py --mode chess      체스 AI 터미널 테스트
"""

import argparse
import os
import sys
import chess
import numpy as np

# ─────────────────────────────────────────
# 하이퍼파라미터 / 상수
# ─────────────────────────────────────────
DEFAULT_PORT       = "COM3"
DEFAULT_STOCKFISH  = "/usr/games/stockfish"
MODEL_DIR          = os.path.join(os.path.dirname(__file__), "models", "correction_model")
CALIB_PATH         = os.path.join(os.path.dirname(__file__), "vision", "calibration.json")

sys.path.insert(0, os.path.dirname(__file__))
from sim.env_simple import TAU_OBS_LIMIT


# ─────────────────────────────────────────
# RL 모델 로드
# ─────────────────────────────────────────
def load_rl_model():
    try:
        from stable_baselines3 import PPO
        # 학습 당시 스케줄(learning_rate 함수)이 cloudpickle로 값 저장돼 있어
        # 다른 환경에서 로드 시 호출하면 깨짐 → 무해한 상수로 대체 (추론엔 무관)
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": 0.0,
        }
        # 2단계 → 1단계 순으로 시도
        for name in ["stage2_final", "stage1_final"]:
            path = os.path.join(MODEL_DIR, name + ".zip")
            if os.path.exists(path):
                model = PPO.load(path, custom_objects=custom_objects)
                print(f"[RL] 모델 로드: {path}")
                return model
        print("[RL] 저장된 모델 없음 → 보정 없이 실행 (학습 후 재실행 권장)")
        return None
    except Exception as e:
        print(f"[RL] 모델 로드 실패: {e} → 보정 없이 실행")
        return None


# ─────────────────────────────────────────
# 게임 루프
# ─────────────────────────────────────────
# RL 보정 한 관절당 최대 허용 크기 (rad).
# ⚠️ 학습 환경의 행동 범위는 ±0.2rad(11.5°)인데, 정작 학습에 쓴 실제 오차
#    노이즈는 ±0.03rad(1.7°)뿐이었다. 즉 행동 범위가 고쳐야 할 오차보다
#    7배 크다. 정책이 덜 수렴하면 큰 값을 내놓고, 그러면 팔이 크게 빗나간다.
#      한 관절 0.20rad → 끝단 3.4cm (1.2칸)
#      세 관절 모두   → 끝단 7.1cm (2.4칸)
#    실제로 e7e5 가 e파일 근처도 못 가고, f8 을 한 칸 빗나갔다.
#    그래서 '학습이 상정한 오차 크기'로 잘라낸다.
RL_MAX_DELTA = 0.05      # rad (2.9°) — 끝단 약 0.85cm


def make_rl_corrector(rl_model, verbose=True):
    """목표 좌표 (x,y,z) → 관절 보정 Δq 를 돌려주는 함수를 만든다.

    관측 12차원: [IK 관절각 3, 실제 관절각 3(모르므로 0), 목표 xyz 3,
                  라그랑주 필요 토크 3]
    학습 때 VecNormalize(norm_obs=False)였으므로 관측은 정규화하지 않는다.
    """
    from utils.ik_solver import inverse_kinematics
    from utils.lagrange import required_torque

    def corrector(x, y, z):
        q1, q2, q3 = inverse_kinematics(x, y, z)
        tau = np.clip(required_torque([q1, q2, q3], [0, 0, 0], [0.1, 0.1, 0.1]),
                      -TAU_OBS_LIMIT, TAU_OBS_LIMIT)
        obs = np.array([q1, q2, q3, 0, 0, 0, x, y, z,
                        tau[0], tau[1], tau[2]], dtype=np.float32)
        raw, _ = rl_model.predict(obs, deterministic=True)
        delta = np.clip(raw, -RL_MAX_DELTA, RL_MAX_DELTA)
        if verbose:
            if np.any(np.abs(raw) > RL_MAX_DELTA):
                print(f"    [RL 보정] Δq = {np.round(delta, 4)} "
                      f"(원래 {np.round(raw, 3)} → ±{RL_MAX_DELTA}로 잘림)")
            else:
                print(f"    [RL 보정] Δq = {np.round(delta, 4)}")
        return delta

    return corrector


def _startup_vision_check(detector, board):
    """시작 배치를 제대로 읽는지 게임 시작 전에 확인한다.

    기물 인식이나 판 방향이 틀렸다면 여기서 드러난다. 게임 도중에
    "이동 감지 실패"만 반복되는 것보다 훨씬 빨리 알 수 있다.
    board: 실제 시작 국면 (--fen 으로 바꿀 수 있다)
    """
    import vision.detect as vd
    from vision.detect import format_state, board_to_state

    expect = board_to_state(board)

    if getattr(detector, "_ref", None) is None:
        print("[카메라] ⚠️ 빈 판 기준 영상이 없습니다 → 밝기 방식으로 판독합니다.")
        print("        밝기만으로는 검은 기물과 어두운 칸을 구분하지 못합니다.")
        print("        기물을 다 치우고 한 번만 찍어두세요:")
        print("          python vision/check_board.py --camera <번호> --capture-empty")

    obs = detector.get_board_state(expect=expect)
    n = sum(obs[r][c] == expect[r][c] for r in range(8) for c in range(8))
    print(f"[카메라] 시작 배치 판독: 64칸 중 {n}칸 일치 "
          f"(ROBOT_SIDE=\"{vd.ROBOT_SIDE}\", 방식={detector.detection_mode()})")
    if n >= 60:
        return
    print("  ⚠️ 판독이 지정한 시작 배치와 많이 다릅니다.")
    print("     ① 판에 실제로 놓은 배치와 --fen 이 같은지 먼저 확인하세요.")
    print("        (기물 32개를 표준 배치로 놓았다면 --fen 을 아예 빼야 합니다)")
    print("     ② 배치가 맞는데도 다르면 인식 문제입니다:")
    print(format_state(obs, expect))
    print("     python vision/check_board.py --camera <번호> --all-sides")


def read_human_move(detector, board):
    """카메라로 사람이 둔 수를 읽는다. 실패하면 직접 입력받는다.

    합법수마다 '그 수를 뒀을 때의 배치'를 만들어 화면과 대조하고, 가장 잘
    맞는 것을 고른다. 칸 몇 개를 잘못 읽어도 정답이 살아남고, 기물을 잡는
    수(도착칸이 원래 차 있던 경우)도 제대로 읽힌다.
    """
    from vision.detect import board_to_state, format_state

    print("  수를 두고 Enter를 누르세요 (u=직접 입력, s=화면 확인) > ", end="", flush=True)
    ans = input().strip().lower()

    if ans == "u":
        return _ask_uci(board)

    move, score, cands = detector.detect_move_with_rules(board)

    if ans == "s" or move is None:
        obs = getattr(detector, "_last_observed", None)
        if obs is not None:
            print("  카메라가 본 배치 (대문자=현재 기보와 다른 칸):")
            print(format_state(obs, board_to_state(board)))
        if ans == "s":
            # 임계값을 조정할 수 있게 숫자까지 보여준다
            print(detector.explain_board_state())

    if move is not None:
        gap = score - cands[1][0] if len(cands) > 1 else score
        print(f"  이동 감지: {move.uci()}  (2위 후보보다 {gap:.1f}점 앞섬)")
        return move

    # 확신이 없으면 후보를 보여주고 사람이 고른다 — 게임이 멈추지 않게.
    if not cands:
        print("  [경고] 화면이 안정되지 않았습니다 (팔·손이 보드를 가리는 중?)")
        return None
    print("  어느 수인지 확신이 서지 않습니다. 가장 비슷한 후보:")
    for i, (sc, mv) in enumerate(cands[:5], 1):
        print(f"    {i}) {mv.uci()}   일치도 {sc:.1f}")
    sel = input("  번호를 고르거나 수를 직접 입력하세요 (예: 1 또는 e2e4) > ").strip()
    if sel.isdigit() and 1 <= int(sel) <= min(5, len(cands)):
        return cands[int(sel) - 1][1]
    try:
        mv = chess.Move.from_uci(sel)
        return mv if mv in board.legal_moves else None
    except Exception:
        return None


def _ask_uci(board):
    uci = input("  수를 입력하세요 (예: e2e4) > ").strip()
    try:
        mv = chess.Move.from_uci(uci)
    except Exception:
        return None
    return mv if mv in board.legal_moves else None


def run_game(args, arm, detector, rl_model):
    from utils.ik_solver   import inverse_kinematics, chess_square_to_xyz
    from utils.lagrange    import check_torque_feasibility
    from utils.chess_utils import (stockfish_move, move_to_squares,
                                   is_capture, physical_ops)

    try:
        board = chess.Board(args.fen)
    except ValueError as e:
        print(f"[main] FEN 오류: {e}"); return
    if not board.is_valid():
        print("[main] 그 FEN 은 유효한 국면이 아닙니다 "
              "(양쪽 킹이 하나씩 있어야 합니다)"); return

    # RL 보정 — 목표마다 계산. 정렬과 실행이 같은 것을 써야 자세가 일치한다.
    corrector = make_rl_corrector(rl_model) if rl_model else None

    # 카메라 폐루프 정렬 콜백 — 하강 직전마다 마커를 보고 위치를 보정
    aligner = None
    if detector is not None:
        from hardware.visual_align import align_over_square
        def aligner(col, row, lift, suction):
            # ⚠️ 반환값(보정 오프셋)을 반드시 그대로 돌려줄 것.
            #    여기서 삼켜버리면 execute_move 가 보정을 못 받아
            #    하강할 때 원래 칸 좌표로 되돌아간다(= 보정 무효).
            return align_over_square(arm, detector, col, row, lift,
                                     suction=suction, corrector=corrector)

    print("\n" + "="*50)
    print("체스 게임 시작! 당신=WHITE / 로봇=BLACK")
    print("="*50 + "\n")

    while not board.is_game_over():
        print(f"\n현재 보드:\n{board}\n")

        # ── 사람 차례 ──
        if board.turn == chess.WHITE:
            if detector:
                # 보드를 읽기 전에 팔을 PARK로 물린다.
                # 팔이 보드 위에 있으면 카메라를 가려 기물 인식이 깨진다.
                try:
                    arm.home()
                except Exception as e:
                    print(f"  [경고] park 복귀 실패: {e}")
                move = read_human_move(detector, board)
                if move is None:
                    continue
            else:
                while True:
                    uci = input("당신의 수를 입력하세요 (예: e2e4): ").strip()
                    try:
                        move = chess.Move.from_uci(uci)
                        break
                    except Exception:
                        print("  잘못된 형식입니다. 다시 입력하세요.")

            if move not in board.legal_moves:
                print(f"  [경고] 불법 이동: {move}. 다시 시도하세요.")
                continue

            board.push(move)
            print(f"  사람: {move}")

        # ── 로봇 차례 ──
        else:
            print("  로봇이 생각 중...")
            move = stockfish_move(board, args.stockfish)
            if move is None:
                print("  [경고] 엔진이 수를 못 냈습니다 (게임 종료?)"); break
            from_sq, to_sq = move_to_squares(move)
            capture = is_capture(board, move)
            # ⚠️ 규칙상 한 수라도 물리적으로는 동작이 여러 개일 수 있다
            #    (캐슬링=킹+룩, 앙파상=옆 칸 폰 제거). board.push 前에 분해해야
            #    is_castling/is_en_passant 판정이 된다.
            ops, notes = physical_ops(board, move)

            board.push(move)
            print(f"  로봇: {move}  {'(기물 잡기)' if capture else ''}")
            for n in notes:
                print(f"    {n}")

            # 역기구학
            x, y, z = chess_square_to_xyz(*from_sq)
            try:
                q1, q2, q3 = inverse_kinematics(x, y, z)
            except ValueError as e:
                print(f"  [경고] IK 실패: {e}")
                continue

            # 라그랑주 토크 검증
            feasible, tau, _ = check_torque_feasibility(
                [q1, q2, q3], [0, 0, 0], [0.1, 0.1, 0.1]
            )
            if not feasible:
                print(f"  [경고] 토크 한계 초과: {np.round(tau, 3)} N·m")

            correction = corrector

            # 실행 (IK 실패 등으로 게임 전체가 죽지 않도록 방어)
            try:
                arm.execute_move(from_sq, to_sq, is_capture=capture,
                                 rl_correction=correction,
                                 confirm=args.confirm,
                                 aligner=aligner,
                                 ops=ops)
            except ValueError as e:
                print(f"  [경고] 팔 동작 실패(도달 불가): {e}")
                print(f"         수는 보드에 반영됨. 기물을 손으로 옮겨주세요.")
                try:
                    arm.home()
                except Exception:
                    pass

    # 게임 종료
    print("\n" + "="*50)
    result = board.result()
    if result == "1-0":
        print("결과: 사람(WHITE) 승리!")
    elif result == "0-1":
        print("결과: 로봇(BLACK) 승리!")
    else:
        print(f"결과: 무승부 ({result})")
    print("="*50)


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="체스 로봇팔 통합 실행")
    parser.add_argument("--mode",      type=str, default="sim",
                        choices=["sim", "real", "vision", "chess"])
    parser.add_argument("--port",      type=str, default=DEFAULT_PORT)
    parser.add_argument("--stockfish", type=str, default=DEFAULT_STOCKFISH)
    parser.add_argument("--camera", type=int, default=0,
                        help="카메라 번호 (이 PC는 보통 1)")
    parser.add_argument("--start", type=int, nargs=3, metavar=("S1","S2","S3"),
                        help="시작 시 팔의 실제 서보 각도 "
                             "(생략하면 PARK_POSE에 있다고 가정)")
    parser.add_argument("--step", type=int, default=None,
                        help="한 스텝당 각도(도). 작을수록 느리고 안전")
    parser.add_argument("--step-delay", type=float, default=None,
                        help="스텝 간 대기(s). 클수록 느리고 안전")
    parser.add_argument("--no-startup", action="store_true",
                        help="기동 절차를 건너뛴다 (팔이 이미 PARK에 잡혀 있을 때)")
    parser.add_argument("--fen", type=str, default=chess.STARTING_FEN,
                        help="시작 국면 (FEN). 기물이 32개가 안 될 때 실제로 "
                             "놓은 배치를 지정한다. "
                             "예: \"4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1\"")
    parser.add_argument("--view", action="store_true", default=True,
                        help="카메라 화면을 계속 띄운다 (test_square의 show와 동일). "
                             "기본 켜짐 — 끄려면 --no-view")
    parser.add_argument("--no-view", dest="view", action="store_false",
                        help="카메라 화면을 띄우지 않는다")
    parser.add_argument("--rl", action="store_true",
                        help="RL 보정을 켠다 (기본 꺼짐). ⚠️ 카메라 폐루프 보정이 "
                             "'실제로 잰 오차'를 고치는 반면 RL 보정은 재지 않은 "
                             "값을 얹는 것이라, 둘을 같이 쓰면 오히려 어긋난다")
    parser.add_argument("--confirm", action="store_true",
                        help="집기/놓기 전에 칸 위에서 멈춰 사람 확인을 받는다 "
                             "(첫 실전 권장. 카메라 자동보정이 아니라 육안 확인)")
    args = parser.parse_args()

    print(f"\n[main] 실행 모드: {args.mode}")

    # ── vision 모드 ──
    if args.mode == "vision":
        from vision.detect import ChessBoardDetector
        try:
            detector = ChessBoardDetector(CALIB_PATH, args.camera)
        except FileNotFoundError as e:
            print(e)
            return
        import cv2
        print("실시간 디버그 뷰 실행 중... q=종료")
        while True:
            ret, frame = detector.cap.read()
            if not ret:
                break
            debug = detector.overlay_debug(frame)
            cv2.imshow("Chess Vision", debug)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        detector.close()
        cv2.destroyAllWindows()
        return

    # ── chess 모드 (터미널만) ──
    if args.mode == "chess":
        from hardware.arm_controller import RealArm
        arm = RealArm(sim=True)
        run_game(args, arm, detector=None, rl_model=None)
        return

    # ── sim / real 모드 ──
    from hardware.arm_controller import RealArm

    sim_mode = args.mode == "sim"
    arm = RealArm(port=args.port, sim=sim_mode,
                  start_pose=tuple(args.start) if args.start else None,
                  ramp_step=args.step, ramp_delay=args.step_delay,
                  auto_home=False)

    # 기동 절차 (팔이 처진 상태에서 안전하게 시작)
    if not sim_mode and not args.no_startup:
        from hardware.arm_controller import interactive_startup
        if not interactive_startup(arm):
            arm.close()
            print("[main] 기동 중단."); return
    else:
        arm.home()

    # RL 모델 로드 (--rl 을 줬을 때만)
    if args.rl:
        rl_model = load_rl_model()
    else:
        rl_model = None
        print("[RL] 보정 꺼짐 (켜려면 --rl). 카메라 폐루프 보정만 사용합니다.")

    # 카메라 + 캘리브레이션
    detector = None
    live = None
    if os.path.exists(CALIB_PATH):
        try:
            from vision.detect import ChessBoardDetector
            detector = ChessBoardDetector(CALIB_PATH, args.camera)
            _startup_vision_check(detector, chess.Board(args.fen))
            if args.view:
                # 라이브 뷰가 카메라 읽기를 독점하므로, 인식은 프록시를 통해
                # 그 최신 프레임을 빌려 쓴다 (프레임 뺏김 방지).
                from vision.live_view import LiveView, DetectorProxy
                live = LiveView(detector)
                live.start()
                detector = DetectorProxy(detector, live)
                print("[카메라] 라이브 뷰 켜짐 — 창을 닫지 마세요")
        except Exception as e:
            print(f"[카메라] 초기화 실패: {e} → 터미널 입력 모드로 대체")
            detector = None
    else:
        print(f"[카메라] calibration.json 없음 → 터미널 입력 모드")

    try:
        run_game(args, arm, detector, rl_model)
    finally:
        if live is not None:
            live.stop()
        if detector:
            detector.close()
        arm.close()


if __name__ == "__main__":
    main()
