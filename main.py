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


# ─────────────────────────────────────────
# RL 모델 로드
# ─────────────────────────────────────────
def load_rl_model():
    try:
        from stable_baselines3 import PPO
        # 2단계 → 1단계 순으로 시도
        for name in ["stage2_final", "stage1_final"]:
            path = os.path.join(MODEL_DIR, name + ".zip")
            if os.path.exists(path):
                model = PPO.load(path)
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
def run_game(args, arm, detector, rl_model):
    from utils.ik_solver   import inverse_kinematics, chess_square_to_xyz
    from utils.lagrange    import check_torque_feasibility
    from utils.chess_utils import stockfish_move, move_to_squares, is_capture

    board = chess.Board()

    print("\n" + "="*50)
    print("체스 게임 시작! 당신=WHITE / 로봇=BLACK")
    print("="*50 + "\n")

    while not board.is_game_over():
        print(f"\n현재 보드:\n{board}\n")

        # ── 사람 차례 ──
        if board.turn == chess.WHITE:
            if detector:
                result = None
                while result is None:
                    result = detector.detect_human_move()
                from_sq, to_sq = result
                move = chess.Move(
                    chess.square(from_sq[0], from_sq[1]),
                    chess.square(to_sq[0],   to_sq[1]),
                )
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
            from_sq, to_sq = move_to_squares(move)
            capture = is_capture(board, move)

            board.push(move)
            print(f"  로봇: {move}  {'(기물 잡기)' if capture else ''}")

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

            # RL 보정 (observation 12차원: IK각 + 실제각 + 목표xyz + 라그랑주토크)
            correction = None
            if rl_model is not None:
                from utils.lagrange import required_torque
                tau = required_torque([q1,q2,q3], [0,0,0], [0.1,0.1,0.1])
                tau_clipped = np.clip(tau, -3.0, 3.0)
                obs = np.array([q1, q2, q3, 0, 0, 0, x, y, z,
                                tau_clipped[0], tau_clipped[1], tau_clipped[2]], dtype=np.float32)
                delta, _ = rl_model.predict(obs, deterministic=True)
                correction = delta
                print(f"  [RL 보정] Δq = {np.round(delta, 4)}")

            # 실행
            arm.execute_move(from_sq, to_sq, is_capture=capture,
                             rl_correction=correction)

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
    parser.add_argument("--camera",   type=int, default=0)
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
    arm = RealArm(port=args.port, sim=sim_mode)

    # RL 모델 로드
    rl_model = load_rl_model()

    # 카메라 + 캘리브레이션
    detector = None
    if os.path.exists(CALIB_PATH):
        try:
            from vision.detect import ChessBoardDetector
            detector = ChessBoardDetector(CALIB_PATH, args.camera)
            board_state = detector.get_board_state()
            print(f"[카메라] 체스판 인식 완료")
        except Exception as e:
            print(f"[카메라] 초기화 실패: {e} → 터미널 입력 모드로 대체")
            detector = None
    else:
        print(f"[카메라] calibration.json 없음 → 터미널 입력 모드")

    run_game(args, arm, detector, rl_model)

    if detector:
        detector.close()
    arm.close()


if __name__ == "__main__":
    main()
