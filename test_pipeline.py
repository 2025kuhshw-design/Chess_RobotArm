"""
파이프라인 단계별 테스트 스크립트
실행: python test_pipeline.py [--test all|urdf|ik|lagrange|vision|serial|rl|full]
"""

import argparse
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

PASS = "✅ PASS"
FAIL = "❌ FAIL"


# ─────────────────────────────────────────
# test 1: URDF 로드 + PyBullet 시뮬 확인
# ─────────────────────────────────────────
def test_urdf():
    print("\n[test_urdf] URDF 파일 로드 + PyBullet 시뮬 확인")
    try:
        import pybullet as p
        import pybullet_data

        client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        urdf_path = os.path.join("setup", "urdf", "robot_simple.urdf")
        if not os.path.exists(urdf_path):
            from setup.stl_to_urdf import generate_simple_urdf
            joint_positions = [[0,0,0.05],[0,0,0.33],[0,0,0.55],[0,0,0.63]]
            generate_simple_urdf(joint_positions, urdf_path)

        robot_id = p.loadURDF(urdf_path, basePosition=[0,0,0], useFixedBase=True)
        n_joints = p.getNumJoints(robot_id)
        p.stepSimulation()
        p.disconnect(client)

        print(f"  URDF 로드 성공: {urdf_path}  관절 수={n_joints}")
        print(f"  {PASS}")
        return True
    except Exception as e:
        print(f"  오류: {e}")
        print(f"  {FAIL}")
        return False


# ─────────────────────────────────────────
# test 2: 역기구학 → 순기구학 왕복 검증
# ─────────────────────────────────────────
def test_ik():
    print("\n[test_ik] IK → FK 왕복 검증 (오차 < 1mm)")
    from utils.ik_solver import inverse_kinematics, forward_kinematics

    test_targets = [
        (0.20,  0.05,  0.10),
        (0.30, -0.10,  0.05),
        (0.15,  0.10,  0.20),
        (0.25,  0.08,  0.12),
        (0.18, -0.05,  0.08),
    ]

    all_pass = True
    for x, y, z in test_targets:
        try:
            q1, q2, q3 = inverse_kinematics(x, y, z)
            fx, fy, fz = forward_kinematics(q1, q2, q3)
            err_mm = math.sqrt((x-fx)**2 + (y-fy)**2 + (z-fz)**2) * 1000
            ok = err_mm < 1.0
            if not ok:
                all_pass = False
            print(f"  ({x:.2f},{y:.2f},{z:.2f}) → 오차={err_mm:.4f}mm  {'OK' if ok else 'FAIL'}")
        except ValueError as e:
            print(f"  ({x},{y},{z}) → 도달 불가: {e}")
            all_pass = False

    print(f"  {PASS if all_pass else FAIL}")
    return all_pass


# ─────────────────────────────────────────
# test 3: 라그랑주 토크 계산 + 한계 확인
# ─────────────────────────────────────────
def test_lagrange():
    print("\n[test_lagrange] 토크 계산값 + 한계 초과 여부 확인")
    from utils.lagrange import required_torque, check_torque_feasibility, SERVO_LIMIT

    cases = [
        {"q": [0.0, 0.3, 0.5], "q_dot": [0.1,0.1,0.1], "q_ddot": [0.1,0.1,0.1]},
        {"q": [0.5, 0.8, 1.0], "q_dot": [0.2,0.2,0.2], "q_ddot": [0.3,0.3,0.3]},
    ]

    all_pass = True
    for i, tc in enumerate(cases):
        q      = np.array(tc["q"])
        q_dot  = np.array(tc["q_dot"])
        q_ddot = np.array(tc["q_ddot"])
        feasible, tau, q_ddot_used = check_torque_feasibility(q, q_dot, q_ddot)
        print(f"  케이스{i+1}: tau={np.round(tau,3)}  한계이내={feasible}")
        if np.any(np.isnan(tau)):
            all_pass = False

    print(f"  {PASS if all_pass else FAIL}")
    return all_pass


# ─────────────────────────────────────────
# test 4: 웹캠 연결 + 기물 인식 테스트
# ─────────────────────────────────────────
def test_vision():
    print("\n[test_vision] 웹캠 연결 + 기물 인식 테스트")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  카메라를 열 수 없음 (카메라 미연결)")
            print(f"  {FAIL} (하드웨어 없음 — 카메라 연결 후 재테스트)")
            return False

        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("  프레임 읽기 실패")
            print(f"  {FAIL}")
            return False

        calib_path = os.path.join("vision", "calibration.json")
        if not os.path.exists(calib_path):
            print(f"  calibration.json 없음: python vision/calibrate.py 먼저 실행")
            print(f"  {FAIL} (캘리브레이션 필요)")
            return False

        from vision.detect import ChessBoardDetector
        detector = ChessBoardDetector(calib_path, camera_index=0)
        board = detector.get_board_state()
        detector.close()

        empty_count = sum(1 for row in board for cell in row if cell == "empty")
        print(f"  보드 인식 완료: 빈칸={empty_count}/64")
        print(f"  {PASS}")
        return True
    except Exception as e:
        print(f"  오류: {e}")
        print(f"  {FAIL}")
        return False


# ─────────────────────────────────────────
# test 5: 아두이노 시리얼 ping 테스트
# ─────────────────────────────────────────
def test_serial(port="COM3"):
    print(f"\n[test_serial] 아두이노 시리얼 ping 테스트 (포트: {port})")
    try:
        import serial
        ser = serial.Serial(port, 9600, timeout=3)
        import time
        time.sleep(2)
        ser.write(b"A90,90,90,0\n")
        resp = ser.readline().decode().strip()
        ser.close()

        if resp == "OK":
            print(f"  응답: {resp}")
            print(f"  {PASS}")
            return True
        else:
            print(f"  예상치 못한 응답: '{resp}'")
            print(f"  {FAIL}")
            return False
    except Exception as e:
        print(f"  오류: {e}")
        print(f"  {FAIL} (아두이노 미연결 — 연결 후 재테스트)")
        return False


# ─────────────────────────────────────────
# test 6: RL 모델 로드 + 64칸 추론 성공률
# ─────────────────────────────────────────
def test_rl_model():
    print("\n[test_rl_model] RL 보정 모델 로드 + 64칸 추론 성공률")
    model_dir = os.path.join("models", "correction_model")

    try:
        from stable_baselines3 import PPO
        model = None
        for name in ["stage2_final", "stage1_final"]:
            path = os.path.join(model_dir, name + ".zip")
            if os.path.exists(path):
                model = PPO.load(path)
                print(f"  모델 로드: {path}")
                break

        if model is None:
            print(f"  모델 없음 — 먼저 python sim/train_correction.py --stage 1 실행")
            print(f"  {FAIL} (학습 필요)")
            return False

        from utils.ik_solver import chess_square_to_xyz, inverse_kinematics
        success = 0
        for col in range(8):
            for row in range(8):
                x, y, z = chess_square_to_xyz(col, row)
                try:
                    q1, q2, q3 = inverse_kinematics(x, y, z)
                    obs = np.array([q1, q2, q3, 0, 0, 0, x, y, z], dtype=np.float32)
                    action, _ = model.predict(obs, deterministic=True)
                    if action is not None and len(action) == 3:
                        success += 1
                except Exception:
                    pass

        rate = success / 64 * 100
        print(f"  64칸 추론 성공률: {success}/64 = {rate:.1f}%")
        ok = success >= 60
        print(f"  {PASS if ok else FAIL}")
        return ok
    except Exception as e:
        print(f"  오류: {e}")
        print(f"  {FAIL}")
        return False


# ─────────────────────────────────────────
# test 7: sim 모드 3수 전체 파이프라인
# ─────────────────────────────────────────
def test_full():
    print("\n[test_full] sim 모드 3수짜리 게임 파이프라인")
    try:
        import chess
        from hardware.arm_controller import RealArm
        from utils.chess_utils       import move_to_squares, is_capture
        from utils.ik_solver         import inverse_kinematics, chess_square_to_xyz
        from utils.lagrange          import check_torque_feasibility

        arm   = RealArm(sim=True)
        board = chess.Board()

        # 미리 정의된 3수 게임 (e2e4, e7e5, d2d4)
        moves_uci = ["e2e4", "e7e5", "d2d4"]

        for i, uci in enumerate(moves_uci):
            move    = chess.Move.from_uci(uci)
            capture = is_capture(board, move)
            from_sq, to_sq = move_to_squares(move)

            board.push(move)

            x, y, z = chess_square_to_xyz(*from_sq)
            q1, q2, q3 = inverse_kinematics(x, y, z)
            feasible, tau, _ = check_torque_feasibility([q1,q2,q3],[0,0,0],[0.1,0.1,0.1])

            arm.execute_move(from_sq, to_sq, is_capture=capture)
            print(f"  수 {i+1}: {uci}  torque_ok={feasible}")

        arm.close()
        print(f"  {PASS}")
        return True
    except Exception as e:
        print(f"  오류: {e}")
        import traceback; traceback.print_exc()
        print(f"  {FAIL}")
        return False


# ─────────────────────────────────────────
# 단독 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all",
                        choices=["all","urdf","ik","lagrange","vision","serial","rl","full"])
    parser.add_argument("--port", type=str, default="COM3")
    args = parser.parse_args()

    results = {}

    def run(name, fn, *fn_args):
        results[name] = fn(*fn_args)

    if args.test in ("all", "urdf"):     run("urdf",     test_urdf)
    if args.test in ("all", "ik"):       run("ik",       test_ik)
    if args.test in ("all", "lagrange"): run("lagrange", test_lagrange)
    if args.test in ("all", "vision"):   run("vision",   test_vision)
    if args.test in ("all", "serial"):   run("serial",   test_serial, args.port)
    if args.test in ("all", "rl"):       run("rl",       test_rl_model)
    if args.test in ("all", "full"):     run("full",     test_full)

    print("\n" + "="*40)
    print("테스트 결과 요약")
    print("="*40)
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{len(results)} 통과")
