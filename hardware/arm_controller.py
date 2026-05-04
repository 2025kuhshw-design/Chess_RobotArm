"""
로봇팔 시리얼 통신 컨트롤러
실제 아두이노 연결 모드 + 시뮬레이션 모드 지원
"""

import time
import math
import sys
import os

# ─────────────────────────────────────────
# 하이퍼파라미터 / 상수
# ─────────────────────────────────────────
DEFAULT_PORT     = "COM3"      # Windows 기본값 (Linux: /dev/ttyUSB0)
DEFAULT_BAUDRATE = 9600
INIT_WAIT_SEC    = 2.0         # 아두이노 초기화 대기
CMD_TIMEOUT_SEC  = 3.0         # OK 응답 타임아웃
SUCTION_ON_WAIT  = 0.5         # 흡착 후 대기 (초)
SUCTION_OFF_WAIT = 0.3         # 해제 후 대기 (초)
LIFT_HEIGHT      = 0.10        # 안전 높이 (m)

# 관절 한계 (서보 각도)
SERVO_MIN = 0
SERVO_MAX = 180

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.ik_solver import inverse_kinematics, chess_square_to_xyz, safe_approach_xyz


class RealArm:
    def __init__(self, port: str = DEFAULT_PORT,
                 baudrate: int = DEFAULT_BAUDRATE,
                 sim: bool = False):
        """
        sim=True: 아두이노 없이 print로 시뮬레이션
        sim=False: 실제 시리얼 통신
        """
        self.sim  = sim
        self.port = port
        self._ser = None

        if not sim:
            self._connect(port, baudrate)
        else:
            print(f"[ArmController] 시뮬레이션 모드 (포트 미사용)")

        self.home()

    # ─────────────────────────────────────────
    # 시리얼 연결
    # ─────────────────────────────────────────
    def _connect(self, port: str, baudrate: int):
        try:
            import serial
        except ImportError:
            raise ImportError("pip install pyserial 을 먼저 실행하세요.")

        self._ser = serial.Serial(port, baudrate, timeout=CMD_TIMEOUT_SEC)
        time.sleep(INIT_WAIT_SEC)   # 아두이노 리셋 대기
        # 초기화 완료 메시지 소비
        while self._ser.in_waiting:
            self._ser.readline()
        print(f"[ArmController] 시리얼 연결: {port} @ {baudrate}baud")

    # ─────────────────────────────────────────
    # 라디안 → 서보 각도 변환
    # ─────────────────────────────────────────
    @staticmethod
    def _rad_to_servo(q1: float, q2: float, q3: float) -> tuple:
        s1 = int(90 + math.degrees(q1))    # 베이스
        s2 = int(90 - math.degrees(q2))    # 어깨 (부호 반전)
        s3 = int(90 + math.degrees(q3))    # 팔꿈치

        s1 = max(SERVO_MIN, min(SERVO_MAX, s1))
        s2 = max(SERVO_MIN, min(SERVO_MAX, s2))
        s3 = max(SERVO_MIN, min(SERVO_MAX, s3))

        return s1, s2, s3

    # ─────────────────────────────────────────
    # 단일 명령 전송
    # ─────────────────────────────────────────
    def _send_cmd(self, s1: int, s2: int, s3: int, suction: bool):
        cmd = f"A{s1},{s2},{s3},{1 if suction else 0}\n"
        if self.sim:
            print(f"  [SIM] {cmd.strip()}")
            return

        self._ser.write(cmd.encode())
        deadline = time.time() + CMD_TIMEOUT_SEC
        while time.time() < deadline:
            if self._ser.in_waiting:
                resp = self._ser.readline().decode().strip()
                if resp == "OK":
                    return
                elif resp == "ERR":
                    raise RuntimeError(f"아두이노 ERR 응답: {cmd.strip()}")
        raise TimeoutError(f"아두이노 응답 타임아웃: {cmd.strip()}")

    # ─────────────────────────────────────────
    # 메서드: move (관절각 → 실행)
    # ─────────────────────────────────────────
    def move(self, q1: float, q2: float, q3: float, suction: bool = False):
        """관절각(라디안) → 서보 각도 변환 후 전송."""
        s1, s2, s3 = self._rad_to_servo(q1, q2, q3)
        self._send_cmd(s1, s2, s3, suction)

    # ─────────────────────────────────────────
    # 메서드: execute_move (체스 이동 1회 완전 실행)
    # ─────────────────────────────────────────
    def execute_move(self, from_sq: tuple, to_sq: tuple,
                     is_capture: bool = False,
                     rl_correction=None):
        """
        from_sq, to_sq = (col, row)
        rl_correction = (delta_q1, delta_q2, delta_q3) or None
        """
        def _move_to(col, row, lift=False):
            if lift:
                x, y, z = safe_approach_xyz(col, row, LIFT_HEIGHT)
            else:
                x, y, z = chess_square_to_xyz(col, row)
            q1, q2, q3 = inverse_kinematics(x, y, z)
            if rl_correction is not None:
                q1 += rl_correction[0]
                q2 += rl_correction[1]
                q3 += rl_correction[2]
            self.move(q1, q2, q3)

        fc, fr = from_sq
        tc, tr = to_sq

        print(f"  [실행] {from_sq} → {to_sq}  capture={is_capture}")

        # 기물 잡기: 도착 칸 기물 먼저 제거
        if is_capture:
            _move_to(tc, tr, lift=True)
            _move_to(tc, tr, lift=False)
            self.move(*inverse_kinematics(*chess_square_to_xyz(tc, tr)), suction=True)
            time.sleep(SUCTION_ON_WAIT)
            _move_to(tc, tr, lift=True)
            # 잡은 기물은 board 밖 임시 위치에 내려놓음 (col=8 영역)
            x_out = chess_square_to_xyz(tc, tr)[0] + 0.05
            y_out = chess_square_to_xyz(tc, tr)[1]
            z_out = chess_square_to_xyz(tc, tr)[2]
            q_out = inverse_kinematics(x_out, y_out, z_out + LIFT_HEIGHT)
            self.move(*q_out, suction=True)
            q_down = inverse_kinematics(x_out, y_out, z_out)
            self.move(*q_down, suction=True)
            time.sleep(SUCTION_OFF_WAIT)
            self.move(*q_down, suction=False)
            time.sleep(SUCTION_OFF_WAIT)
            self.move(*q_out, suction=False)

        # 1) 출발 칸 위 안전 높이
        _move_to(fc, fr, lift=True)
        # 2) 출발 칸 하강
        _move_to(fc, fr, lift=False)
        # 3) 흡착기 ON
        q1, q2, q3 = inverse_kinematics(*chess_square_to_xyz(fc, fr))
        self.move(q1, q2, q3, suction=True)
        time.sleep(SUCTION_ON_WAIT)
        # 4) 안전 높이로 상승
        _move_to(fc, fr, lift=True)
        # 5) 도착 칸 위 안전 높이
        _move_to(tc, tr, lift=True)
        # 6) 도착 칸 하강
        _move_to(tc, tr, lift=False)
        # 7) 흡착기 OFF
        q1, q2, q3 = inverse_kinematics(*chess_square_to_xyz(tc, tr))
        self.move(q1, q2, q3, suction=False)
        time.sleep(SUCTION_OFF_WAIT)
        # 8) 안전 높이로 상승
        _move_to(tc, tr, lift=True)
        # 9) 홈 복귀
        self.home()

    # ─────────────────────────────────────────
    # 메서드: home
    # ─────────────────────────────────────────
    def home(self):
        """모든 관절 중립(0 라디안 = 서보 90도)으로 복귀."""
        if self.sim:
            print("  [SIM] A90,90,90,0  (홈)")
            return
        self._send_cmd(90, 90, 90, False)

    def close(self):
        if self._ser is not None:
            self._ser.close()


# ─────────────────────────────────────────
# 단독 실행: 시뮬레이션 모드 동작 확인
# ─────────────────────────────────────────
if __name__ == "__main__":
    arm = RealArm(sim=True)

    print("\n=== 체스 이동 시뮬레이션: e2 → e4 ===")
    # e2 = col=4, row=1 / e4 = col=4, row=3
    arm.execute_move((4, 1), (4, 3), is_capture=False)

    print("\n=== 기물 잡기 시뮬레이션: d4 → e5 ===")
    arm.execute_move((3, 3), (4, 4), is_capture=True)

    arm.close()
    print("\n✅ arm_controller 시뮬레이션 완료")
