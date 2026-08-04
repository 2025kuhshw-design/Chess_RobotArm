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
LIFT_HEIGHT      = 0.04        # 기본 안전 높이 (m) — 기물 7mm 기준 충분
LIFT_MIN         = 0.015       # 최소 리프트 (m) — 기물(7mm)보다 커야 함

# ─────────────────────────────────────────
# 잡은 기물을 내려놓는 구역 (보드 밖, 팔 도달 범위 안)
# ─────────────────────────────────────────
# 체스판은 x 0.110~0.343, y -0.060~0.173 을 차지한다. 이 구역은 보드 바깥
# (로봇 왼쪽 +y)이면서 베이스 가동범위 안이라 IK가 성공한다.
# 실제 배치에 맞춰 조정할 것 (보드나 구조물과 겹치지 않는지 확인).
CAPTURE_X0      = 0.080    # 첫 슬롯 x (m)
CAPTURE_Y0      = 0.230    # 첫 슬롯 y (m) — 보드 왼쪽(+y) 바깥
CAPTURE_SPACING = 0.030    # 슬롯 간격 (m)
CAPTURE_COLS    = 4        # 가로 슬롯 수
CAPTURE_ROWS    = 2        # 세로 슬롯 수

# 관절 한계 (서보 각도)
SERVO_MIN = 0
SERVO_MAX = 180

# ─────────────────────────────────────────
# 안전: 관절별 소프트 각도 제한 (도)
# ─────────────────────────────────────────
# 각 서보가 갈 수 있는 안전 범위. 팔이 테이블·구조물에 처박히거나 스톨나는
# 각도를 아예 막는다. 기본은 전 범위(0~180)이니, 실물에서 부딪히는 각도를
# 확인한 뒤 좁혀 넣을 것. 예: 어깨(s2)가 40° 밑에서 테이블에 닿으면
#   SERVO_SAFE_MIN = [0, 40, 0]  처럼 설정.
# 제한에 걸리면 그 각도로 잘리고 경고를 출력한다(모르고 지나치지 않게).
SERVO_SAFE_MIN = [0,   0,   0]     # [베이스, 어깨, 팔꿈치] 최소 허용각
SERVO_SAFE_MAX = [180, 180, 180]   # [베이스, 어깨, 팔꿈치] 최대 허용각

# ─────────────────────────────────────────
# 안전: 부드러운 램프 이동 (충격/스톨 방지)
# ─────────────────────────────────────────
# 목표 각도로 즉시 슬램하지 않고 RAMP_STEP_DEG 씩 나눠 이동한다.
# 값을 작게/대기를 길게 하면 더 부드럽고 안전(대신 느려짐).
RAMP_STEP_DEG = 4       # 한 스텝당 최대 각도 변화 (도)
RAMP_DELAY    = 0.015   # 스텝 간 대기 (s)


def _clamp_safe(s1: int, s2: int, s3: int, warn: bool = True) -> tuple:
    """관절별 소프트 제한으로 클램프. 잘리면 경고 출력."""
    raw = [int(s1), int(s2), int(s3)]
    out = [max(SERVO_SAFE_MIN[j], min(SERVO_SAFE_MAX[j], raw[j])) for j in range(3)]
    if warn and out != raw:
        print(f"  [안전제한] 서보각 {raw} → {out} (소프트 제한에 걸림)")
    return out[0], out[1], out[2]

# ─────────────────────────────────────────
# 서보 캘리브레이션 (혼 재장착 없이 소프트웨어로 맞춤)
# ─────────────────────────────────────────
# SERVOx_HOME: 해당 관절이 IK 기준자세(q=0)일 때 보낼 서보 각도.
#   - 베이스  q1=0: 팔이 체스판 정면(중앙)을 똑바로 가리킴
#   - 어깨    q2=0: 상완(윗팔)이 수평(테이블과 평행)으로 앞을 향함
#   - 팔꿈치  q3=0: 전완(아랫팔)이 상완과 일직선(완전히 편 상태)
# SERVOx_DIR: +1 또는 -1. 서보 각도를 키웠을 때 관절이 IK의 +방향으로 돌면 +1.
#   (실물에서 반대로 움직이면 부호를 뒤집는다)
SERVO1_HOME = 38;  SERVO1_DIR = +1   # 베이스 (s1 커질수록 로봇 왼쪽=+y)
SERVO2_HOME = -10; SERVO2_DIR = +1   # 어깨 (수직=80, 작아질수록 앞으로)
SERVO3_HOME = 50;  SERVO3_DIR = -1   # 팔꿈치 (일직선=50, 커질수록 앞으로)

# 게임 시작/휴식 자세 = Z자 접힘 (서보 각도 직접 지정: s1=베이스, s2=어깨, s3=팔꿈치)
# 수를 두는 사이 팔이 이 Z자 자세로 접혀 카메라 시야를 안 가리고 토크 부하도 줄인다.
# ⚠️ 아래 값은 시작 추정치. 시리얼 모니터로 A{s1},{s2},{s3},0 을 넣어보며
#    실제로 Z자로 접히는 각도를 찾아 이 세 값을 수정할 것.
PARK_POSE = (90, 45, 135)   # (베이스 정면, 어깨 올림, 팔꿈치 접음)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.ik_solver import (inverse_kinematics, chess_square_to_xyz,
                              safe_approach_xyz,
                              L1_DEFAULT, L2_DEFAULT, L3_DEFAULT, PIECE_Z)


def _safe_lift_xy(x: float, y: float) -> float:
    """(x,y)의 수평 거리를 고려해 IK가 성공하는 최대 리프트를 반환.
    LIFT_HEIGHT와 실제 도달 한계 중 작은 값을 쓰고, LIFT_MIN 이상을 보장."""
    r = math.sqrt(x ** 2 + y ** 2)
    arm_reach = L1_DEFAULT + L2_DEFAULT          # 0.295 m
    # 수직 여유: sqrt(reach^2 - r^2) - L3 - PIECE_Z - 5mm 마진
    max_z_safe = math.sqrt(max(arm_reach ** 2 - r ** 2, 0)) - L3_DEFAULT - PIECE_Z - 0.005
    return max(LIFT_MIN, min(LIFT_HEIGHT, max_z_safe))


def _safe_lift(col: int, row: int) -> float:
    """체스 칸 기준 안전 리프트."""
    x, y, _ = chess_square_to_xyz(col, row)
    return _safe_lift_xy(x, y)


def capture_slot_xyz(index: int) -> tuple:
    """잡은 기물을 내려놓을 위치 (보드 밖, 팔이 확실히 닿는 구역).

    이전 방식(잡은 칸에서 x+5cm)은 h1 같은 먼 칸에서 반경을 넘어 IK가
    실패했다(r>0.295). 베이스에 가까운 고정 구역에 순서대로 쌓는다.
    index가 슬롯 수를 넘으면 순환한다(기물이 겹치지만 동작은 유지)."""
    col = index % CAPTURE_COLS
    row = (index // CAPTURE_COLS) % CAPTURE_ROWS
    x = CAPTURE_X0 + col * CAPTURE_SPACING
    y = CAPTURE_Y0 - row * CAPTURE_SPACING
    return (x, y, PIECE_Z)


class RealArm:
    def __init__(self, port: str = DEFAULT_PORT,
                 baudrate: int = DEFAULT_BAUDRATE,
                 sim: bool = False,
                 start_pose: tuple = None):
        """
        sim=True: 아두이노 없이 print로 시뮬레이션
        sim=False: 실제 시리얼 통신
        start_pose: 시작 시 팔의 실제 서보 각도 (s1,s2,s3).
            아두이노는 부팅 시 서보 출력을 켜지 않으므로(무부하) 실제 위치를
            알 수 없다. 전원을 끈 채 팔을 손으로 옮겼다면 반드시 지정할 것.
            없으면 PARK_POSE에 있다고 가정한다.
        """
        self.sim  = sim
        self.port = port
        self._ser = None
        # 램프 이동의 출발점으로 쓰이는 현재 위치 추정값.
        self._cur = list(start_pose) if start_pose else list(PARK_POSE)
        self._captured_count = 0    # 캡처 구역에 쌓은 기물 수

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
        """관절각(라디안) → 서보 각도. 서보 가동범위를 벗어나면 ValueError.

        ⚠️ 예전에는 범위를 벗어나도 조용히 클램프해서, 팔이 아무 경고 없이
        엉뚱한 위치로 갔다. 이제는 명시적으로 실패시켜 호출부가 알 수 있게 한다.
        """
        s1 = int(round(SERVO1_HOME + SERVO1_DIR * math.degrees(q1)))   # 베이스
        s2 = int(round(SERVO2_HOME + SERVO2_DIR * math.degrees(q2)))   # 어깨
        s3 = int(round(SERVO3_HOME + SERVO3_DIR * math.degrees(q3)))   # 팔꿈치

        out = []
        for name, v in (("s1", s1), ("s2", s2), ("s3", s3)):
            if not (SERVO_MIN <= v <= SERVO_MAX):
                raise ValueError(
                    f"서보 가동범위 밖: {name}={v} "
                    f"(허용 {SERVO_MIN}~{SERVO_MAX}). 이 칸은 현재 배치에서 "
                    f"팔이 닿지 않습니다."
                )
            out.append(v)
        return tuple(out)

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
    # 램프 전송: 현재 위치 → 목표까지 조금씩 이동 (충격/스톨 방지)
    # ─────────────────────────────────────────
    def _ramp_send(self, s1: int, s2: int, s3: int, suction: bool):
        """소프트 제한 클램프 후, RAMP_STEP_DEG씩 나눠 목표까지 부드럽게 이동."""
        t1, t2, t3 = _clamp_safe(s1, s2, s3)
        target = [t1, t2, t3]

        if self.sim:
            self._cur = list(target)
            self._send_cmd(target[0], target[1], target[2], suction)
            return

        while True:
            moved = False
            for j in range(3):
                if self._cur[j] < target[j]:
                    self._cur[j] = min(target[j], self._cur[j] + RAMP_STEP_DEG)
                    moved = True
                elif self._cur[j] > target[j]:
                    self._cur[j] = max(target[j], self._cur[j] - RAMP_STEP_DEG)
                    moved = True
            self._send_cmd(self._cur[0], self._cur[1], self._cur[2], suction)
            if not moved:
                break
            time.sleep(RAMP_DELAY)

    # ─────────────────────────────────────────
    # 메서드: move (관절각 → 실행)
    # ─────────────────────────────────────────
    def move(self, q1: float, q2: float, q3: float, suction: bool = False):
        """관절각(라디안) → 서보 각도 변환 후 램프 이동으로 전송."""
        s1, s2, s3 = self._rad_to_servo(q1, q2, q3)
        self._ramp_send(s1, s2, s3, suction)

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
                x, y, z = safe_approach_xyz(col, row, _safe_lift(col, row))
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
            # 잡은 기물은 보드 밖 캡처 구역에 내려놓음 (항상 도달 가능한 위치)
            x_out, y_out, z_out = capture_slot_xyz(self._captured_count)
            self._captured_count += 1
            lift_out = _safe_lift_xy(x_out, y_out)
            q_out = inverse_kinematics(x_out, y_out, z_out + lift_out)
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
        """게임 시작/휴식 자세(Z자 접힘)로 램프 이동 복귀."""
        s1, s2, s3 = PARK_POSE
        self._ramp_send(s1, s2, s3, False)

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
