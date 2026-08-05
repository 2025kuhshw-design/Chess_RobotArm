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

# 정밀 동작(하강·흡착) 직전에 팔의 잔류 진동이 잦아들기를 기다리는 시간 (초).
# 팔 구조물의 고유진동수가 약 3~4Hz로 측정돼, 이동을 멈춰도 몇 번 더 흔들린다.
# 흔들리는 상태에서 하강하면 흡착컵이 기물 중심을 벗어난다.
# 0.4초면 3.5Hz 진동이 1~2주기 감쇠한다. 흔들림이 심하면 늘릴 것.
SETTLE_WAIT      = 0.4
EXPECTED_FW      = "3"         # servo_control.ino 의 FW_VERSION 과 일치해야 함

# 기물을 집으러 하강할 때, 기물 윗면(PIECE_Z)보다 얼마나 더 내려갈지 (m).
# 흡착컵은 고무라 살짝 눌러야 밀착돼 진공이 걸린다. 또 어깨높이·링크길이
# 실측 오차도 여기서 함께 흡수한다.
#   값을 키우면 더 깊이 내려감. 너무 크면 기물을 밀거나 서보에 무리.
#   test_square에서 'zoff 8' 처럼 mm 단위로 실시간 조정해 값을 찾을 것.
TOUCH_PRESS      = 0.004       # 4mm 더 눌러 내려감
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
# 펌웨어(servo_control.ino)의 ANGLE_HARD_MAX 와 맞춘 값. 일반 취미용 서보의
# 표시 규격은 180°지만, 실제로는 200° 근처까지 도는 개체가 많다.
# ⚠️ 180°를 넘는 구간은 개체마다 다르다. 서보가 기계적 스토퍼에 막히면
#    멈춘 채로 계속 힘을 주다가 기어가 갈리거나 탄다(실제로 s2에서 겪음).
#    올리기 전에 반드시 SERVO_MAX_TEST_NOTE 의 절차대로 관절별 실제 한계를
#    확인하고, 그 값을 SERVO_SAFE_MAX 에 적어 넣을 것.
SERVO_MIN = 0
SERVO_MAX = 200

SERVO_MAX_TEST_NOTE = """
관절별 실제 상한 확인 절차 (한 관절씩, 기물 없이):
    python hardware/test_square.py --port COM5
    square> set <현재 각도 3개>
    square> a ... 180 ...      ← 목표 관절만 180으로
    square> a ... 185 ...      ← 5°씩만 올린다
    square> a ... 190 ...
  각 단계에서 팔이 '실제로 더 움직였는지' 눈으로 확인한다.
  더 안 움직이는데 명령만 올라가면 → 거기가 스토퍼. 즉시 되돌리고
  그 직전 값을 SERVO_SAFE_MAX 에 적는다.
  '끼익'·'드르륵' 소리가 나면 이미 기어가 갈리는 중 → 즉시 Ctrl+C, 전원 차단.
"""

# ─────────────────────────────────────────
# 안전: 관절별 소프트 각도 제한 (도)
# ─────────────────────────────────────────
# 각 서보가 갈 수 있는 안전 범위. 팔이 테이블·구조물에 처박히거나 스톨나는
# 각도를 아예 막는다. 기본은 전 범위(0~180)이니, 실물에서 부딪히는 각도를
# 확인한 뒤 좁혀 넣을 것. 예: 어깨(s2)가 40° 밑에서 테이블에 닿으면
#   SERVO_SAFE_MIN = [0, 40, 0]  처럼 설정.
# 제한에 걸리면 그 각도로 잘리고 경고를 출력한다(모르고 지나치지 않게).
#
# ⚠️ 여기가 실제로 적용되는 상한이다. SERVO_MAX(200)는 "펌웨어가 받아주는
#    최대"일 뿐이고, 아래 값이 그보다 낮으면 아래 값으로 잘린다.
#    180 초과 구간을 쓰려면 SERVO_MAX_TEST_NOTE 절차로 확인한 뒤 올릴 것.
SERVO_SAFE_MIN = [0,   0,   0]     # [베이스, 어깨, 팔꿈치] 최소 허용각
SERVO_SAFE_MAX = [180, 180, 180]   # [베이스, 어깨, 팔꿈치] 최대 허용각 (확인 후 상향)

# ─────────────────────────────────────────
# 안전: 부드러운 램프 이동 (충격/스톨 방지)
# ─────────────────────────────────────────
# 목표 각도로 즉시 슬램하지 않고 RAMP_STEP_DEG 씩 나눠 이동한다.
# 값을 작게/대기를 길게 하면 더 부드럽고 안전(대신 느려짐).
# 기본값은 안전 우선(약 25~30°/초). 기어 손상 이력이 있어 보수적으로 잡았다.
RAMP_STEP_DEG = 2       # 한 스텝당 최대 각도 변화 (도)
RAMP_DELAY    = 0.05    # 스텝 간 대기 (s)


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
SERVO1_HOME = 85;  SERVO1_DIR = +1   # 베이스 (s1 커질수록 로봇 왼쪽=+y)
                                     # 2차 교체 서보: 85°가 정면 (실측, 이전 서보는 38°)
SERVO2_HOME = -10; SERVO2_DIR = +1   # 어깨 (수직=80, 작아질수록 앞으로)
SERVO3_HOME = 50;  SERVO3_DIR = -1   # 팔꿈치 (일직선=50, 커질수록 앞으로)

# 게임 시작/휴식 자세 = Z자 접힘 (서보 각도 직접 지정: s1=베이스, s2=어깨, s3=팔꿈치)
# 수를 두는 사이 팔이 이 Z자 자세로 접혀 카메라 시야를 안 가리고 토크 부하도 줄인다.
# 실측 확정값 — 이 자세가 팔의 물리적 휴식 자세이기도 하다.
# ⚠️ 전원을 켜고 프로그램을 시작할 때 팔이 실제로 이 자세에 있어야 안전하다.
#    (다른 자세에 있으면 첫 명령에서 그 차이만큼 튄다 → start_pose로 알려줄 것)
PARK_POSE = (85, 140, 180)   # (베이스 정면, 어깨, 팔꿈치) — Z자

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


def touch_xyz(col: int, row: int) -> tuple:
    """기물을 집거나 놓을 때 하강할 좌표.
    기물 윗면보다 TOUCH_PRESS 만큼 더 내려가 흡착컵이 눌리도록 한다."""
    x, y, z = chess_square_to_xyz(col, row)
    return (x, y, z - TOUCH_PRESS)


def interactive_startup(arm, max_angle: int = 180) -> bool:
    """전원 인가 직후의 안전한 기동 절차.

    아두이노는 부팅 시 서보를 잡지 않으므로 팔이 중력으로 처져 있다.
    이 상태에서 아무 위치나 명령하면 서보가 최대 속도로 튄다(기어 손상 위험).
    그래서 순서를 이렇게 잡는다:

      1) engage <s1> <s2> <s3>  — 지금 팔이 있는 각도를 알려준다.
         그 값을 그대로 보내므로 서보는 '움직이지 않고 힘만' 들어온다.
      2) 조그 명령으로 PARK 자세까지 천천히 올린다 (램프 적용).
      3) start — 세팅 완료. 게임/테스트로 넘어간다.

    반환: True=정상 완료, False=사용자가 중단.
    """
    P = PARK_POSE
    print("\n" + "=" * 56)
    print(" 기동 절차 (팔이 처진 상태에서 안전하게 시작)")
    print("=" * 56)
    print(" ⭐ 가장 쉬운 방법:")
    print(f"    팔을 손으로 Z자 자세로 잡은 채 → 'engage' 입력")
    print(f"    (= engage {P[0]} {P[1]} {P[2]} 와 동일. 실제와 맞으니 안 움직임)")
    print()
    print("  engage             팔이 지금 Z자(PARK)에 있다고 선언")
    print("  engage <a> <b> <c> 다른 각도에 있으면 그 값으로 선언")
    print("  <a b c> / s2 120   천천히 이동 (램프 적용)")
    print("  park               PARK 자세로 천천히 이동")
    print("  start              세팅 완료 → 진행")
    print("  q                  중단")
    print(f"  ※ PARK_POSE = A{P[0]},{P[1]},{P[2]}")
    print("  ⚠️ engage 값이 실제 위치와 다르면 그 차이만큼 서보가 튑니다.")
    print("     크게 움직였다면 값이 틀린 것 → 멈춘 그 위치를 다시 engage.")
    print("  🛑 긁는 소리 나면 즉시 Ctrl+C → 전원 차단\n")

    engaged = False
    while True:
        try:
            line = input("startup> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if not line:
            continue
        p = line.split()

        try:
            if p[0] == "q":
                return False

            if p[0] == "start":
                if not engaged:
                    print("  아직 engage 하지 않았습니다. "
                          "'engage <s1> <s2> <s3>' 먼저 실행하세요.")
                    continue
                print(f"  세팅 완료. 현재 위치 A{arm._cur[0]},{arm._cur[1]},{arm._cur[2]}")
                return True

            if p[0] == "engage" and len(p) in (1, 4):
                # 인자 없으면 PARK(Z자)에 있다고 간주 — 손으로 잡고 쓰는 경우
                pose = (list(PARK_POSE) if len(p) == 1
                        else [max(0, min(max_angle, int(v))) for v in p[1:]])
                arm._cur = list(pose)
                # 같은 값을 한 번만 보내 서보에 힘만 들어오게 한다
                arm._send_cmd(pose[0], pose[1], pose[2], False)
                engaged = True
                print(f"  A{pose[0]},{pose[1]},{pose[2]} 로 고정 (움직임 없어야 정상)")
                print("  → 크게 움직였다면 값이 실제와 다른 것. "
                      "멈춘 그 위치를 다시 engage 하세요.")
                continue

            if not engaged:
                print("  먼저 'engage <s1> <s2> <s3>' 로 현재 각도를 알려주세요.")
                continue

            if p[0] == "park":
                print(f"  PARK A{PARK_POSE[0]},{PARK_POSE[1]},{PARK_POSE[2]} 로 천천히 이동")
                arm._ramp_send(*PARK_POSE, False)
            elif len(p) == 3 and all(v.lstrip("-").isdigit() for v in p):
                t = [max(0, min(max_angle, int(v))) for v in p]
                print(f"  A{t[0]},{t[1]},{t[2]} 로 천천히 이동")
                arm._ramp_send(t[0], t[1], t[2], False)
            elif p[0] in ("s1", "s2", "s3") and len(p) == 2:
                t = list(arm._cur)
                t[int(p[0][1]) - 1] = max(0, min(max_angle, int(p[1])))
                print(f"  A{t[0]},{t[1]},{t[2]} 로 천천히 이동")
                arm._ramp_send(t[0], t[1], t[2], False)
            else:
                print("  명령: engage a b c | <a b c> | s2 120 | park | start | q")
        except ValueError:
            print("  입력 오류. 다시.")
        except KeyboardInterrupt:
            print("\n  [중단] 현재 위치에서 멈춤. 전원 확인하세요.")


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
                 start_pose: tuple = None,
                 ramp_step: int = None,
                 ramp_delay: float = None,
                 auto_home: bool = True):
        """
        sim=True: 아두이노 없이 print로 시뮬레이션
        sim=False: 실제 시리얼 통신
        start_pose: 시작 시 팔의 실제 서보 각도 (s1,s2,s3).
            아두이노는 부팅 시 서보 출력을 켜지 않으므로(무부하) 실제 위치를
            알 수 없다. 전원을 끈 채 팔을 손으로 옮겼다면 반드시 지정할 것.
            없으면 PARK_POSE에 있다고 가정한다.
        ramp_step/ramp_delay: 이동 속도 (기본 RAMP_STEP_DEG/RAMP_DELAY).
            작은 step + 큰 delay = 느리고 안전.
        auto_home: True면 초기화 직후 PARK_POSE로 이동한다.
            ⚠️ 아두이노 부팅 직후 서보는 무부하라 팔이 중력으로 처져 있다.
            그 상태에서 첫 명령이 나가면 서보가 '자기 최대 속도'로 되돌아가며,
            이 첫 동작만은 램프로 늦출 수 없다(출발 위치를 모르므로).
            도구에서 사용자가 직접 시점을 고르게 하려면 False로 둘 것.
        """
        self.sim  = sim
        self.port = port
        self._ser = None
        self.ramp_step  = RAMP_STEP_DEG if ramp_step  is None else max(1, ramp_step)
        self.ramp_delay = RAMP_DELAY    if ramp_delay is None else max(0.0, ramp_delay)
        # 램프 이동의 출발점으로 쓰이는 현재 위치 추정값.
        self._cur = list(start_pose) if start_pose else list(PARK_POSE)
        self._captured_count = 0    # 캡처 구역에 쌓은 기물 수

        if not sim:
            self._connect(port, baudrate)
        else:
            print(f"[ArmController] 시뮬레이션 모드 (포트 미사용)")

        if auto_home:
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

        # 초기화 메시지에서 펌웨어 버전 확인 (.ino 재업로드 누락 감지)
        banner = ""
        while self._ser.in_waiting:
            banner += self._ser.readline().decode(errors="replace").strip() + " "
        print(f"[ArmController] 시리얼 연결: {port} @ {baudrate}baud")

        fw = None
        if "fw=" in banner:
            fw = banner.split("fw=")[1].split()[0]
        if fw == EXPECTED_FW:
            print(f"[ArmController] 펌웨어 fw={fw} ✓")
        else:
            print("=" * 60)
            print(f"⚠️ 아두이노 펌웨어가 최신이 아닙니다 "
                  f"(감지: {fw or '버전 없음(구버전)'} / 필요: {EXPECTED_FW})")
            print("   Arduino IDE에서 hardware/arduino/servo_control.ino 를")
            print("   반드시 다시 업로드하세요.")
            print("   구버전은 부팅 시 서보를 90,90,90으로 '전속력' 이동시켜")
            print("   기어가 손상될 수 있습니다.")
            print("=" * 60)

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

        # 실제 상한은 SERVO_MAX(펌웨어 한계)와 SERVO_SAFE_MAX(확인된 기계 한계)
        # 중 낮은 쪽이다. 초과분을 조용히 잘라내면 팔이 엉뚱한 데 서 있는데도
        # '도착했다'고 여기게 되므로, 자르지 않고 실패시킨다.
        out = []
        for j, (name, v) in enumerate((("s1", s1), ("s2", s2), ("s3", s3))):
            lo = max(SERVO_MIN, SERVO_SAFE_MIN[j])
            hi = min(SERVO_MAX, SERVO_SAFE_MAX[j])
            if not (lo <= v <= hi):
                raise ValueError(
                    f"서보 가동범위 밖: {name}={v} (허용 {lo}~{hi}). "
                    f"이 칸은 현재 배치에서 팔이 닿지 않습니다."
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
                    self._cur[j] = min(target[j], self._cur[j] + self.ramp_step)
                    moved = True
                elif self._cur[j] > target[j]:
                    self._cur[j] = max(target[j], self._cur[j] - self.ramp_step)
                    moved = True
            self._send_cmd(self._cur[0], self._cur[1], self._cur[2], suction)
            if not moved:
                break
            time.sleep(self.ramp_delay)

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
                     rl_correction=None,
                     confirm: bool = False,
                     aligner=None):
        """
        from_sq, to_sq = (col, row)
        rl_correction = (delta_q1, delta_q2, delta_q3) or None
        confirm=True 면 칸 위 안전높이에서 멈춰 사람 확인을 받은 뒤 하강한다.
            (사람이 눈으로 보고 판단)
        aligner=f(col, row, lift, suction) 를 주면 하강 직전마다 호출해
            카메라 폐루프 보정을 한다. None이면 보정 없이 진행.
        """
        def _align(col, row, suction=False):
            if aligner is None:
                return
            try:
                aligner(col, row, _safe_lift(col, row), suction)
            except Exception as e:
                print(f"    [정렬] 실패 — 보정 없이 진행 ({e})")
        def _ask(msg):
            if not confirm:
                return True
            ans = input(f"    {msg} [Enter=진행] [x=이 수 건너뛰기] > ").strip().lower()
            return ans not in ("x", "q", "n")
        # ⚠️ suction 인자를 반드시 넘길 것. 기본값(False)으로 두면 기물을
        #    든 채 이동하는 구간에서 흡착이 풀려 기물을 떨어뜨린다.
        def _move_to(col, row, lift=False, suction=False):
            if lift:
                x, y, z = safe_approach_xyz(col, row, _safe_lift(col, row))
            else:
                x, y, z = touch_xyz(col, row)   # 흡착컵이 눌리도록 조금 더 하강
            q1, q2, q3 = inverse_kinematics(x, y, z)
            if rl_correction is not None:
                q1 += rl_correction[0]
                q2 += rl_correction[1]
                q3 += rl_correction[2]
            self.move(q1, q2, q3, suction=suction)

        fc, fr = from_sq
        tc, tr = to_sq

        print(f"  [실행] {from_sq} → {to_sq}  capture={is_capture}")

        # 기물 잡기: 도착 칸 기물 먼저 제거
        if is_capture:
            _move_to(tc, tr, lift=True)
            time.sleep(SETTLE_WAIT)
            _align(tc, tr)                   # 카메라 보정 후 하강
            _move_to(tc, tr, lift=False)
            time.sleep(SETTLE_WAIT)
            self.move(*inverse_kinematics(*touch_xyz(tc, tr)), suction=True)
            time.sleep(SUCTION_ON_WAIT)
            _move_to(tc, tr, lift=True, suction=True)     # 든 채로 상승
            # 잡은 기물은 보드 밖 캡처 구역에 내려놓음 (항상 도달 가능한 위치)
            x_out, y_out, z_out = capture_slot_xyz(self._captured_count)
            self._captured_count += 1
            lift_out = _safe_lift_xy(x_out, y_out)
            q_out = inverse_kinematics(x_out, y_out, z_out + lift_out)
            self.move(*q_out, suction=True)               # 든 채로 이동
            q_down = inverse_kinematics(x_out, y_out, z_out)
            self.move(*q_down, suction=True)              # 든 채로 하강
            time.sleep(SUCTION_OFF_WAIT)
            self.move(*q_down, suction=False)             # 여기서 놓음
            time.sleep(SUCTION_OFF_WAIT)
            self.move(*q_out, suction=False)

        # 1) 출발 칸 위 안전 높이
        _move_to(fc, fr, lift=True)
        time.sleep(SETTLE_WAIT)          # 흔들림이 잦아든 뒤 하강
        _align(fc, fr)                   # 카메라 보정
        if not _ask("흡착컵이 집을 기물 바로 위인가요?"):
            print("    건너뜀 — 기물을 손으로 옮겨주세요"); self.home(); return
        # 2) 출발 칸 하강
        _move_to(fc, fr, lift=False)
        time.sleep(SETTLE_WAIT)          # 흔들림이 잦아든 뒤 흡착
        # 3) 흡착기 ON
        q1, q2, q3 = inverse_kinematics(*touch_xyz(fc, fr))
        self.move(q1, q2, q3, suction=True)
        time.sleep(SUCTION_ON_WAIT)
        # 4) 안전 높이로 상승 (기물 든 상태 유지)
        _move_to(fc, fr, lift=True, suction=True)
        # 5) 도착 칸 위 안전 높이 (기물 든 상태 유지)
        _move_to(tc, tr, lift=True, suction=True)
        time.sleep(SETTLE_WAIT)
        _align(tc, tr, suction=True)     # 기물 든 채로 보정 (흡착 유지)
        _ask("이 칸에 놓을까요?")   # 취소해도 어차피 놓아야 하므로 진행
        # 6) 도착 칸 하강 (기물 든 상태 유지)
        _move_to(tc, tr, lift=False, suction=True)
        time.sleep(SETTLE_WAIT)          # 흔들림이 잦아든 뒤 놓기
        # 7) 흡착기 OFF — 여기서 처음으로 놓는다
        q1, q2, q3 = inverse_kinematics(*touch_xyz(tc, tr))
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
