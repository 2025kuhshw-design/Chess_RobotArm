"""
서보 조그(jog) 도구 — 캘리브레이션용.
서보 각도 3개를 직접 입력해 팔을 움직여 보며 SERVO*_HOME 값을 찾는다.

실행: python hardware/servo_jog.py --port COM5

입력 예:
  90 90 90      → 세 서보를 각각 90,90,90
  s1 120        → 1번 서보만 120으로 (나머지 유지)
  suction 1     → 흡착 ON  /  suction 0 → OFF
  q             → 종료(홈 자세로)
"""

import argparse
import time
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="COM5")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--step", type=int, default=2,
                        help="한 스텝당 각도(도) — 작을수록 부드럽고 안전")
    parser.add_argument("--step-delay", type=float, default=0.02,
                        help="스텝 간 대기(s)")
    parser.add_argument("--max-angle", type=int, default=180,
                        help="허용 최대각. 180 초과가 필요할 때만 올릴 것 "
                             "(예: 200). ⚠️ 서보 하드 스톱 스톨 위험 — 조금씩!")
    parser.add_argument("--start", type=int, nargs=3, metavar=("S1", "S2", "S3"),
                        help="현재 팔의 대략적인 서보 각도. 전원을 끄고 팔을 손으로 "
                             "옮겼다면 반드시 지정할 것 (안 하면 90,90,90에서 "
                             "출발한다고 가정해 첫 이동이 튄다)")
    args = parser.parse_args()
    MAXA = max(180, min(200, args.max_angle))   # 펌웨어 상한(200)과 일치

    try:
        import serial
    except ImportError:
        print("pip install pyserial 를 먼저 실행하세요.")
        sys.exit(1)

    ser = serial.Serial(args.port, args.baud, timeout=3)
    time.sleep(2)
    ser.reset_input_buffer()
    print(f"[jog] 연결됨: {args.port}")

    # 현재 위치 추정값. 아두이노는 부팅 시 서보 출력을 켜지 않으므로(무부하),
    # 실제 팔이 어디 있는지 알 수 없다. --start 로 알려주지 않으면 90,90,90 가정.
    s = list(args.start) if args.start else [90, 90, 90]
    suction = 0

    def _send_now():
        cmd = f"A{s[0]},{s[1]},{s[2]},{suction}\n"
        ser.write(cmd.encode())
        return ser.readline().decode().strip()

    def ramp_to(target):
        """현재 s에서 target까지 --step 도씩 부드럽게 이동 (충격/스톨 방지).
        긁는 소리가 나면 즉시 Ctrl+C → 전원 차단."""
        target = [max(0, min(MAXA, int(v))) for v in target]
        while s != target:
            for j in range(3):
                if s[j] < target[j]:
                    s[j] = min(target[j], s[j] + args.step)
                elif s[j] > target[j]:
                    s[j] = max(target[j], s[j] - args.step)
            _send_now()
            time.sleep(args.step_delay)
        print(f"  도달 A{s[0]},{s[1]},{s[2]},{suction}")

    print("사용법: '90 90 90' | 's1 120' | 'set 100 80 120' | 'suction 1' | 'q'(종료)")
    print("  ※ 각도까지 조금씩 부드럽게 이동합니다. 긁는 소리 나면 즉시 Ctrl+C!")
    print(f"  현재 위치 가정: A{s[0]},{s[1]},{s[2]}  (실제와 다르면 'set'으로 교정)")
    print("  ⚠️ 시작 시 아무 명령도 보내지 않습니다(서보 무부하). 첫 이동 명령부터 제어 시작.")
    # 시작 시 전송하지 않는다 — 실제 위치를 모르는 채로 명령하면 그 차이만큼 슬램된다.

    while True:
        try:
            line = input("jog> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "q":
            break

        parts = line.split()
        target = list(s)
        try:
            if len(parts) == 3 and all(p.lstrip("-").isdigit() for p in parts):
                target = [int(p) for p in parts]
            elif parts[0] in ("s1", "s2", "s3") and len(parts) == 2:
                idx = int(parts[0][1]) - 1
                target[idx] = int(parts[1])
            elif parts[0] == "set" and len(parts) == 4:
                # 움직이지 않고 '현재 위치 가정'만 교정 (슬램 방지용)
                s[:] = [max(0, min(MAXA, int(p))) for p in parts[1:]]
                print(f"  현재 위치 가정을 A{s[0]},{s[1]},{s[2]} 로 교정 (전송 안 함)")
                continue
            elif parts[0] == "suction" and len(parts) == 2:
                suction = 1 if parts[1] == "1" else 0
                resp = _send_now()
                print(f"  흡착 {suction} 전송 A{s[0]},{s[1]},{s[2]},{suction} "
                      f"→ 응답 {resp!r}")
                continue
            else:
                print("  형식: '90 90 90' | 's1 120' | 'set 100 80 120' | "
                      "'suction 1' | 'q'")
                continue
        except (ValueError, IndexError):
            print("  입력 오류. 다시.")
            continue

        try:
            ramp_to(target)
        except KeyboardInterrupt:
            print("\n  [중단] 현재 위치에서 멈춤. 전원 확인하세요.")

    # 종료: 홈으로 되돌리지 않고 마지막 각도 그대로 유지한 채 종료.
    # (아두이노 타임아웃 제거되어 포트 닫아도 서보가 현재 위치를 붙잡음)
    ser.close()
    print(f"[jog] 종료. 현재 위치 유지: A{s[0]},{s[1]},{s[2]},{suction}")


if __name__ == "__main__":
    main()
