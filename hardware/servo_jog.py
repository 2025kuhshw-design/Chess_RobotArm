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

    s = [90, 90, 90]          # 현재(마지막으로 명령한) 위치
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

    print("사용법: '90 90 90' | 's1 120' | 'suction 1' | 'q'(종료)")
    print("  ※ 각도까지 조금씩 부드럽게 이동합니다. 긁는 소리 나면 즉시 Ctrl+C!")
    _send_now()

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
            elif parts[0] == "suction" and len(parts) == 2:
                suction = 1 if parts[1] == "1" else 0
                resp = _send_now()
                print(f"  흡착 {suction} 전송 A{s[0]},{s[1]},{s[2]},{suction} "
                      f"→ 응답 {resp!r}")
                continue
            else:
                print("  형식: '90 90 90' | 's1 120' | 'suction 1' | 'q'")
                continue
        except (ValueError, IndexError):
            print("  입력 오류. 다시.")
            continue

        try:
            ramp_to(target)
        except KeyboardInterrupt:
            print("\n  [중단] 현재 위치에서 멈춤. 전원 확인하세요.")

    # 종료: 부드럽게 홈으로
    ramp_to([90, 90, 90])
    ser.close()
    print("[jog] 종료.")


if __name__ == "__main__":
    main()
