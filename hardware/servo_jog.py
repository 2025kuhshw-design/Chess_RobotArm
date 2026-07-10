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
    args = parser.parse_args()

    try:
        import serial
    except ImportError:
        print("pip install pyserial 를 먼저 실행하세요.")
        sys.exit(1)

    ser = serial.Serial(args.port, args.baud, timeout=3)
    time.sleep(2)
    ser.reset_input_buffer()
    print(f"[jog] 연결됨: {args.port}")

    s = [90, 90, 90]
    suction = 0

    def send():
        cmd = f"A{s[0]},{s[1]},{s[2]},{suction}\n"
        ser.write(cmd.encode())
        resp = ser.readline().decode().strip()
        print(f"  보냄 {cmd.strip()}  → 응답 {resp}")

    print("사용법: '90 90 90' | 's1 120' | 'suction 1' | 'q'(종료)")
    send()

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
        try:
            if len(parts) == 3 and all(p.lstrip("-").isdigit() for p in parts):
                s = [max(0, min(180, int(p))) for p in parts]
            elif parts[0].startswith("s") and len(parts) == 2:
                idx = int(parts[0][1]) - 1
                s[idx] = max(0, min(180, int(parts[1])))
            elif parts[0] == "suction" and len(parts) == 2:
                suction = 1 if parts[1] == "1" else 0
            else:
                print("  형식: '90 90 90' | 's1 120' | 'suction 1' | 'q'")
                continue
        except (ValueError, IndexError):
            print("  입력 오류. 다시.")
            continue

        send()

    # 종료: 홈으로
    s = [90, 90, 90]
    suction = 0
    send()
    ser.close()
    print("[jog] 종료.")


if __name__ == "__main__":
    main()
