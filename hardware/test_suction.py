"""
흡착기(펌프/밸브) 단독 테스트.

펌프=ch4, 밸브=ch5 (servo_control.ino 기준). suction 비트만 토글한다.
명령은 항상 A90,90,90,{s} 형식이라 서보 채널(ch0~2)에도 90도가 나간다.
→ 흡착기만 안전하게 보려면 **서보 3개는 PCA에서 분리(뽑아)하고** 테스트 권장.
   (서보 부하를 빼서 전원 문제 격리 + 팔이 안 움직임)

실행: python hardware/test_suction.py --port COM5

명령:
  on            흡착 ON  (펌프 작동, 밸브 닫힘=진공 유지)
  off           흡착 OFF (펌프 정지, 밸브 열림=해제)
  pulse [초]    ON 후 지정 초 뒤 자동 OFF (기본 2초)
  q             종료(OFF 후)
"""

import argparse
import time
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=str, default="COM5")
    ap.add_argument("--baud", type=int, default=9600)
    args = ap.parse_args()

    try:
        import serial
    except ImportError:
        print("pip install pyserial 먼저 실행."); sys.exit(1)

    ser = serial.Serial(args.port, args.baud, timeout=3)
    time.sleep(2)               # 아두이노 리셋 대기
    ser.reset_input_buffer()    # READY 등 초기 메시지 제거
    print(f"[흡착 테스트] 연결됨 {args.port}")
    print("  ⚠️ 서보는 PCA에서 분리하고 테스트하는 걸 권장 (부하 격리+안전)")
    print("  명령: on / off / pulse [초] / q")
    print("  🛑 펌프 켤 때 전원 스위치 손에 두세요.\n")

    def send(suction):
        cmd = f"A90,90,90,{suction}\n"
        ser.write(cmd.encode())
        resp = ser.readline().decode().strip()
        print(f"    → {cmd.strip()}  응답 {resp!r}")
        return resp

    try:
        while True:
            line = input("suction> ").strip().lower()
            if not line:
                continue
            parts = line.split()

            if parts[0] == "q":
                break
            elif parts[0] == "on":
                print("  흡착 ON")
                send(1)
            elif parts[0] == "off":
                print("  흡착 OFF")
                send(0)
            elif parts[0] == "pulse":
                sec = 2.0
                if len(parts) == 2:
                    try: sec = float(parts[1])
                    except ValueError: pass
                print(f"  흡착 ON → {sec:.1f}초 후 자동 OFF")
                send(1)
                time.sleep(sec)
                send(0)
            else:
                print("  명령: on / off / pulse [초] / q")
    except (EOFError, KeyboardInterrupt):
        print()

    # 종료 시 반드시 OFF
    print("  종료 — 흡착 OFF")
    send(0)
    ser.close()
    print("[흡착 테스트] 종료.")


if __name__ == "__main__":
    main()
