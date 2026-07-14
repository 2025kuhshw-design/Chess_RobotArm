"""
하드웨어 격리 진단 도구.
부품을 '하나씩만' 작동시켜, 어디서 전원이 죽는지(과전류 차단) 좁혀낸다.

전제: 아두이노에 servo_control.ino 업로드됨, 시리얼 모니터 닫힘.
주의: 소프트웨어는 전류/전압을 직접 못 잰다. 각 테스트에서 '눈으로' 관찰
      (서보가 힘있게 버티나? 펌프가 도나? 갑자기 다 힘빠지나?)해서 답한다.

실행: python hardware/diagnose.py --port COM5
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
        print("pip install pyserial 먼저."); sys.exit(1)

    ser = serial.Serial(args.port, args.baud, timeout=3)
    time.sleep(2)
    ser.reset_input_buffer()
    print(f"[진단] 연결됨 {args.port}\n")

    cur = [90, 90, 90]

    def send(a1, a2, a3, suc):
        cmd = f"A{a1},{a2},{a3},{suc}\n"
        ser.write(cmd.encode())
        resp = ser.readline().decode().strip()
        print(f"    → 보냄 {cmd.strip()}  응답 {resp!r}")
        return resp

    def ramp(target, suc, step=3, delay=0.03):
        """현재→target 부드럽게 (기어 보호)."""
        t = [max(0, min(200, v)) for v in target]
        while cur != t:
            for j in range(3):
                if cur[j] < t[j]: cur[j] = min(t[j], cur[j] + step)
                elif cur[j] > t[j]: cur[j] = max(t[j], cur[j] - step)
            send(cur[0], cur[1], cur[2], suc)
            time.sleep(delay)

    def ask(q):
        return input(f"    ❓ {q} (y/n): ").strip().lower().startswith("y")

    results = {}

    tests = [
        "통신 테스트 (OK 응답 확인)",
        "서보1(베이스)만 움직이기",
        "서보2(어깨)만 움직이기",
        "서보3(팔꿈치)만 움직이기",
        "펌프 ON (서보 정지 상태) — 과전류 재현 주의",
        "펌프 ON + 서보 동시 — 과전류 재현 주의",
    ]

    while True:
        print("\n" + "=" * 50)
        for i, t in enumerate(tests):
            print(f"  [{i}] {t}")
        print("  [r] 홈(90,90,90)으로 리셋   [q] 종료+요약")
        sel = input("선택> ").strip().lower()

        if sel == "q":
            break
        if sel == "r":
            ramp([90, 90, 90], 0); continue

        if sel == "0":
            print("  통신 테스트...")
            resp = send(cur[0], cur[1], cur[2], 0)
            ok = (resp == "OK")
            print(f"  {'✅ 통신 정상' if ok else '❌ 응답 이상 — 포트/업로드/시리얼모니터 확인'}")
            results["통신"] = ok

        elif sel in ("1", "2", "3"):
            j = int(sel) - 1
            name = ["서보1(베이스)", "서보2(어깨)", "서보3(팔꿈치)"][j]
            print(f"  {name}만 90→110으로 천천히 움직입니다...")
            tgt = [90, 90, 90]; tgt[j] = 110
            ramp(tgt, 0)
            moved = ask(f"{name}가 움직이고 '힘있게' 버티나요?")
            limp = ask("이 도중에 다른 서보들이 갑자기 힘빠졌나요?")
            results[name] = ("정상" if moved and not limp else
                             "힘빠짐/전원죽음" if limp else "안움직임")
            ramp([90, 90, 90], 0)

        elif sel == "4":
            print("  ⚠️ 펌프만 ON (서보는 90 유지). 전원 스위치 손에!")
            if not ask("진행할까요?"): continue
            send(cur[0], cur[1], cur[2], 1)
            ran = ask("펌프가 도나요?")
            died = ask("펌프 켜자마자 서보·펌프가 다 힘빠졌나요?")
            send(cur[0], cur[1], cur[2], 0)
            results["펌프단독"] = ("정상작동" if ran and not died else
                                 "켜자마자 전원죽음(과전류)" if died else "반응없음")

        elif sel == "5":
            print("  ⚠️ 펌프 ON + 서보3 이동 동시. 과전류 재현 가능. 전원 스위치 손에!")
            if not ask("진행할까요?"): continue
            send(cur[0], cur[1], cur[2], 1)
            time.sleep(0.3)
            ramp([90, 90, 130], 1)
            died = ask("동시 작동 중 전원이 죽었나요?")
            send(90, 90, 90, 0)
            ramp([90, 90, 90], 0)
            results["펌프+서보동시"] = "전원죽음(합산 과전류)" if died else "정상"

        else:
            print("  잘못된 선택")

    # 종료 요약
    ser.close()
    print("\n" + "=" * 50)
    print("진단 요약")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k:14} : {v}")
    print("""
해석:
  · '펌프단독'이 켜자마자 전원죽음  → 펌프 인러시(순간전류)가 6V 공급 한계 초과
      → 펌프 전원을 PCA 안 거치고 외부전원 직결 + 인러시 억제/여유 큰 전원
  · '펌프단독' 정상인데 '펌프+서보동시' 죽음 → 합산 전류가 8A 초과
      → 펌프/서보 전원 분리 or 더 큰 전원
  · 특정 서보 이동 때 다른 서보 힘빠짐 → 그 서보/배선 접촉불량 or 전압강하
  · 통신부터 실패 → 코드/포트/시리얼모니터 문제 (전원 아님)
""")


if __name__ == "__main__":
    main()
