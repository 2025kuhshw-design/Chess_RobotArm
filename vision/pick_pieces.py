"""
기물 윗면 색 고르기 — 판 색깔과 무관하게 기물을 인식하게 만든다.

왜 이게 필요한가:
  카메라는 기물의 **윗면만** 본다. 검은 기물 윗면이 어두운 칸과 같은 색이면
  어떤 알고리즘으로도 구분할 수 없다. 실제로 검은 기물이 어두운 칸에서만
  통째로 안 잡혔다.
  → 양쪽 기물 윗면에 **체스판에 없는 색** 스티커/테이프를 붙이면 끝난다.

준비 — **진한 색 두 가지**를 쓸 것:
  · 흰쪽(사람) 기물 윗면 — 예: 초록
  · 검은쪽(로봇) 기물 윗면 — 예: 파랑
  ⚠️ 검은색 금지 — 어두운 칸과 같아져서 문제가 그대로다.
  ⚠️ 흰색도 금지 — 밝은 칸(연한 나무색)과 같아져서 똑같은 문제가 생긴다.
     (실측: 흰색/빨강 조합 40/64 vs 초록/파랑 조합 64/64)
  ⚠️ 노랑은 흡착컵 마커 색이라 피할 것 (겹치면 저장할 때 경고가 뜬다).

사용법:
  python vision/pick_pieces.py --camera 1

  1  키를 누른 뒤 → **흰쪽 기물** 윗면을 클릭 (여러 개 클릭할수록 좋음)
  2  키를 누른 뒤 → **검은쪽 기물** 윗면을 클릭
  s  vision/detect.py 에 저장하고 PIECE_COLOR_MODE 를 켠다
  r  초기화   q  저장 없이 종료

화면에서 흰쪽은 초록, 검은쪽은 파랑으로 칠해진다.
**기물 윗면만** 칠해지고 판은 안 칠해지면 성공.
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vision.detect import (ChessBoardDetector, CELL_SIZE_PX,
                           PIECE_COLOR_MIN_FRAC, MARKER_HSV_RANGES)

WIN = "pick pieces  (1=white side  2=black side  s=save  r=reset  q=quit)"
SAMPLE = 5                       # 클릭 지점 주변 표본 크기 (홀수)
PAD_H, PAD_S, PAD_V = 10, 70, 70  # 표본에서 넓힐 여유


def make_range(samples):
    a = np.array(samples)
    lo = [max(0, int(a[:, 0].min()) - PAD_H),
          max(0, int(a[:, 1].min()) - PAD_S),
          max(0, int(a[:, 2].min()) - PAD_V)]
    hi = [min(179, int(a[:, 0].max()) + PAD_H),
          min(255, int(a[:, 1].max()) + PAD_S),
          min(255, int(a[:, 2].max()) + PAD_V)]
    return tuple(lo), tuple(hi)


def overlaps(a, b):
    """두 HSV 범위가 겹치는가 (세 축 모두 겹쳐야 겹침)."""
    return all(a[0][i] <= b[1][i] and b[0][i] <= a[1][i] for i in range(3))


def warn_marker_clash(rw, rb):
    """기물 색이 흡착컵 마커 색과 겹치면 경고.
    겹치면 정렬할 때 기물을 마커로 착각한다."""
    for name, r in (("흰쪽", rw), ("검은쪽", rb)):
        for mlo, mhi in MARKER_HSV_RANGES:
            if r and overlaps(r, (mlo, mhi)):
                print(f"  ⚠️ {name} 기물 색이 흡착컵 마커 색과 겹칩니다!")
                print(f"     기물 {r}  vs  마커 ({mlo}, {mhi})")
                print("     → 정렬할 때 기물을 마커로 착각합니다. 다른 색을 쓰거나")
                print("       pick_marker.py 로 마커 색을 바꾸세요.")


def write_to_detect(rw, rb):
    """detect.py 의 기물 색 설정을 새 값으로 교체하고 색 방식을 켠다."""
    path = os.path.join(os.path.dirname(__file__), "detect.py")
    src = open(path, encoding="utf-8").read()

    def fmt(name, r):
        if r is None:
            return f"{name} = []"
        return f"{name} = [\n    ({r[0]}, {r[1]}),   # pick_pieces.py 로 실측\n]"

    # ⚠️ 반드시 줄머리(^)에 고정할 것. 위쪽 설명 주석에도
    #    "PIECE_COLOR_MODE = True 로 켠다" 라는 문구가 있어서, 고정하지 않으면
    #    그 주석이 먼저 걸리고 정작 진짜 설정값은 False 로 남는다(실제로 겪음).
    subs = [
        (r"^PIECE_COLOR_MODE = \w+", "PIECE_COLOR_MODE = True"),
        (r"^PIECE_HSV_WHITE = \[.*?\]", fmt("PIECE_HSV_WHITE", rw)),
        (r"^PIECE_HSV_BLACK = \[.*?\]", fmt("PIECE_HSV_BLACK", rb)),
    ]
    for pat, rep in subs:
        new, n = re.subn(pat, rep, src, count=1, flags=re.S | re.M)
        if n == 0:
            print(f"  [실패] detect.py 에서 {pat} 를 못 찾았습니다.")
            return False
        src = new
    open(path, "w", encoding="utf-8").write(src)
    print(f"  저장 완료 → {path}")
    print(f"    PIECE_COLOR_MODE = True")
    print(f"    흰쪽  {rw}")
    print(f"    검은쪽 {rb}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    args = ap.parse_args()

    det = ChessBoardDetector(camera_index=args.camera)
    samples = {"w": [], "b": []}
    target = ["w"]
    param = [None]

    def on_mouse(event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN or param[0] is None:
            return
        hsv = param[0]
        h, w = hsv.shape[:2]
        r = SAMPLE // 2
        patch = hsv[max(0, y-r):min(h, y+r+1),
                    max(0, x-r):min(w, x+r+1)].reshape(-1, 3).astype(int)
        samples[target[0]].extend(patch.tolist())
        m = patch.mean(axis=0)
        side = "흰쪽" if target[0] == "w" else "검은쪽"
        print(f"  [{side}] HSV≈({m[0]:.0f},{m[1]:.0f},{m[2]:.0f})  "
              f"표본 {len(samples[target[0]])}개")

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    print("\n  1 = 흰쪽 기물 고르기 / 2 = 검은쪽 기물 고르기")
    print("  기물 윗면을 클릭하세요. s=저장  r=초기화  q=종료\n")

    try:
        while True:
            ok, frame = det.cap.read()
            if not ok:
                continue
            top = det._get_top_view(frame)
            hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
            param[0] = hsv
            disp = top.copy()

            counts = {}
            for key, color in (("w", (0, 255, 0)), ("b", (255, 80, 0))):
                if not samples[key]:
                    continue
                lo, hi = make_range(samples[key])
                mask = cv2.morphologyEx(
                    cv2.inRange(hsv, np.array(lo), np.array(hi)),
                    cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                disp[mask > 0] = color
                # 기물이 있다고 판정될 칸 수
                m0 = int(CELL_SIZE_PX * 0.2)
                n = 0
                for gy in range(8):
                    for gx in range(8):
                        sl = (slice(gy*CELL_SIZE_PX+m0, (gy+1)*CELL_SIZE_PX-m0),
                              slice(gx*CELL_SIZE_PX+m0, (gx+1)*CELL_SIZE_PX-m0))
                        if (mask[sl] > 0).mean() >= PIECE_COLOR_MIN_FRAC:
                            n += 1
                counts[key] = n

            for i in range(1, 8):     # 격자선
                cv2.line(disp, (i*CELL_SIZE_PX, 0), (i*CELL_SIZE_PX, 400), (0,0,0), 1)
                cv2.line(disp, (0, i*CELL_SIZE_PX), (400, i*CELL_SIZE_PX), (0,0,0), 1)

            side = "WHITE side" if target[0] == "w" else "BLACK side"
            cv2.putText(disp, f"picking: {side}", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            txt = f"cells  white={counts.get('w','-')}  black={counts.get('b','-')}"
            cv2.putText(disp, txt + "  (16 each = OK)", (5, 392),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

            cv2.imshow(WIN, disp)
            k = cv2.waitKey(30) & 0xFF
            if k == ord('q'):
                break
            if k == ord('1'):
                target[0] = "w"; print("  → 흰쪽 기물 고르는 중")
            if k == ord('2'):
                target[0] = "b"; print("  → 검은쪽 기물 고르는 중")
            if k == ord('r'):
                samples["w"].clear(); samples["b"].clear(); print("  초기화")
            if k == ord('s'):
                if not samples["w"] and not samples["b"]:
                    print("  최소 한쪽은 골라야 합니다 (1 로 흰쪽, 2 로 검은쪽)")
                    continue
                if not samples["w"] or not samples["b"]:
                    only = "검은쪽" if samples["b"] else "흰쪽"
                    other = "흰쪽" if samples["b"] else "검은쪽"
                    print(f"  {only}만 등록합니다 → 그 색이 보이면 {only}, "
                          f"아니면 {other} 으로 봅니다.")
                    print("  (한쪽만 칠했을 때 쓰는 방식입니다)")
                rw = make_range(samples["w"]) if samples["w"] else None
                rb = make_range(samples["b"]) if samples["b"] else None
                warn_marker_clash(rw, rb)
                if write_to_detect(rw, rb):
                    print("  확인: python vision/check_board.py "
                          f"--camera {args.camera}")
                break
    finally:
        det.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
