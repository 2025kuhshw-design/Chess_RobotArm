"""
흡착컵 마커 색 고르기 — 화면에서 마커를 클릭하면 HSV 범위를 계산해 준다.

마커가 안 잡히는 이유는 대부분 `MARKER_HSV_RANGES` 가 실제 색과 안 맞아서다.
색을 눈으로 추측해 숫자를 고치는 건 어렵다. 실제 화면에서 그 픽셀들이
어떤 HSV 값인지 직접 재는 게 훨씬 빠르고 정확하다.

사용법:
  python vision/pick_marker.py --camera 1

  1. 팔을 체스판 위(마커가 카메라에 보이는 위치)로 옮겨 둔다
  2. 창이 뜨면 **마커 위를 클릭**한다 (여러 번 클릭하면 범위가 넓어진다)
  3. 화면에 잡힌 영역이 초록으로 표시된다 — 마커만 초록이면 성공
  4. `s` 를 누르면 vision/detect.py 에 바로 적용
     `r` 초기화 / `q` 저장 없이 종료

⚠️ 팔이 체스판 밖에 있으면 탑뷰에 안 나온다. 먼저 test_square 로
   아무 칸(예: e5) 위로 보낸 뒤 실행할 것.
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vision.detect import (ChessBoardDetector, TOP_SIZE, CELL_SIZE_PX,
                           MARKER_MIN_AREA, MARKER_MAX_AREA)

WIN = "pick marker  (click marker / s=save  r=reset  q=quit)"
# 클릭 지점 주변 이 크기의 정사각형을 표본으로 삼는다 (홀수)
SAMPLE = 5
# 표본 HSV 에서 좌우로 넓힐 여유. H는 좁게, S·V는 넉넉히.
PAD_H, PAD_S, PAD_V = 8, 60, 60


def sample_hsv(hsv, x, y):
    """클릭 지점 주변의 HSV 표본을 (N,3) 으로."""
    h, w = hsv.shape[:2]
    r = SAMPLE // 2
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    return hsv[y0:y1, x0:x1].reshape(-1, 3).astype(int)


def make_range(samples):
    """표본들을 모두 담는 HSV 범위 하나를 만든다."""
    a = np.array(samples)
    lo = [max(0, int(a[:, 0].min()) - PAD_H),
          max(0, int(a[:, 1].min()) - PAD_S),
          max(0, int(a[:, 2].min()) - PAD_V)]
    hi = [min(179, int(a[:, 0].max()) + PAD_H),
          min(255, int(a[:, 1].max()) + PAD_S),
          min(255, int(a[:, 2].max()) + PAD_V)]
    return tuple(lo), tuple(hi)


def write_to_detect(lo, hi):
    """vision/detect.py 의 MARKER_HSV_RANGES 를 새 값으로 교체."""
    path = os.path.join(os.path.dirname(__file__), "detect.py")
    src = open(path, encoding="utf-8").read()
    new = (f"MARKER_HSV_RANGES = [\n"
           f"    ({lo}, {hi}),   # pick_marker.py 로 실측\n"
           f"]")
    pat = re.compile(r"MARKER_HSV_RANGES = \[.*?\n\]", re.S)
    if not pat.search(src):
        print("  [실패] detect.py 에서 MARKER_HSV_RANGES 를 못 찾았습니다.")
        return False
    open(path, "w", encoding="utf-8").write(pat.sub(new, src, count=1))
    print(f"  저장 완료 → {path}")
    print(f"    MARKER_HSV_RANGES = [({lo}, {hi})]")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--raw", action="store_true",
                    help="탑뷰로 펴지 않고 원본 화면에서 고른다 "
                         "(팔이 체스판 밖에 있을 때)")
    args = ap.parse_args()

    det = ChessBoardDetector(camera_index=args.camera)
    samples = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv = param[0]
            if hsv is None:
                return
            s = sample_hsv(hsv, x, y)
            samples.extend(s.tolist())
            m = s.mean(axis=0)
            print(f"  클릭 ({x},{y}) HSV≈({m[0]:.0f},{m[1]:.0f},{m[2]:.0f})  "
                  f"표본 {len(samples)}개")

    param = [None]
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse, param)

    print("\n마커 위를 클릭하세요. 여러 번 클릭하면 범위가 넓어집니다.")
    print("  s=detect.py에 저장   r=초기화   q=저장 없이 종료\n")

    try:
        while True:
            ok, frame = det.cap.read()
            if not ok:
                continue
            view = frame if args.raw else det._get_top_view(frame)
            hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)
            param[0] = hsv
            disp = view.copy()

            if samples:
                lo, hi = make_range(samples)
                mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                        np.ones((3, 3), np.uint8))
                disp[mask > 0] = (0, 255, 0)          # 잡힌 영역 = 초록

                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                areas = sorted((cv2.contourArea(c) for c in cnts), reverse=True)
                big = areas[0] if areas else 0
                ok_area = MARKER_MIN_AREA <= big <= MARKER_MAX_AREA
                cv2.putText(disp, f"H{lo[0]}-{hi[0]} S{lo[1]}-{hi[1]} "
                                  f"V{lo[2]}-{hi[2]}", (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                cv2.putText(disp, f"blobs={len(cnts)} max={big:.0f}px "
                                  f"({'OK' if ok_area else 'BAD'})", (5, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 160, 0) if ok_area else (0, 0, 255), 1)
            else:
                cv2.putText(disp, "click on the marker", (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow(WIN, disp)
            k = cv2.waitKey(30) & 0xFF
            if k == ord('q'):
                break
            if k == ord('r'):
                samples.clear(); print("  초기화")
            if k == ord('s'):
                if not samples:
                    print("  먼저 마커를 클릭하세요"); continue
                lo, hi = make_range(samples)
                if write_to_detect(lo, hi):
                    print("  이제 test_square 의 'mark' 로 확인해 보세요.")
                break
    finally:
        det.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
