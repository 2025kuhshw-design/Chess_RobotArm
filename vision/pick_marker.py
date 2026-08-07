"""
흡착컵 마커 색 고르기 — 화면에서 마커를 클릭하면 HSV 범위를 계산해 준다.

마커가 안 잡히는 이유는 대부분 `MARKER_HSV_RANGES` 가 실제 색과 안 맞아서다.
색을 눈으로 추측해 숫자를 고치는 건 어렵다. 실제 화면에서 그 픽셀들이
어떤 HSV 값인지 직접 재는 게 훨씬 빠르고 정확하다.

⚠️ 한 자세에서만 클릭하면 **다른 칸에서 마커를 놓친다**. 실제로 겪은 일:
   랭크 8 전체에서 마커가 안 잡혔다. 팔이 로봇 쪽으로 접히면 마커가
   그늘에 들어가 어두워지는데, 밝을 때 잡은 V 하한(80) 아래로 떨어져서다.
     밝은 빨강   V=200 → 범위 안
     그늘진 빨강 V=42  → 범위 밖  (감마를 올리면 더 심해진다)
   → **팔을 여러 칸으로 옮겨 가며** 클릭해서 밝기 변화를 다 담아야 한다.
     가까운 칸(랭크8)과 먼 칸(랭크3)을 꼭 포함할 것.

사용법:
  python vision/pick_marker.py --camera 1

  1. 팔을 판 **밖**으로 치우고 `b` — 판 전체를 음성 표본으로 모은다
     (마커 범위가 판 색까지 삼키지 않게 하는 기준점)
  2. 팔을 랭크8 쪽 칸으로 보내고 → 마커 클릭
  3. 팔을 랭크5, 랭크3 쪽으로도 보내고 → 각각 마커 클릭
     (다른 창에서 test_square 의 `move e8` 같은 명령으로 옮긴다)
  4. 화면에 잡힌 영역이 초록으로 표시된다 — 마커만 초록이면 성공
  5. `s` 를 누르면 vision/colors.json 에 저장된다
     `3` 판(음성) 클릭 / `r` 초기화 / `q` 저장 없이 종료

이 도구는 colors.json 의 **marker 항목만** 바꾼다. 캘리브레이션·빈 판
기준 영상·기물 색은 건드리지 않으므로 그것들을 다시 할 필요가 없다.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vision.detect import (ChessBoardDetector,
                           MARKER_MIN_AREA, MARKER_MAX_AREA,
                           PIECE_HSV_WHITE, PIECE_HSV_BLACK,
                           save_colors, COLORS_PATH)
from vision.pick_pieces import fit_range, cell_center_pixels

WIN = "pick marker  (click marker  3=board  b=auto-board  s=save r=reset q=quit)"
# 클릭 지점 주변 이 크기의 정사각형을 표본으로 삼는다 (홀수)
SAMPLE = 5
# 마커가 그늘에 들어가도 놓치지 않도록 V 하한을 이 값까지 끌어내린다.
# ⚠️ 마커는 팔 자세에 따라 밝기가 크게 변한다. 반면 색상(H)·채도(S)는
#    거의 그대로다. 그래서 V 는 넓게 열고 H·S 로 가르는 편이 안전하다.
#    단, 음성(판 색)이 들어오면 그만큼만 열린다 — fit_range 가 판단한다.
V_FLOOR = 30
# 축별로 넓혀도 되는 한계.
#   H : 좁게 — 색상은 조명이 바뀌어도 거의 안 변한다. 넓히면 음성 표본에
#       없던 색(파란 기물 H=112 등)까지 마커로 잡는다. 실제로 겪었다.
#   S : 보통
#   V : 넓게 — 팔 자세에 따라 밝기가 가장 크게 변하는 축이다.
PAD_MAX = (10, 40, 255)


def sample_hsv(hsv, x, y):
    """클릭 지점 주변의 HSV 표본을 (N,3) 으로."""
    h, w = hsv.shape[:2]
    r = SAMPLE // 2
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    return hsv[y0:y1, x0:x1].reshape(-1, 3).astype(int)


def _wraps(samples):
    """빨강처럼 H=0 을 감싸는 색인가 (H가 0 근처와 179 근처에 모두 있는가)."""
    h = np.asarray(samples)[:, 0]
    return bool((h < 15).any() and (h > 165).any())


def _rot(a, on):
    """H를 +90 돌린다. 0을 감싸는 색을 '가운데'로 옮겨 연속 구간으로 만든다."""
    a = np.asarray(a, dtype=float).copy()
    if on and len(a):
        a[:, 0] = (a[:, 0] + 90.0) % 180.0
    return a


def make_ranges(samples, neg=()):
    """표본을 담되 **판 색은 안 들어오는 선까지만** 넓힌 HSV 범위 목록.

    ⚠️ 예전에는 무조건 H±8 S±60 V±60 으로 넓혔다. 두 방향으로 다 틀린다:
       판 색을 삼키거나(오검출), 그늘진 마커를 놓친다(V 하한이 밝을 때
       표본에 묶여서). 음성 표본을 기준으로 넓힐 폭을 정하면 둘 다 풀린다.

    ⚠️ 빨강은 H가 0을 감싼다(밝을 때 H≈0, 짙은 그늘에서 H≈178).
       구간 하나로는 표현할 수 없어서, 감싸는 경우 H를 90도 돌려 맞춘 뒤
       되돌리면서 **두 구간**으로 쪼갠다. detect 는 구간 목록을 OR 한다.
    """
    rot = _wraps(samples)
    pos_r = _rot(samples, rot)
    neg_r = _rot(neg, rot) if len(neg) else []
    lo, hi, pads, leak = fit_range(pos_r.tolist(),
                                   neg_r.tolist() if len(neg) else [],
                                   pad_max=PAD_MAX)

    # 마커는 그늘에서 어두워진다 — V 하한만 추가로 끌어내린다.
    # 단 음성이 그 안에 들어오면 되돌린다.
    lo = list(lo)
    cand = list(lo); cand[2] = min(lo[2], V_FLOOR)
    if len(neg):
        a = np.asarray(neg_r, dtype=float)
        if np.all((a >= np.array(cand)) & (a <= np.array(hi)),
                  axis=1).sum() <= len(a) * 0.002:
            lo = cand
    else:
        lo = cand

    if not rot:
        return [(tuple(lo), tuple(hi))], pads, leak

    lo_h, hi_h = (lo[0] - 90) % 180, (hi[0] - 90) % 180
    if lo_h <= hi_h:
        return [((lo_h, lo[1], lo[2]), (hi_h, hi[1], hi[2]))], pads, leak
    # 0을 감싼다 → 0쪽 구간과 179쪽 구간으로 나눈다
    return ([((0, lo[1], lo[2]), (hi_h, hi[1], hi[2])),
             ((lo_h, lo[1], lo[2]), (179, hi[1], hi[2]))], pads, leak)


def mask_of(hsv, ranges):
    m = None
    for lo, hi in ranges:
        one = cv2.inRange(hsv, np.array(lo), np.array(hi))
        m = one if m is None else cv2.bitwise_or(m, one)
    return m


def warn_piece_clash(ranges):
    """마커 범위가 등록된 기물 색과 겹치면 경고.

    겹치면 게임 중에 기물을 마커로 착각해 엉뚱한 곳으로 정렬한다.
    (pick_pieces 의 반대 방향 검사와 짝을 이룬다)
    """
    def ov(a, b):
        return all(a[0][i] <= b[1][i] and b[0][i] <= a[1][i] for i in range(3))
    for name, rs in (("흰쪽", PIECE_HSV_WHITE), ("검은쪽", PIECE_HSV_BLACK)):
        for pr in rs or []:
            for mr in ranges:
                if ov(mr, pr):
                    print(f"  ⚠️ 마커 범위가 {name} 기물 색과 겹칩니다!")
                    print(f"     마커 {mr}  vs  기물 {pr}")
                    print("     → 게임 중 기물을 마커로 착각합니다.")
                    return


def write_to_detect(ranges):
    """실측한 마커 색을 vision/colors.json 에 저장한다.

    ⚠️ detect.py 를 직접 고치지 않는다 — git이 추적하는 파일이라 고치면
       `git pull` 이 매번 충돌한다.
    ⚠️ colors.json 의 marker 항목만 바꾼다. 캘리브레이션·빈 판 기준 영상·
       기물 색은 그대로 남으므로 그것들을 다시 할 필요가 없다.
    """
    save_colors(marker=list(ranges))
    print(f"  저장 완료 → {COLORS_PATH}")
    for lo, hi in ranges:
        print(f"    MARKER_HSV_RANGES += ({lo}, {hi})")
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
    neg = []
    target = ["m"]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv = param[0]
            if hsv is None:
                return
            s = sample_hsv(hsv, x, y)
            (samples if target[0] == "m" else neg).extend(s.tolist())
            m = s.mean(axis=0)
            name = "마커" if target[0] == "m" else "판(음성)"
            n = len(samples) if target[0] == "m" else len(neg)
            print(f"  [{name}] ({x},{y}) HSV≈({m[0]:.0f},{m[1]:.0f},{m[2]:.0f})  "
                  f"표본 {n}개")

    param = [None]
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse, param)

    print("\n마커 위를 클릭하세요. 여러 번 클릭하면 범위가 넓어집니다.")
    print("  s=colors.json에 저장   r=초기화   q=저장 없이 종료\n")

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
                ranges, _pads, leak = make_ranges(samples, neg)
                lo, hi = ranges[0][0], ranges[-1][1]
                mask = cv2.morphologyEx(mask_of(hsv, ranges), cv2.MORPH_OPEN,
                                        np.ones((3, 3), np.uint8))
                disp[mask > 0] = (0, 255, 0)          # 잡힌 영역 = 초록

                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                areas = sorted((cv2.contourArea(c) for c in cnts), reverse=True)
                big = areas[0] if areas else 0
                ok_area = MARKER_MIN_AREA <= big <= MARKER_MAX_AREA
                htxt = " | ".join(f"H{a[0]}-{b[0]}" for a, b in ranges)
                cv2.putText(disp, f"{htxt} S{lo[1]}-{hi[1]} "
                                  f"V{lo[2]}-{hi[2]}   neg={len(neg)}", (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                cv2.putText(disp, f"blobs={len(cnts)} max={big:.0f}px "
                                  f"({'OK' if ok_area else 'BAD'})"
                                  f"{'  !! board leak' if leak else ''}",
                            (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 160, 0) if ok_area and not leak else (0, 0, 255), 1)
                # 그늘진 자세에서 놓칠 위험을 미리 알린다
                if lo[2] > 60:
                    cv2.putText(disp, "V floor high - may lose marker in shadow",
                                (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                (0, 140, 255), 1)
            else:
                cv2.putText(disp, f"click on the marker  "
                                  f"(picking: {'MARKER' if target[0]=='m' else 'BOARD'})",
                            (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow(WIN, disp)
            k = cv2.waitKey(30) & 0xFF
            if k == ord('q'):
                break
            if k == ord('1'):
                target[0] = "m"; print("  → 마커 고르는 중")
            if k == ord('3'):
                target[0] = "n"; print("  → 판(음성) 고르는 중")
            if k == ord('b'):
                # ⚠️ 팔이 판 위에 있으면 마커까지 음성으로 넣어 버린다.
                for gy in range(8):
                    for gx in range(8):
                        neg.extend(cell_center_pixels(hsv, gx, gy, 3).tolist())
                print(f"  판 64칸에서 음성 표본 수집 → 총 {len(neg)}개")
                print("  (팔이 판 위에 있었다면 r 로 초기화하고 다시 하세요)")
            if k == ord('r'):
                samples.clear(); neg.clear(); print("  초기화")
            if k == ord('s'):
                if not samples:
                    print("  먼저 마커를 클릭하세요"); continue
                if not neg:
                    print("  ⚠️ 판(음성) 표본이 없습니다 — 마커 범위가 판 색을")
                    print("     삼키는지 확인할 방법이 없습니다. b 를 권합니다.")
                ranges, pads, leak = make_ranges(samples, neg)
                lo, hi = ranges[0][0], ranges[-1][1]
                for a, b in ranges:
                    print(f"  범위 H {a[0]}~{b[0]}  S {a[1]}~{b[1]}  V {a[2]}~{b[2]}")
                if len(ranges) > 1:
                    print("  (빨강이 H=0 을 감싸서 두 구간으로 저장합니다)")
                if leak:
                    print(f"  ❌ 판 색 {leak}개가 범위 안에 있습니다 — 오검출 위험")
                if lo[2] > 60:
                    print(f"  ⚠️ V 하한이 {lo[2]} 로 높습니다. 팔이 그늘에 들어가는")
                    print("     칸(주로 랭크8)에서 마커를 놓칠 수 있습니다.")
                    print("     → 그 칸으로 팔을 옮겨 마커를 한 번 더 클릭하세요.")
                warn_piece_clash(ranges)
                if write_to_detect(ranges):
                    print("  이제 test_square 의 'mark' 로 확인해 보세요.")
                break
    finally:
        det.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
