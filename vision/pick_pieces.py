"""
기물 윗면 색 고르기 — 판 색깔과 무관하게 기물을 인식하게 만든다.

왜 이게 필요한가:
  카메라는 기물의 **윗면만** 본다. 기물 윗면이 그 아래 칸과 같은 색이면
  어떤 알고리즘으로도 구분할 수 없다.

이 도구가 예전에 실패한 이유 (실제로 겪은 일):
  흰 기물과 카페라떼색 칸은 색상(H)이 거의 같다 — 둘 다 H≈18.
  갈라놓는 축은 **채도(S)** 하나뿐인데(흰 S≈5, 라떼 S≈83), 예전 코드는
  표본 범위를 무조건 S±70 으로 넓혔다. 그러면 흰 기물 범위가 S 0~75 가 되어
  라떼(83)와 사실상 붙어버린다. **넓히는 폭이 문제였지 색이 문제가 아니었다.**

그래서 지금은:
  · **음성 표본**(판 색)을 같이 모으고, 그 음성이 하나도 안 들어가는
    **가장 큰 상자**를 자동으로 찾는다. 넓힐 수 있는 만큼만 넓힌다.
  · **감마**로 채도 간격 자체를 벌린다(g=2.0 이면 흰↔라떼 간격 78→129).
    감마는 detect.py 의 탑뷰 한 곳에서만 걸리므로 실제 인식에도 그대로 적용된다.

사용법:
  python vision/pick_pieces.py --camera 1

  1  → **흰쪽 기물** 윗면을 클릭 (여러 개 클릭할수록 좋음)
  2  → **검은쪽 기물** 윗면을 클릭
  3  → **판**(기물 없는 칸)을 클릭 — 여기에 들어가면 안 되는 색
  b  → 시작 배치라면 랭크 3~6(빈 칸 32개)을 음성 표본으로 자동 수집  ★추천
  [ ] → 감마 -0.1 / +0.1  (어둡게 하면 채도 대비가 커진다)
  s  vision/colors.json 에 저장하고 색 방식을 켠다
  r  초기화   q  저장 없이 종료

화면에서 흰쪽은 초록, 검은쪽은 파랑으로 칠해진다.
**기물 윗면만** 칠해지고 판은 안 칠해지면 성공. 아래 줄이 16/16 이면 완성.

⚠️ 감마를 바꿔서 저장했다면 빈 판 기준 영상도 다시 찍어야 한다:
     python vision/check_board.py --camera 1 --capture-empty
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import vision.detect as vd
from vision.detect import (ChessBoardDetector, CELL_SIZE_PX, MARGIN_PX,
                           BOARD_PX, cell_px, chess_to_grid,
                           PIECE_COLOR_MIN_FRAC, MARKER_HSV_RANGES,
                           save_colors, COLORS_PATH)

WIN = ("pick pieces  (1=white 2=black 3=board  b=auto-board  "
       "[ ]=gamma  s=save r=reset q=quit)")
SAMPLE = 5                       # 클릭 지점 주변 표본 크기 (홀수)

# 상자를 넓힐 후보 폭. 음성 표본이 안 들어가는 선에서 가장 큰 조합을 고른다.
PAD_CANDIDATES = (70, 55, 40, 30, 24, 18, 14, 10, 8, 6, 4, 2, 0)
# 클릭이 기물 테두리에 걸렸을 때를 대비해 양쪽 끝은 잘라낸다.
CORE_PCT = (2, 98)
# 그래도 음성이 들어가면 더 세게 잘라 본다.
FALLBACK_PCT = ((5, 95), (10, 90), (20, 80), (30, 70))
HSV_MAX = np.array([179.0, 255.0, 255.0])


def _box(samples, pct):
    a = np.asarray(samples, dtype=float)
    return (np.percentile(a, pct[0], axis=0), np.percentile(a, pct[1], axis=0))


def _n_inside(lo, hi, neg):
    if len(neg) == 0:
        return 0
    return int(np.all((neg >= lo) & (neg <= hi), axis=1).sum())


def fit_range(pos, neg):
    """양성은 감싸되 음성은 하나도 안 들어가는 **가장 큰** HSV 상자.

    반환: ((lo), (hi), pads, leak)
      pads = 축별로 실제 넓힌 폭, leak = 그래도 들어간 음성 표본 수(0이면 완벽)

    ⚠️ 고정 폭(±70)으로 넓히던 예전 방식과 다른 점이 여기다. 넓힐 여유가
       있는 축(H, V)은 넉넉히 넓히고, 여유가 없는 축(흰 기물의 S)은 거의
       안 넓힌다. 판정 상자가 판 색에 닿는 순간을 자동으로 감지한다.
    """
    neg = np.asarray(neg, dtype=float) if len(neg) else np.zeros((0, 3))

    for pct in (CORE_PCT,) + FALLBACK_PCT:
        lo0, hi0 = _box(pos, pct)
        best = None
        for ph in PAD_CANDIDATES:
            for ps in PAD_CANDIDATES:
                for pv in PAD_CANDIDATES:
                    pad = np.array([ph, ps, pv], dtype=float)
                    lo = np.maximum(0.0, lo0 - pad)
                    hi = np.minimum(HSV_MAX, hi0 + pad)
                    if _n_inside(lo, hi, neg):
                        continue
                    score = ph + ps + pv
                    if best is None or score > best[0]:
                        best = (score, lo, hi, (ph, ps, pv))
            if best is not None and best[3][0] == PAD_CANDIDATES[0]:
                break          # H를 최대로 넓히고도 통과 — 더 볼 필요 없음
        if best is not None:
            _, lo, hi, pads = best
            return _as_int(lo), _as_int(hi), pads, 0

    # 여기까지 왔다면 기물 색과 판 색이 HSV 상에서 진짜로 겹친다.
    lo0, hi0 = _box(pos, FALLBACK_PCT[-1])
    return _as_int(lo0), _as_int(hi0), (0, 0, 0), _n_inside(lo0, hi0, neg)


def _as_int(v):
    return tuple(int(round(x)) for x in v)


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


def write_colors(rw, rb, gamma):
    """실측한 기물 색을 vision/colors.json 에 저장하고 색 방식을 켠다.

    ⚠️ detect.py 를 직접 고치지 않는다. detect.py 는 git이 추적하는 파일이라
       고치면 `git pull` 이 매번 충돌한다. 색은 설치 환경마다 다른 값이므로
       calibration.json 처럼 별도 파일에 둔다(git 제외).
    """
    save_colors(white=[rw] if rw else [],
                black=[rb] if rb else [],
                color_mode=True, gamma=gamma)
    print(f"  저장 완료 → {COLORS_PATH}")
    print(f"    PIECE_COLOR_MODE = True")
    print(f"    감마   {gamma:.2f}")
    print(f"    흰쪽   {rw}")
    print(f"    검은쪽 {rb}")
    return True


def cell_center_pixels(hsv, gx, gy, step=2):
    """칸 중앙 60% 의 HSV 픽셀들 (N,3)."""
    x0, y0 = cell_px(gx, gy)
    m = int(CELL_SIZE_PX * 0.2)
    sl = hsv[y0+m:y0+CELL_SIZE_PX-m:step, x0+m:x0+CELL_SIZE_PX-m:step]
    return sl.reshape(-1, 3).astype(int)


def harvest_empty_cells(hsv):
    """시작 배치 기준으로 **비어 있는 랭크 3~6** 의 32칸을 음성 표본으로.

    시작 배치에서는 랭크 1·2 가 흰쪽, 7·8 이 검은쪽, 3~6 은 비어 있다.
    이 32칸에는 밝은 칸과 어두운 칸이 반씩 들어 있어서 판 색을 고루 담는다.
    ROBOT_SIDE 가 무엇이든 chess_to_grid 가 알아서 화면 격자로 바꿔 준다.
    """
    out = []
    for row in range(2, 6):          # 랭크 3~6
        for col in range(8):
            gx, gy = chess_to_grid(col, row)
            out.append(cell_center_pixels(hsv, gx, gy, step=3))
    return np.concatenate(out).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=None,
                    help="시작 감마 (기본: colors.json 값). 1보다 크면 어두워지고 "
                         "채도 대비가 커진다")
    args = ap.parse_args()

    if args.gamma is not None:
        vd.set_gamma(args.gamma)

    det = ChessBoardDetector(camera_index=args.camera)
    samples = {"w": [], "b": [], "n": []}
    target = ["w"]
    param = [None]
    fitted = {}          # key -> (lo, hi, pads, leak)
    sig = [None]         # 표본이 바뀌었는지 확인용

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
        side = {"w": "흰쪽", "b": "검은쪽", "n": "판(음성)"}[target[0]]
        print(f"  [{side}] HSV≈({m[0]:.0f},{m[1]:.0f},{m[2]:.0f})  "
              f"표본 {len(samples[target[0]])}개")

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    print("\n  1 = 흰쪽 기물 / 2 = 검은쪽 기물 / 3 = 판(들어가면 안 되는 색)")
    print("  b = 시작 배치의 빈 칸 32개를 음성 표본으로 자동 수집  ★추천")
    print("  [ ] = 감마 조절   s=저장  r=초기화  q=종료\n")

    try:
        while True:
            ok, frame = det.cap.read()
            if not ok:
                continue
            top = det._get_top_view(frame)        # 감마가 여기서 적용된다
            hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
            param[0] = hsv
            disp = top.copy()

            # 표본이 바뀌었을 때만 상자를 다시 맞춘다 (탐색이 무거워서)
            cur = (len(samples["w"]), len(samples["b"]), len(samples["n"]))
            if cur != sig[0]:
                sig[0] = cur
                fitted.clear()
                for key in ("w", "b"):
                    if samples[key]:
                        fitted[key] = fit_range(samples[key], samples["n"])

            counts = {}
            for key, color in (("w", (0, 255, 0)), ("b", (255, 80, 0))):
                if key not in fitted:
                    continue
                lo, hi, _pads, _leak = fitted[key]
                mask = cv2.morphologyEx(
                    cv2.inRange(hsv, np.array(lo), np.array(hi)),
                    cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                disp[mask > 0] = color
                # 기물이 있다고 판정될 칸 수
                m0 = int(CELL_SIZE_PX * 0.2)
                n = 0
                for gy in range(8):
                    for gx in range(8):
                        _x0, _y0 = cell_px(gx, gy)
                        sl = (slice(_y0+m0, _y0+CELL_SIZE_PX-m0),
                              slice(_x0+m0, _x0+CELL_SIZE_PX-m0))
                        if (mask[sl] > 0).mean() >= PIECE_COLOR_MIN_FRAC:
                            n += 1
                counts[key] = n

            for i in range(1, 8):     # 격자선 (판 영역 안에만)
                gx0, gy0 = cell_px(i, i)
                cv2.line(disp, (gx0, MARGIN_PX), (gx0, MARGIN_PX+BOARD_PX), (0,0,0), 1)
                cv2.line(disp, (MARGIN_PX, gy0), (MARGIN_PX+BOARD_PX, gy0), (0,0,0), 1)
            cv2.rectangle(disp, (MARGIN_PX, MARGIN_PX),
                          (MARGIN_PX+BOARD_PX-1, MARGIN_PX+BOARD_PX-1), (0,0,0), 2)

            side = {"w": "WHITE side", "b": "BLACK side",
                    "n": "BOARD (negative)"}[target[0]]
            cv2.putText(disp, f"picking: {side}   gamma {vd.GAMMA:.2f}   "
                              f"neg {len(samples['n'])}", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            leak = sum(f[3] for f in fitted.values())
            txt = (f"cells  white={counts.get('w','-')}  black={counts.get('b','-')}"
                   f"  (16 each = OK)")
            if leak:
                txt += f"  !! board leak {leak}"
            cv2.putText(disp, txt, (5, disp.shape[0]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

            cv2.imshow(WIN, disp)
            k = cv2.waitKey(30) & 0xFF
            if k == ord('q'):
                break
            if k == ord('1'):
                target[0] = "w"; print("  → 흰쪽 기물 고르는 중")
            if k == ord('2'):
                target[0] = "b"; print("  → 검은쪽 기물 고르는 중")
            if k == ord('3'):
                target[0] = "n"; print("  → 판(음성) 고르는 중 — 빈 칸을 클릭하세요")
            if k == ord('b'):
                samples["n"].extend(harvest_empty_cells(hsv))
                print(f"  빈 칸 32개에서 음성 표본 수집 → 총 "
                      f"{len(samples['n'])}개")
                print("  (시작 배치가 아니면 r 로 초기화하고 3 으로 직접 클릭하세요)")
            if k in (ord('['), ord(']')):
                g = max(0.4, min(3.0, vd.GAMMA + (0.1 if k == ord(']') else -0.1)))
                vd.set_gamma(g)
                # 감마가 바뀌면 예전 표본의 HSV 는 더 이상 유효하지 않다
                samples["w"].clear(); samples["b"].clear(); samples["n"].clear()
                print(f"  감마 {g:.2f} — 밝기가 바뀌었으므로 표본을 비웠습니다. "
                      "다시 고르세요")
            if k == ord('r'):
                samples["w"].clear(); samples["b"].clear(); samples["n"].clear()
                print("  초기화")
            if k == ord('s'):
                if not samples["w"] and not samples["b"]:
                    print("  최소 한쪽은 골라야 합니다 (1 로 흰쪽, 2 로 검은쪽)")
                    continue
                if not samples["n"]:
                    print("  ⚠️ 판(음성) 표본이 없습니다 — 판 색에 닿는지 확인할 "
                          "방법이 없어 예전처럼 뭉갤 수 있습니다.")
                    print("     b 를 눌러 자동 수집하거나 3 으로 빈 칸을 클릭하세요.")
                if not samples["w"] or not samples["b"]:
                    only = "검은쪽" if samples["b"] else "흰쪽"
                    other = "흰쪽" if samples["b"] else "검은쪽"
                    print(f"  {only}만 등록합니다 → 그 색이 보이면 {only}, "
                          f"아니면 {other} 으로 봅니다.")
                    print("  (한쪽만 칠했을 때 쓰는 방식입니다)")
                rw = rb = None
                for key, name in (("w", "흰쪽"), ("b", "검은쪽")):
                    if key not in fitted:
                        continue
                    lo, hi, pads, lk = fitted[key]
                    print(f"  {name}: H {lo[0]}~{hi[0]}  S {lo[1]}~{hi[1]}  "
                          f"V {lo[2]}~{hi[2]}   (넓힌 폭 H±{pads[0]} S±{pads[1]} "
                          f"V±{pads[2]})")
                    if lk:
                        print(f"     ❌ 판 색 {lk}개가 이 범위 안에 있습니다 — "
                              "이 색으론 구분이 안 됩니다.")
                        print("        ] 로 감마를 올려 보고, 그래도 안 되면 "
                              f"{name} 기물 윗면을 진한 색으로 칠하세요.")
                    if key == "w":
                        rw = (lo, hi)
                    else:
                        rb = (lo, hi)
                warn_marker_clash(rw, rb)
                if write_colors(rw, rb, vd.GAMMA):
                    print("  확인: python vision/check_board.py "
                          f"--camera {args.camera}")
                    if abs(vd.GAMMA - 1.0) > 1e-3:
                        print("  ⚠️ 감마를 바꿨으니 빈 판 기준 영상도 다시:")
                        print(f"     python vision/check_board.py --camera "
                              f"{args.camera} --capture-empty")
                break
    finally:
        det.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
