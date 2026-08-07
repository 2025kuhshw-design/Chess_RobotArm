"""
체스판 자동 캘리브레이션 — 격자를 카메라가 스스로 찾는다.

사람이 네 모서리를 클릭하는 방식은 두 가지 한계가 있다:
  · 손으로 찍으니 몇 픽셀씩 어긋나고, 그 오차가 판 전체에 퍼진다
  · 네 점만 쓰므로 렌즈 왜곡·클릭 실수를 걸러낼 방법이 없다

여기서는 OpenCV 의 `findChessboardCorners` 로 **내부 교점 7x7 = 49개**를
한꺼번에 찾고, 그 49개에 격자를 최소제곱으로 맞춰 바깥 모서리를 계산한다.
점이 49개라 한두 개가 흔들려도 결과가 거의 안 변한다.

⚠️ 되는 조건 — 안 되면 손으로 찍는 방식으로 돌아가면 된다:
  · 체스판에 **기물이 하나도 없어야** 한다 (기물이 교점을 가린다)
  · 판 전체가 화면에 들어와야 한다
  · 밝고 고른 조명. 강한 그림자나 반사가 있으면 실패한다
  · 칸 대비가 뚜렷해야 한다 (아주 옅은 나무색 판은 실패할 수 있다)

실행:
  python vision/auto_calibrate.py --camera 1
  python vision/auto_calibrate.py --camera 1 --save     # 확인 없이 바로 저장
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "calibration.json")
# 8x8 판의 내부 교점 수 (가장자리는 교점이 아니다)
INNER = (7, 7)
# 서브픽셀 보정 창 크기
SUBPIX_WIN = (11, 11)
# 몇 프레임을 시도할지 (조명이 흔들려 한 번에 실패할 수 있다)
MAX_FRAMES = 30


def find_inner_corners(gray):
    """7x7 내부 교점을 찾아 (49,2) 로 반환. 못 찾으면 None."""
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    ok, corners = cv2.findChessboardCorners(gray, INNER, flags)
    if not ok:
        return None
    corners = cv2.cornerSubPix(
        gray, corners, SUBPIX_WIN, (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    return corners.reshape(-1, 2)


def outer_corners_from_inner(pts):
    """내부 교점 49개 → 8x8 판의 바깥 모서리 4개.

    교점 (i,j) 는 판 좌표로 (i+1, j+1) 칸 경계에 있다(i,j = 0..6).
    그 대응 49쌍으로 호모그래피를 구한 뒤 (0,0),(8,0),(8,8),(0,8) 을 보낸다.
    ⚠️ 4점만 쓰는 수동 방식과 달리 49점 최소제곱이라 한두 점이 흔들려도
       결과가 거의 안 변한다.
    """
    obj = np.array([[i + 1.0, j + 1.0] for j in range(INNER[1])
                    for i in range(INNER[0])], dtype=np.float32)
    H, mask = cv2.findHomography(obj, pts.astype(np.float32), cv2.RANSAC, 3.0)
    if H is None:
        return None, None
    board = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32)
    outer = cv2.perspectiveTransform(board.reshape(-1, 1, 2), H).reshape(-1, 2)
    # 잔차 — 얼마나 잘 맞았는지 사람이 판단할 수 있게
    proj = cv2.perspectiveTransform(obj.reshape(-1, 1, 2), H).reshape(-1, 2)
    resid = np.linalg.norm(proj - pts, axis=1)
    return outer, resid


def order_corners(outer, gray_shape):
    """detect.py 가 기대하는 순서(좌상→우상→우하→좌하)로 정렬한다.

    findChessboardCorners 는 판의 방향에 따라 순서가 뒤집혀 나올 수 있어서,
    화면 좌표 기준으로 다시 정렬해야 한다.
    """
    c = np.array(outer, dtype=float)
    s = c.sum(axis=1)          # x+y : 좌상이 최소, 우하가 최대
    d = np.diff(c, axis=1)[:, 0]   # y-x : 우상이 최소, 좌하가 최대
    return np.array([c[np.argmin(s)], c[np.argmin(d)],
                     c[np.argmax(s)], c[np.argmax(d)]])


def draw_result(frame, pts, outer, resid):
    out = frame.copy()
    cv2.drawChessboardCorners(out, INNER, pts.reshape(-1, 1, 2).astype(np.float32), True)
    poly = outer.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [poly], True, (0, 255, 255), 2)
    for i, (x, y) in enumerate(outer):
        cv2.circle(out, (int(x), int(y)), 6, (255, 0, 255), -1)
        cv2.putText(out, str(i + 1), (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(out, f"inner corners {len(pts)}  fit residual "
                     f"avg {resid.mean():.2f}px  max {resid.max():.2f}px",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--save", action="store_true",
                    help="확인 없이 바로 저장")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"카메라 {args.camera} 를 열 수 없습니다."); return
    print("\n체스판에서 기물을 모두 치우고, 판 전체가 화면에 들어오게 하세요.")
    print("격자를 찾는 중... (실패하면 조명을 밝게 하거나 각도를 조금 바꿔 보세요)\n")

    found = None
    try:
        for i in range(MAX_FRAMES):
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pts = find_inner_corners(gray)
            if pts is not None:
                found = (frame, pts)
                break
            if i % 10 == 9:
                print(f"  {i+1}프레임 시도 중...")
    finally:
        pass

    if found is None:
        cap.release()
        print("❌ 격자를 못 찾았습니다.")
        print("   · 기물이 하나라도 판 위에 있으면 실패합니다")
        print("   · 판 전체가 화면에 들어와야 합니다")
        print("   · 그림자·반사가 심하면 실패합니다")
        print("   → 수동으로: python vision/calibrate.py --camera "
              f"{args.camera}")
        return

    frame, pts = found
    outer, resid = outer_corners_from_inner(pts)
    if outer is None:
        cap.release(); print("❌ 격자는 찾았지만 모서리 계산에 실패했습니다."); return
    outer = order_corners(outer, frame.shape)

    print(f"✅ 내부 교점 {len(pts)}개 검출")
    print(f"   격자 맞춤 잔차: 평균 {resid.mean():.2f}px, 최대 {resid.max():.2f}px")
    if resid.max() > 5:
        print("   ⚠️ 잔차가 큽니다 — 렌즈 왜곡이 심하거나 일부 교점을 잘못 잡았을 수 있습니다")
    print("   계산된 8x8 바깥 모서리 (좌상→우상→우하→좌하):")
    for i, (x, y) in enumerate(outer):
        print(f"     {i+1}: ({x:.1f}, {y:.1f})")

    data = {"corners": [[round(float(x), 2), round(float(y), 2)] for x, y in outer],
            "method": "auto (findChessboardCorners)",
            "inner_points": int(len(pts)),
            "fit_residual_px": round(float(resid.max()), 3)}

    if args.save:
        with open(OUTPUT_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[저장 완료] {OUTPUT_PATH}")
        cap.release(); return

    # 확인 화면
    try:
        win = "auto calibrate  (s=save  q=quit)"
        cv2.imshow(win, draw_result(frame, pts, outer, resid))
        print("\n  노란 사각형이 8x8 영역과 맞으면 s 를 눌러 저장, q 는 취소")
        while True:
            k = cv2.waitKey(50) & 0xFF
            if k == ord('s'):
                with open(OUTPUT_PATH, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"\n[저장 완료] {OUTPUT_PATH}")
                break
            if k == ord('q'):
                print("\n취소했습니다."); break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
