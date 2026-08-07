"""
카메라 노출·화이트밸런스 고정 — "watch 하면 맞는데 한 번만 읽으면 틀린다"의 해답.

무엇을 겪었나:
  · `check_board --watch` 는 61 → 62 → 64/64 로 올라가며 결국 맞았다
  · 그런데 `--watch` 없이 한 장만 읽으면 48/64
  · 감마를 3.0 까지 올려도 카페라떼 칸이 흰 기물 범위에 들어왔다

왜 그런가:
  자동노출(AE)과 자동 화이트밸런스(AWB)는 **수 초에 걸쳐 수렴**한다.
  라이브 뷰는 계속 돌기 때문에 수렴한 뒤의 색을 보지만, 한 번 열고 바로
  읽는 프로그램은 수렴 전 색을 본다. 그래서 같은 장면인데 판정이 갈렸다.

  특히 AWB 가 고약하다. AWB 는 화면 평균을 무채색으로 맞추려 하는데,
  이 체스판은 대부분이 베이지(카페라떼)색이다. 그래서 AWB 가 그 색조를
  통째로 상쇄해 **밝은 칸을 흰색으로 만들어 버린다**. 실측 역산으로는
  밝은 칸의 채도가 3 까지 떨어졌다(나무색이면 80 근처여야 한다).
  채도가 0에 가까우면 흰 기물과 같은 색이므로 원리상 구분이 불가능하고,
  감마로도 못 살린다(0에 뭘 곱해도 0이다).

이 도구가 하는 일:
  1. AE/AWB 가 **수렴할 때까지** 기다린다 (채널별 평균이 멈출 때까지)
  2. 그때의 노출·화이트밸런스 값을 되읽는다
  3. 그 값으로 **고정**하고, 정말 고정됐는지 다시 확인한다
  4. vision/colors.json 에 저장한다 → 모든 도구가 같은 색을 보게 된다

실행:
  python vision/lock_camera.py --camera 1
  python vision/lock_camera.py --camera 1 --unlock       # 자동으로 되돌리기

⚠️ 카메라가 이 속성을 지원하지 않으면 고정에 실패한다. 그 경우 아래를 쓴다:
     v4l2-ctl -d /dev/video1 --list-ctrls
     v4l2-ctl -d /dev/video1 -c auto_exposure=1 -c exposure_time_absolute=250
     v4l2-ctl -d /dev/video1 -c white_balance_automatic=0 -c white_balance_temperature=4600
   그래도 안 되면, 최소한 프로그램을 켜고 3초쯤 기다렸다 읽으면 된다.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import vision.detect as vd
from vision.detect import save_colors, COLORS_PATH, apply_camera_controls

SETTLE_TOL = 0.3      # 채널 평균이 이만큼 안 움직이면 '멈췄다'
SETTLE_N = 12         # 연속 몇 프레임 멈춰야 수렴으로 볼지
MAX_FRAMES = 300      # 10초쯤


def settle(cap, verbose=True):
    """AE/AWB 가 멈출 때까지 프레임을 흘려보낸다. 반환: (수렴했는가, 프레임수)"""
    prev, stable, used = None, 0, 0
    for _ in range(MAX_FRAMES):
        ok, f = cap.read()
        if not ok:
            continue
        used += 1
        m = f[::8, ::8].reshape(-1, 3).mean(axis=0)
        if prev is not None and float(np.abs(m - prev).max()) < SETTLE_TOL:
            stable += 1
            if stable >= SETTLE_N:
                if verbose:
                    print(f"  자동 조정이 {used}프레임 만에 멈췄습니다.")
                return True, used
        else:
            stable = 0
        prev = m
        if verbose and used % 30 == 0:
            print(f"  수렴 대기 중... {used}프레임")
    if verbose:
        print(f"  ⚠️ {used}프레임 동안 안 멈췄습니다 (조명이 깜빡이는 중일 수 있음)")
    return False, used


def report(cap, label):
    e = cap.get(cv2.CAP_PROP_EXPOSURE)
    w = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
    a = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    print(f"  {label}: 노출 {e:g}  화이트밸런스 {w:g}  (auto_exposure={a:g})")
    return e, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--unlock", action="store_true",
                    help="고정을 풀고 자동으로 되돌린다")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"카메라 {args.camera} 를 열 수 없습니다."); return
    try:
        if args.unlock:
            for v in vd.AUTO_EXPOSURE_VALUES:
                if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v):
                    break
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            save_colors(cam_exposure=None, cam_wb=None)
            print(f"자동노출·자동 화이트밸런스로 되돌렸습니다 → {COLORS_PATH}")
            return

        print("\n조명을 실제 대국 때와 같게 하고, 체스판이 화면에 들어오게 두세요.")
        print("자동노출·자동 화이트밸런스가 수렴할 때까지 기다립니다...\n")
        settle(cap)
        exposure, wb = report(cap, "수렴한 값")

        if exposure == 0 and wb == 0:
            print("\n❌ 카메라가 노출/화이트밸런스 값을 알려주지 않습니다.")
            print("   OpenCV 로는 고정할 수 없는 기종입니다. v4l2-ctl 을 쓰세요:")
            print(f"     v4l2-ctl -d /dev/video{args.camera} --list-ctrls")
            return

        print("\n이 값으로 고정합니다...")
        apply_camera_controls(cap, exposure=exposure, wb_temp=wb)
        for _ in range(15):
            cap.read()
        e2, w2 = report(cap, "고정 후")

        # 정말 고정됐는지 — 손으로 조명을 가려도 안 변해야 진짜 고정이다.
        # 여기서는 최소한 값이 유지되는지만 확인한다.
        ok_e = abs(e2 - exposure) <= max(1.0, abs(exposure) * 0.1)
        ok_w = abs(w2 - wb) <= max(1.0, abs(wb) * 0.1)
        if not (ok_e and ok_w):
            print("\n  ⚠️ 값이 그대로 유지되지 않습니다 — 이 카메라는 OpenCV 로")
            print("     고정이 안 되는 기종일 수 있습니다. v4l2-ctl 을 쓰거나,")
            print("     프로그램을 켜고 3초쯤 기다렸다 읽으세요.")

        save_colors(cam_exposure=exposure, cam_wb=wb)
        print(f"\n[저장 완료] {COLORS_PATH}")
        print("  이제 모든 도구가 같은 노출·색으로 봅니다.")
        print("\n다음 순서:")
        print(f"  python vision/pick_pieces.py --camera {args.camera}   # 색 다시 등록")
        print(f"  python vision/check_board.py --camera {args.camera} --capture-empty")
        print(f"  python vision/check_board.py --camera {args.camera}")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
