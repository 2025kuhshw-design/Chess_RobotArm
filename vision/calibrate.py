"""
카메라 캘리브레이션 도구
마우스 클릭으로 체스판 4개 코너 지정 → calibration.json 저장
실행: python calibrate.py --camera 0
"""

import cv2
import json
import argparse
import numpy as np
import os

# ─────────────────────────────────────────
# 하이퍼파라미터 / 상수
# ─────────────────────────────────────────
PREVIEW_SIZE    = 400      # 탑뷰 변환 미리보기 크기 (px)
CORNER_COLORS   = [(0,0,255),(0,165,255),(0,255,0),(255,0,0)]  # 코너 순서별 색상
CORNER_LABELS   = ["좌상(1)", "우상(2)", "우하(3)", "좌하(4)"]
OUTPUT_PATH     = os.path.join(os.path.dirname(__file__), "calibration.json")

# 보드 방향은 detect.py 한 곳에서만 관리한다 (힌트가 실제 설정과 어긋나지 않도록)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.detect import ROBOT_SIDE, chess_to_grid

# ROBOT_SIDE별 안내 문구와 로봇이 있는 변
SIDE_INFO = {
    "bottom": ("ROBOT THIS SIDE (corners 3-4)", "bottom"),
    "top":    ("ROBOT THIS SIDE (corners 1-2)", "top"),
    "left":   ("ROBOT (corners 1-4)",           "left"),
    "right":  ("ROBOT (corners 2-3)",           "right"),
}


class CameraCalibrator:
    def __init__(self, camera_index: int = 0, click_mode: str = "corners"):
        self.camera_index = camera_index
        self.click_mode   = click_mode   # "corners"=8x8 바깥모서리 / "centers"=모서리칸 중앙
        self.corners      = []           # 클릭한 좌표 리스트
        self.frame        = None
        self.cap          = None
        self._done        = False

    # ─────────────────────────────────────────
    # 클릭점 → 8×8 바깥 모서리 4점
    # ─────────────────────────────────────────
    def effective_corners(self):
        """저장/변환에 쓸 '8×8 영역 바깥 모서리' 4점을 반환.

        click_mode="centers"면 클릭한 4점은 모서리 '칸의 중앙'이다. 중앙끼리는
        7칸 거리인데 8칸으로 나누면 전체가 1/8씩 어긋나므로, 호모그래피로
        칸 단위 좌표계를 세우고 바깥 모서리(0,0)~(8,8)를 역변환해 얻는다.
        """
        if len(self.corners) < 4:
            return None
        if self.click_mode == "corners":
            return [list(map(float, c)) for c in self.corners]

        S = 100.0   # 칸 1개 = 100 단위 (임의 스케일)
        src = np.array(self.corners, dtype=np.float32)
        # 클릭한 중앙 4점 ↔ 칸 좌표 (0.5,0.5) (7.5,0.5) (7.5,7.5) (0.5,7.5)
        dst = np.array([[0.5*S, 0.5*S], [7.5*S, 0.5*S],
                        [7.5*S, 7.5*S], [0.5*S, 7.5*S]], dtype=np.float32)
        M    = cv2.getPerspectiveTransform(src, dst)
        Minv = np.linalg.inv(M)
        outer = np.array([[[0, 0], [8*S, 0], [8*S, 8*S], [0, 8*S]]],
                         dtype=np.float32)
        pts = cv2.perspectiveTransform(outer, Minv)[0]
        return [[float(p[0]), float(p[1])] for p in pts]

    # ─────────────────────────────────────────
    # 마우스 콜백
    # ─────────────────────────────────────────
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corners) < 4:
            self.corners.append([x, y])
            label = CORNER_LABELS[len(self.corners) - 1]
            color = CORNER_COLORS[len(self.corners) - 1]
            print(f"  코너 {len(self.corners)}/4 지정: {label} → ({x}, {y})")

            if len(self.corners) == 4:
                print("  ✅ 4개 코너 지정 완료! 's' 키로 저장, 'r' 키로 초기화")

    # ─────────────────────────────────────────
    # 탑뷰 변환 (퍼스펙티브)
    # ─────────────────────────────────────────
    def _get_top_view(self, frame: np.ndarray) -> np.ndarray | None:
        eff = self.effective_corners()
        if eff is None:
            return None
        src = np.array(eff, dtype=np.float32)
        dst = np.array([
            [0,              0             ],
            [PREVIEW_SIZE-1, 0             ],
            [PREVIEW_SIZE-1, PREVIEW_SIZE-1],
            [0,              PREVIEW_SIZE-1],
        ], dtype=np.float32)
        M   = cv2.getPerspectiveTransform(src, dst)
        top = cv2.warpPerspective(frame, M, (PREVIEW_SIZE, PREVIEW_SIZE))
        return top

    # ─────────────────────────────────────────
    # 캘리브레이션 저장
    # ─────────────────────────────────────────
    def _save(self):
        # 저장되는 건 항상 '8×8 바깥 모서리' (detect.py가 기대하는 형식)
        eff = self.effective_corners()
        data = {"corners": [[round(v, 2) for v in c] for c in eff]}
        with open(OUTPUT_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[저장 완료] {OUTPUT_PATH}")
        if self.click_mode == "centers":
            print(f"  클릭(칸 중앙): {self.corners}")
            print(f"  → 계산된 바깥모서리: {data['corners']}")
        else:
            print(f"  corners: {data['corners']}")

    # ─────────────────────────────────────────
    # 메인 실행
    # ─────────────────────────────────────────
    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 {self.camera_index}를 열 수 없습니다.")

        win_main    = "Chess Board Calibration  (q=종료, s=저장, r=초기화)"
        win_preview = "Top View Preview"
        cv2.namedWindow(win_main)
        cv2.setMouseCallback(win_main, self._mouse_callback)

        print("\n=== 카메라 캘리브레이션 ===")
        if self.click_mode == "centers":
            print("모드: centers — 모서리 '칸의 중앙' 4곳을 클릭")
            print("  (바깥 모서리는 코드가 자동 계산)")
        else:
            print("모드: corners — 8x8 영역의 '바깥 모서리' 4곳을 클릭")
            print("  ※ 칸 중앙이 아니라 칸이 시작되는 맨 바깥 꼭짓점!")
            print("  ※ 중앙이 찍기 편하면: --click centers 옵션 사용")
        print("순서: 1) 좌상  2) 우상  3) 우하  4) 좌하")
        print("키: s=저장  r=초기화  q=종료\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[경고] 프레임 읽기 실패")
                break
            self.frame = frame.copy()

            display = frame.copy()
            for i, (cx, cy) in enumerate(self.corners):
                cv2.circle(display, (cx, cy), 8, CORNER_COLORS[i], -1)
                cv2.putText(display, CORNER_LABELS[i], (cx+10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, CORNER_COLORS[i], 2)

            # centers 모드: 계산된 8x8 바깥 경계를 청록색으로 표시(검증용)
            if self.click_mode == "centers" and len(self.corners) == 4:
                eff = self.effective_corners()
                pts = np.array(eff, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(display, [pts], True, (255, 255, 0), 2)
                cv2.putText(display, "computed 8x8 edge", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            remaining = 4 - len(self.corners)
            status_text = (f"코너 {len(self.corners)}/4 지정됨"
                           if self.corners else "체스판 코너를 클릭하세요")
            cv2.putText(display, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow(win_main, display)

            # 탑뷰 미리보기
            top = self._get_top_view(frame)
            if top is not None:
                # 격자선 오버레이
                cell = PREVIEW_SIZE // 8
                for i in range(1, 8):
                    cv2.line(top, (i*cell, 0), (i*cell, PREVIEW_SIZE), (0,255,0), 1)
                    cv2.line(top, (0, i*cell), (PREVIEW_SIZE, i*cell), (0,255,0), 1)
                # 방향 힌트: detect.py의 ROBOT_SIDE에 맞춰 로봇 쪽 변과 a1을 표시
                MAG = (255, 0, 255)
                label, side = SIDE_INFO[ROBOT_SIDE]
                P = PREVIEW_SIZE
                if side == "bottom":
                    cv2.line(top, (0, P-2), (P, P-2), MAG, 4)
                    cv2.putText(top, label, (6, P-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAG, 2)
                elif side == "top":
                    cv2.line(top, (0, 2), (P, 2), MAG, 4)
                    cv2.putText(top, label, (6, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAG, 2)
                elif side == "left":
                    cv2.line(top, (2, 0), (2, P), MAG, 4)
                    cv2.putText(top, label, (8, P//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAG, 2)
                else:  # right
                    cv2.line(top, (P-2, 0), (P-2, P), MAG, 4)
                    cv2.putText(top, label, (P-150, P//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAG, 2)

                # a1 칸 강조 (ROBOT_SIDE에 따라 위치가 달라짐)
                gx, gy = chess_to_grid(0, 0)
                ax, ay = gx * cell, gy * cell
                cv2.rectangle(top, (ax+1, ay+1), (ax+cell-1, ay+cell-1), MAG, 2)
                cv2.putText(top, "a1", (ax+5, ay+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAG, 2)
                cv2.imshow(win_preview, top)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and len(self.corners) == 4:
                self._save()
            elif key == ord('r'):
                self.corners.clear()
                print("  코너 초기화 완료. 다시 클릭하세요.")

        self.cap.release()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────
# 단독 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="체스판 카메라 캘리브레이션")
    parser.add_argument("--camera", type=int, default=0, help="카메라 인덱스 (기본: 0)")
    parser.add_argument("--click", choices=["corners", "centers"], default="corners",
                        help="corners=8x8 바깥 모서리를 클릭(기본) / "
                             "centers=모서리 '칸의 중앙'을 클릭하면 바깥 모서리를 "
                             "자동 계산 (테두리가 둥글어 모서리가 애매할 때 편함)")
    args = parser.parse_args()

    calibrator = CameraCalibrator(camera_index=args.camera, click_mode=args.click)
    calibrator.run()
