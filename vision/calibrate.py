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


class CameraCalibrator:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.corners      = []          # 클릭한 코너 좌표 리스트
        self.frame        = None
        self.cap          = None
        self._done        = False

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
        if len(self.corners) < 4:
            return None
        src = np.array(self.corners, dtype=np.float32)
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
        data = {"corners": self.corners}
        with open(OUTPUT_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[저장 완료] {OUTPUT_PATH}")
        print(f"  corners: {self.corners}")

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
        print("체스판 코너를 순서대로 클릭하세요:")
        print("  1) 좌상  2) 우상  3) 우하  4) 좌하")
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
    args = parser.parse_args()

    calibrator = CameraCalibrator(camera_index=args.camera)
    calibrator.run()
