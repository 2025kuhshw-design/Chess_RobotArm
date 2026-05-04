"""
체스판·기물 인식 모듈
탑뷰 변환 → 8×8 셀 분할 → 기물 분류
"""

import cv2
import json
import numpy as np
import os
import time

# ─────────────────────────────────────────
# 하이퍼파라미터 / 상수
# ─────────────────────────────────────────
CALIB_PATH   = os.path.join(os.path.dirname(__file__), "calibration.json")
TOP_SIZE     = 400        # 탑뷰 이미지 크기 (px)
CELL_SIZE_PX = TOP_SIZE // 8   # 한 셀 크기 (px)

# 기물 분류 임계값
WHITE_THRESH = 160     # 이 이상이면 흰 기물
BLACK_THRESH = 80      # 이 이하면 검은 기물
# 기물이 있는지 판단: 셀 중앙 영역의 분산이 이 이상이면 기물 있음
PIECE_VAR_THRESH = 200


class ChessBoardDetector:
    def __init__(self, calibration_path: str = CALIB_PATH, camera_index: int = 0):
        self.camera_index = camera_index
        self._load_calibration(calibration_path)
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 {camera_index}를 열 수 없습니다. "
                               "먼저 python vision/calibrate.py 를 실행하세요.")
        self._prev_board = None
        print(f"[Detector] 카메라 {camera_index} 초기화 완료")

    # ─────────────────────────────────────────
    # 캘리브레이션 로드
    # ─────────────────────────────────────────
    def _load_calibration(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"캘리브레이션 파일 없음: {path}\n"
                "  먼저 python vision/calibrate.py 를 실행하세요."
            )
        with open(path) as f:
            data = json.load(f)
        corners = np.array(data["corners"], dtype=np.float32)
        dst = np.array([
            [0,           0          ],
            [TOP_SIZE-1,  0          ],
            [TOP_SIZE-1,  TOP_SIZE-1 ],
            [0,           TOP_SIZE-1 ],
        ], dtype=np.float32)
        self._M = cv2.getPerspectiveTransform(corners, dst)
        print(f"[Detector] 캘리브레이션 로드: {path}")

    # ─────────────────────────────────────────
    # 탑뷰 변환
    # ─────────────────────────────────────────
    def _get_top_view(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self._M, (TOP_SIZE, TOP_SIZE))

    # ─────────────────────────────────────────
    # 셀 분류
    # ─────────────────────────────────────────
    def _classify_cell(self, cell_img: np.ndarray, is_white_square: bool) -> str:
        """
        셀 중앙 60%를 분석해 기물 유무 및 색상 판단.
        체커보드 패턴 기반 적응적 임계값 적용.
        """
        h, w = cell_img.shape[:2]
        margin = int(h * 0.2)
        center = cell_img[margin:h-margin, margin:w-margin]

        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        var_val  = float(np.var(gray))

        # 분산이 낮으면 빈 칸 (균일한 색)
        if var_val < PIECE_VAR_THRESH:
            return "empty"

        # 밝기로 흰/검 분류
        if mean_val >= WHITE_THRESH:
            return "white"
        elif mean_val <= BLACK_THRESH:
            return "black"
        else:
            # 중간값: 배경 색상 보정 (체커보드 기반)
            ref = 128 if is_white_square else 50
            return "white" if mean_val > ref else "black"

    # ─────────────────────────────────────────
    # 메서드 1: 현재 보드 상태 반환
    # ─────────────────────────────────────────
    def get_board_state(self) -> list:
        """
        현재 프레임 캡처 → 8×8 보드 상태 반환
        board[row][col] ∈ {"empty", "white", "black"}
        """
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("카메라 프레임 읽기 실패")

        top   = self._get_top_view(frame)
        board = []

        for row in range(8):
            board_row = []
            for col in range(8):
                x1 = col * CELL_SIZE_PX
                y1 = row * CELL_SIZE_PX
                x2 = x1 + CELL_SIZE_PX
                y2 = y1 + CELL_SIZE_PX
                cell_img = top[y1:y2, x1:x2]
                is_white_sq = (row + col) % 2 == 0
                state = self._classify_cell(cell_img, is_white_sq)
                board_row.append(state)
            board.append(board_row)

        return board

    # ─────────────────────────────────────────
    # 메서드 2: 사람 이동 감지
    # ─────────────────────────────────────────
    def detect_human_move(self) -> tuple:
        """
        엔터 대기 → 이전/현재 보드 비교 → (from_sq, to_sq) 반환
        from_sq, to_sq = (col, row) 형식
        """
        print("  수를 두고 Enter를 누르세요...", end="", flush=True)
        input()   # 엔터 대기

        current = self.get_board_state()

        if self._prev_board is None:
            self._prev_board = current
            print("  [경고] 이전 상태 없음. 현재 상태를 기준으로 저장했습니다.")
            return None

        disappeared = []   # 기물이 사라진 칸 (from 후보)
        appeared    = []   # 기물이 나타난 칸 (to 후보)

        for row in range(8):
            for col in range(8):
                prev = self._prev_board[row][col]
                curr = current[row][col]
                if prev != "empty" and curr == "empty":
                    disappeared.append((col, row))
                elif prev == "empty" and curr != "empty":
                    appeared.append((col, row))

        self._prev_board = current

        if len(disappeared) == 1 and len(appeared) == 1:
            from_sq = disappeared[0]
            to_sq   = appeared[0]
            print(f"  이동 감지: {from_sq} → {to_sq}")
            return from_sq, to_sq
        elif len(disappeared) == 1 and len(appeared) == 0:
            # 기물 잡기 (상대 기물 위치로 이동)
            from_sq = disappeared[0]
            print(f"  기물 잡기 감지: {from_sq} (도착칸 재확인 필요)")
            return from_sq, from_sq
        else:
            print(f"  [경고] 이동 감지 실패 (사라짐={disappeared}, 나타남={appeared}). 재시도하세요.")
            return None

    # ─────────────────────────────────────────
    # 메서드 3: 디버그 오버레이
    # ─────────────────────────────────────────
    def overlay_debug(self, frame: np.ndarray) -> np.ndarray:
        """격자선·기물 위치·인식 결과 오버레이."""
        top = self._get_top_view(frame)

        try:
            board = self.get_board_state()
        except Exception:
            board = None

        # 격자선
        for i in range(1, 8):
            cv2.line(top, (i*CELL_SIZE_PX, 0), (i*CELL_SIZE_PX, TOP_SIZE), (0,255,0), 1)
            cv2.line(top, (0, i*CELL_SIZE_PX), (TOP_SIZE, i*CELL_SIZE_PX), (0,255,0), 1)

        # 기물 표시
        if board:
            for row in range(8):
                for col in range(8):
                    state = board[row][col]
                    cx = col * CELL_SIZE_PX + CELL_SIZE_PX // 2
                    cy = row * CELL_SIZE_PX + CELL_SIZE_PX // 2
                    if state == "white":
                        cv2.circle(top, (cx, cy), 15, (255,255,255), -1)
                        cv2.circle(top, (cx, cy), 15, (0,0,0), 1)
                    elif state == "black":
                        cv2.circle(top, (cx, cy), 15, (50,50,50), -1)

        return top

    def close(self):
        if self.cap is not None:
            self.cap.release()


# ─────────────────────────────────────────
# 단독 실행: 실시간 디버그 뷰
# ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--calib",  type=str, default=CALIB_PATH)
    args = parser.parse_args()

    try:
        detector = ChessBoardDetector(calibration_path=args.calib, camera_index=args.camera)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    print("실시간 디버그 뷰 실행 중... q=종료")
    while True:
        ret, frame = detector.cap.read()
        if not ret:
            break
        debug = detector.overlay_debug(frame)
        cv2.imshow("Chess Detector Debug", debug)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    cv2.destroyAllWindows()
