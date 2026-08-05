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

# 디버그 오버레이 색상 (BGR)
LABEL_COLOR  = (0, 215, 255)    # 칸 좌표 라벨: 노랑
ORIGIN_COLOR = (255, 0, 255)    # a1(로봇 원점) 강조: 마젠타

# ─────────────────────────────────────────
# 보드 방향: 탑뷰에서 로봇팔이 어느 쪽에 있는가
# ─────────────────────────────────────────
# 캘리브레이션 클릭 순서는 1=좌상 2=우상 3=우하 4=좌하 이고,
# 탑뷰에서 그 순서대로 좌상/우상/우하/좌하에 배치된다.
#   "bottom" → 로봇이 아래쪽(코너 3·4 변)
#   "top"    → 로봇이 위쪽(코너 1·2 변)
#   "left"   → 로봇이 왼쪽(코너 1·4 변)   ← 현재 배치
#   "right"  → 로봇이 오른쪽(코너 2·3 변)
#
# 로봇 좌표계(utils/ik_solver.py): 랭크 1→8 은 사람 쪽에서 로봇 쪽으로,
# 파일 a→h 는 로봇 기준 왼쪽(+y). 위에서 내려다본 화면이므로 이 두 축의
# 화면상 방향이 ROBOT_SIDE 에 따라 결정된다.
# (사람은 로봇 맞은편에 앉는다고 가정 — 랭크1이 사람 쪽, 랭크8이 로봇 쪽)
ROBOT_SIDE = "left"

# ─────────────────────────────────────────
# 흡착컵 마커 (시각 피드백 보정용)
# ─────────────────────────────────────────
# 흡착컵 옆(카메라에서 보이는 면)에 눈에 띄는 색 스티커를 붙이고, 그 색을
# 여기에 등록한다. 카메라가 이 마커를 보고 "지금 실제로 어디에 가 있는지"를
# 알아내 오차만큼 다시 움직인다(폐루프 보정).
#   기계적 유격 탓에 오차가 매번 달라지므로 고정 보정식으로는 한계가 있다.
# HSV 범위. OpenCV의 H는 0~179.
# 기본값은 **빨강** — 흡착기 마운트에 감긴 빨간 점퍼선을 그대로 마커로 쓴다.
# 빨강은 H가 0에서 끊기므로 구간을 둘로 나눠야 한다.
MARKER_HSV_RANGES = [
    ((  0, 110,  80), ( 10, 255, 255)),   # 빨강 (H 낮은 쪽)
    ((170, 110,  80), (179, 255, 255)),   # 빨강 (H 높은 쪽)
]
MARKER_MIN_AREA = 40      # 이보다 작은 덩어리는 잡음으로 무시 (탑뷰 픽셀)
WARMUP_FRAMES   = 8       # 카메라 열자마자 버릴 프레임 수 (자동노출 안정화)
MARKER_RETRY    = 5       # 마커를 못 찾았을 때 새 프레임으로 재시도할 횟수

# 마커가 흡착컵 중심 바로 위에 있지 않을 때의 보정 (칸 단위).
# 예: 마커가 흡착컵보다 파일 방향으로 +0.3칸 치우쳐 보이면 (0.3, 0.0).
# test_square의 'markcal <칸>' 명령으로 자동 측정할 수 있다.
MARKER_OFFSET_COL = 0.0
MARKER_OFFSET_ROW = 0.0


def chess_to_grid(col: int, row: int) -> tuple:
    """체스 좌표(col=파일 0~7, row=랭크 0~7) → 탑뷰 격자 인덱스 (gx, gy).
    gx=왼쪽부터 0~7, gy=위부터 0~7."""
    if ROBOT_SIDE == "left":
        # 로봇 왼쪽 → 랭크는 왼쪽(로봇쪽)으로, 파일은 위로
        return (7 - row, 7 - col)
    if ROBOT_SIDE == "right":
        # 로봇 오른쪽 → 랭크는 오른쪽으로, 파일은 아래로
        return (row, col)
    if ROBOT_SIDE == "top":
        # 로봇 위 → 랭크는 위로, 파일은 오른쪽으로
        return (col, 7 - row)
    if ROBOT_SIDE == "bottom":
        # 로봇 아래 → 랭크는 아래로, 파일은 왼쪽으로
        return (7 - col, row)
    raise ValueError(f"ROBOT_SIDE 값이 잘못됨: {ROBOT_SIDE}")


def grid_uv_to_colrow(u: float, v: float) -> tuple:
    """탑뷰의 연속 격자 좌표 (u, v) → 연속 체스 좌표 (col, row).

    u,v 는 칸 단위(0~8). 격자 칸 (gx,gy)의 중심이 (gx+0.5, gy+0.5).
    chess_to_grid의 역변환을 실수 영역으로 확장한 것 — 흡착컵이 칸 중심에서
    얼마나 벗어났는지를 칸 단위로 재는 데 쓴다.
    """
    gx, gy = u - 0.5, v - 0.5
    if ROBOT_SIDE == "left":
        return (7 - gy, 7 - gx)
    if ROBOT_SIDE == "right":
        return (gy, gx)
    if ROBOT_SIDE == "top":
        return (gx, 7 - gy)
    if ROBOT_SIDE == "bottom":
        return (7 - gx, gy)
    raise ValueError(f"ROBOT_SIDE 값이 잘못됨: {ROBOT_SIDE}")


class ChessBoardDetector:
    def __init__(self, calibration_path: str = CALIB_PATH, camera_index: int = 0):
        self.camera_index = camera_index
        self._load_calibration(calibration_path)
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 {camera_index}를 열 수 없습니다. "
                               "먼저 python vision/calibrate.py 를 실행하세요.")
        self._prev_board = None
        # 워밍업: 카메라를 막 열면 자동노출·화이트밸런스가 잡히기 전이라
        # 첫 몇 프레임이 어둡거나 색이 틀어져 마커를 놓친다.
        for _ in range(WARMUP_FRAMES):
            self.cap.read()
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
        현재 프레임 캡처 → 8×8 보드 상태 반환.
        board[row][col] ∈ {"empty", "white", "black"}
        ⚠️ 인덱스는 **체스/로봇 좌표**다 (col=파일 a~h, row=랭크 1~8).
           화면 격자 위치는 chess_to_grid()로 변환해 읽는다.
        """
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("카메라 프레임 읽기 실패")

        top   = self._get_top_view(frame)
        board = []

        for row in range(8):
            board_row = []
            for col in range(8):
                gx, gy = chess_to_grid(col, row)
                x1 = gx * CELL_SIZE_PX
                y1 = gy * CELL_SIZE_PX
                cell_img = top[y1:y1 + CELL_SIZE_PX, x1:x1 + CELL_SIZE_PX]
                is_white_sq = (gx + gy) % 2 == 0
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
    def overlay_debug(self, frame: np.ndarray, show_labels: bool = True) -> np.ndarray:
        """
        격자선·기물 위치·인식 결과 오버레이.
        show_labels=True면 각 칸에 체스 좌표(a1~h8)와 (col,row)를 찍고
        a1(=로봇 원점) 칸을 마젠타로 강조한다. 카메라 회전/반전 검증용.
        """
        top = self._get_top_view(frame)

        try:
            board = self.get_board_state()
        except Exception:
            board = None

        # 격자선
        for i in range(1, 8):
            cv2.line(top, (i*CELL_SIZE_PX, 0), (i*CELL_SIZE_PX, TOP_SIZE), (0,255,0), 1)
            cv2.line(top, (0, i*CELL_SIZE_PX), (TOP_SIZE, i*CELL_SIZE_PX), (0,255,0), 1)

        # 기물 표시 (board는 체스 좌표 → 화면 위치로 변환)
        if board:
            for row in range(8):
                for col in range(8):
                    state = board[row][col]
                    gx, gy = chess_to_grid(col, row)
                    cx = gx * CELL_SIZE_PX + CELL_SIZE_PX // 2
                    cy = gy * CELL_SIZE_PX + CELL_SIZE_PX // 2
                    if state == "white":
                        cv2.circle(top, (cx, cy), 15, (255,255,255), -1)
                        cv2.circle(top, (cx, cy), 15, (0,0,0), 1)
                    elif state == "black":
                        cv2.circle(top, (cx, cy), 15, (50,50,50), -1)

        # 칸 좌표 라벨 + 원점 강조 (회전/반전 검증용)
        if show_labels:
            for row in range(8):
                for col in range(8):
                    gx, gy = chess_to_grid(col, row)
                    x1 = gx * CELL_SIZE_PX
                    y1 = gy * CELL_SIZE_PX
                    # 체스 표기: col→파일(a-h), row→랭크(1-8)
                    notation = f"{chr(ord('a') + col)}{row + 1}"

                    # a1(col0,row0)=로봇 원점 칸 강조
                    if col == 0 and row == 0:
                        cv2.rectangle(top, (x1+1, y1+1),
                                      (x1+CELL_SIZE_PX-1, y1+CELL_SIZE_PX-1),
                                      ORIGIN_COLOR, 2)
                        cv2.putText(top, "a1", (x1+3, y1+18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORIGIN_COLOR, 2)
                    else:
                        cv2.putText(top, notation, (x1+3, y1+14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, LABEL_COLOR, 1)
                    # 작게 (col,row)도 표시
                    cv2.putText(top, f"{col},{row}", (x1+3, y1+CELL_SIZE_PX-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,200,200), 1)

            # 흡착컵 마커 표시 (색 범위 튜닝용)
            mk = self.find_marker(frame)
            if mk is not None:
                gx, gy = chess_to_grid(int(round(mk[0])), int(round(mk[1])))
                # 연속 좌표를 다시 픽셀로 — 표시용이므로 근사로 충분
                cv2.putText(top, f"MARK {chr(97+int(round(mk[0])))}{int(round(mk[1]))+1}",
                            (5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(top, (gx*CELL_SIZE_PX + CELL_SIZE_PX//2,
                                 gy*CELL_SIZE_PX + CELL_SIZE_PX//2), 8, (0, 0, 255), 2)
            else:
                cv2.putText(top, "MARK: not found", (5, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 축 방향 안내 — 로봇이 있는 변을 표시
            side_txt = {"bottom": "ROBOT THIS SIDE (v)", "top": "ROBOT THIS SIDE (^)",
                        "left": "ROBOT <", "right": "> ROBOT"}[ROBOT_SIDE]
            pos = {"bottom": (5, TOP_SIZE-6), "top": (5, 16),
                   "left": (5, TOP_SIZE//2), "right": (TOP_SIZE-95, TOP_SIZE//2)}[ROBOT_SIDE]
            cv2.putText(top, side_txt, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORIGIN_COLOR, 2)

        return top

    # ─────────────────────────────────────────
    # 메서드 4: 흡착컵 마커 위치 (시각 피드백 보정용)
    # ─────────────────────────────────────────
    def find_marker(self, frame=None) -> tuple:
        """흡착컵에 붙인 색 마커의 위치를 찾아 연속 체스 좌표 (col, row)로 반환.

        못 찾으면 None. 마커가 없으면 색 범위(MARKER_HSV_LO/HI)를 조정하거나
        --mode vision 디버그 뷰에서 마커가 잡히는지 먼저 확인할 것.

        기계적 유격 때문에 로봇의 오차는 매번 달라진다. 그래서 고정된 보정식
        대신, 실제로 어디에 가 있는지 카메라로 보고 그만큼 움직이는 방식이
        필요하다. 이 함수가 그 '보는' 부분이다.
        """
        if frame is None:
            # 한 프레임만 보고 판단하면 노출 변동·모션 블러로 놓칠 수 있다.
            # 프레임을 새로 받아가며 몇 번 재시도한다.
            for attempt in range(MARKER_RETRY):
                ret, f = self.cap.read()
                if not ret:
                    continue
                r = self._find_marker_in(f)
                if r is not None:
                    return r
            return None
        return self._find_marker_in(frame)

    def _find_marker_in(self, frame):
        """단일 프레임에서 마커 찾기 (재시도 없음)."""
        top = self._get_top_view(frame)
        hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)

        mask = None
        for lo, hi in MARKER_HSV_RANGES:
            m = cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= MARKER_MIN_AREA]
        if not cnts:
            return None
        M = cv2.moments(max(cnts, key=cv2.contourArea))
        if M["m00"] == 0:
            return None
        px, py = M["m10"] / M["m00"], M["m01"] / M["m00"]
        c, r = grid_uv_to_colrow(px / CELL_SIZE_PX, py / CELL_SIZE_PX)
        # 마커가 흡착컵 바로 위가 아니면 그 차이를 빼서 흡착컵 위치로 환산
        return (c - MARKER_OFFSET_COL, r - MARKER_OFFSET_ROW)

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
