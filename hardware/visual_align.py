"""
카메라 피드백 기반 위치 보정 (폐루프).

로봇팔 결합부에 유격이 있으면 오차가 매번 달라진다. 그래서 미리 구한
고정 보정식(board_fit)만으로는 한계가 있다. 이 모듈은 다른 방식을 쓴다:

    1. 목표 칸 위로 이동한다
    2. 카메라로 흡착컵 마커가 '실제로' 어디 있는지 본다
    3. 목표와의 차이만큼 다시 움직인다
    4. 오차가 허용치 안에 들어올 때까지 반복

준비물: 흡착컵 옆(카메라에서 보이는 면)에 눈에 띄는 색 스티커.
       색 범위는 vision/detect.py 의 MARKER_HSV_RANGES 에서 조정한다.
       `python main.py --mode vision` 화면에 "MARK ..."가 뜨면 준비 완료.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.ik_solver import (inverse_kinematics, chess_square_to_xyz,
                             CELL_SIZE, BOARD_ORIGIN_X, BOARD_ORIGIN_Y, PIECE_Z)

# 허용 오차 (칸 단위). 0.15칸 ≈ 4.4mm — 흡착컵이 기물을 잡기에 충분.
ALIGN_TOL_CELLS = 0.15
# 최대 반복 횟수. 유격 때문에 완전히 수렴하지 않을 수 있으므로 상한을 둔다.
ALIGN_MAX_ITER = 4
# 보정 이득. 1.0이면 측정 오차만큼 그대로 되돌린다. 진동하면 0.6~0.8로 낮춘다.
ALIGN_GAIN = 0.8


def colrow_to_xy(col_f: float, row_f: float) -> tuple:
    """연속 체스 좌표 → 월드 (x, y). chess_square_to_xyz의 실수 버전.
    ⚠️ 실측 보정(board_fit)은 적용하지 않는다 — 시각 피드백이 그 역할을 대신한다."""
    x = BOARD_ORIGIN_X + (7 - row_f) * CELL_SIZE + CELL_SIZE / 2
    y = BOARD_ORIGIN_Y + col_f * CELL_SIZE + CELL_SIZE / 2
    return (x, y)


def align_over_square(arm, detector, col: int, row: int, lift: float,
                      tol=ALIGN_TOL_CELLS, max_iter=ALIGN_MAX_ITER,
                      gain=ALIGN_GAIN, verbose=True, view_cb=None,
                      suction: bool = False) -> bool:
    """목표 칸 위에서 카메라를 보며 흡착컵을 정렬한다.

    arm      : RealArm
    detector : ChessBoardDetector (find_marker 사용)
    col,row  : 목표 칸
    lift     : 접근 높이 (m)
    view_cb  : 매 반복마다 호출되는 콜백(화면 갱신용). None이면 생략
    suction  : 기물을 들고 있는 중이면 True. ⚠️ 빠뜨리면 정렬 도중
               흡착이 풀려 기물을 떨어뜨린다.
    반환     : True=허용치 안으로 수렴, False=마커 미검출 또는 미수렴
    """
    # 현재 명령 중인 목표 (연속 좌표). 보정하며 이 값을 조금씩 옮긴다.
    tgt_col, tgt_row = float(col), float(row)
    last_err = None

    for i in range(max_iter):
        if view_cb is not None:
            view_cb()
        mk = detector.find_marker()
        if mk is None:
            if verbose:
                print("    [정렬] 마커를 못 찾음 → 보정 없이 진행")
                print("      · --mode vision 창이 켜져 있으면 닫으세요 "
                      "(카메라를 두 프로그램이 동시에 못 씁니다)")
                print("      · 팔이 마커를 가리고 있지 않은지 확인")
                print("      · 색 범위: vision/detect.py 의 MARKER_HSV_RANGES")
            return False

        # 마커가 있는 곳 - 있어야 할 곳 = 오차 (칸 단위)
        err_col = mk[0] - col
        err_row = mk[1] - row
        err = math.hypot(err_col, err_row)
        if verbose:
            print(f"    [정렬 {i+1}] 오차 {err*CELL_SIZE*1000:5.1f}mm "
                  f"(파일 {err_col:+.2f}칸, 랭크 {err_row:+.2f}칸)")

        if err <= tol:
            if verbose:
                print(f"    [정렬] 허용치({tol*CELL_SIZE*1000:.0f}mm) 안 — 완료")
            return True

        # 발산 감지: 오차가 오히려 커지면 중단 (이득이 너무 큼)
        if last_err is not None and err > last_err * 1.2:
            if verbose:
                print("    [정렬] 오차가 커짐 — 중단 (ALIGN_GAIN을 낮춰 보세요)")
            return False
        last_err = err

        # 오차만큼 목표를 반대로 옮긴다
        tgt_col -= gain * err_col
        tgt_row -= gain * err_row
        x, y = colrow_to_xy(tgt_col, tgt_row)
        try:
            arm.move(*inverse_kinematics(x, y, PIECE_Z + lift),
                     suction=suction)
        except ValueError as e:
            if verbose:
                print(f"    [정렬] 보정 위치가 도달 불가 — 중단 ({e})")
            return False

        import time
        import hardware.arm_controller as ac
        time.sleep(ac.SETTLE_WAIT)      # 흔들림이 멎은 뒤 다시 측정
        if view_cb is not None:
            view_cb()

    if verbose:
        print(f"    [정렬] {max_iter}회 반복 후에도 허용치 미달 — 그대로 진행")
    return False
