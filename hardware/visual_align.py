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
                             CELL_SIZE, PIECE_Z)

# 허용 오차 (칸 단위). 0.0687칸 ≈ 2.0mm.
ALIGN_TOL_CELLS = 0.0687
# 최대 반복 횟수. 유격 때문에 완전히 수렴하지 않을 수 있으므로 상한을 둔다.
ALIGN_MAX_ITER = 4
# 보정 이득. 1.0이면 측정 오차만큼 그대로 되돌린다. 진동하면 0.6~0.8로 낮춘다.
ALIGN_GAIN = 0.8
# 측정 오차가 이보다 크면 '마커를 잘못 잡은 것'으로 보고 보정하지 않는다.
# 로봇의 실제 오차는 보통 1칸(29mm) 미만이다. 그보다 훨씬 큰 값이 나오면
# 흡착컵이 아니라 다른 물체(체스판·테이프·조명 반사)를 마커로 오인한 것이다.
# 그 값을 믿고 움직이면 팔이 엉뚱한 곳으로 크게 이동해 위험하다.
ALIGN_MAX_ERR_CELLS = 1.2


def offset_xy(col: int, row: int, dcol: float, drow: float) -> tuple:
    """칸 (col,row)의 좌표에 '칸 단위 오프셋'을 더한 월드 (x, y).

    chess_square_to_xyz 를 기준으로 삼고 거기에 델타만 얹는다.
      x = ... + (7-row)*CELL  →  drow 만큼 늘면 x 는 그만큼 준다
      y = ... + col*CELL      →  dcol 만큼 늘면 y 도 그만큼 는다
    이렇게 해야 실측 보정(board_fit)이 적용된 기준점 위에서 보정이 쌓인다.
    """
    x, y, _ = chess_square_to_xyz(col, row)
    return (x - drow * CELL_SIZE, y + dcol * CELL_SIZE)


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

    반환: (dcol, drow) — 칸 단위 보정 오프셋.
      ⚠️ 호출한 쪽은 **이 값을 이후 하강에도 반드시 적용해야 한다.**
         예전에는 bool만 돌려줬고, 하강할 때 IK를 원래 칸 좌표로 다시
         계산했다. 그래서 정렬로 옮겨놓은 위치가 하강 직전에 도로
         원위치로 돌아가, 카메라 보정이 사실상 아무 효과가 없었다.
      정렬을 못 했으면 (0.0, 0.0) — 보정 없이 원래 칸으로 가면 된다.
    """
    # 목표 칸에서 얼마나 옮겼는지 (칸 단위). 이 값을 호출부에 돌려준다.
    dcol, drow = 0.0, 0.0
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
                print("      · 마커 색을 다시 고르려면: python vision/pick_marker.py")
            return (dcol, drow)

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
            return (dcol, drow)

        # 측정값이 비현실적으로 크면 오인식 — 움직이지 않고 중단
        if err > ALIGN_MAX_ERR_CELLS:
            if verbose:
                print(f"    [정렬] 오차 {err*CELL_SIZE*1000:.0f}mm 는 너무 큽니다 "
                      f"(한도 {ALIGN_MAX_ERR_CELLS*CELL_SIZE*1000:.0f}mm)")
                print("      → 흡착컵이 아닌 다른 물체를 마커로 잡은 것으로 보입니다.")
                print("      → 보정하지 않고 진행합니다. 'show'로 무엇이 잡히는지"
                      " 확인하고 마커 색/오프셋을 점검하세요.")
            return (0.0, 0.0)      # 믿을 수 없는 측정 — 원래 칸으로

        # 발산 감지: 오차가 오히려 커지면 중단 (이득이 너무 큼)
        if last_err is not None and err > last_err * 1.2:
            if verbose:
                print("    [정렬] 오차가 커짐 — 중단 (ALIGN_GAIN을 낮춰 보세요)")
            return (dcol, drow)
        last_err = err

        # 오차만큼 목표를 반대로 옮긴다
        new_dcol = dcol - gain * err_col
        new_drow = drow - gain * err_row
        x, y = offset_xy(col, row, new_dcol, new_drow)
        try:
            arm.move(*inverse_kinematics(x, y, PIECE_Z + lift),
                     suction=suction)
        except ValueError as e:
            if verbose:
                print(f"    [정렬] 보정 위치가 도달 불가 — 이전 값 유지 ({e})")
            return (dcol, drow)
        dcol, drow = new_dcol, new_drow

        import time
        import hardware.arm_controller as ac
        time.sleep(ac.SETTLE_WAIT)      # 흔들림이 멎은 뒤 다시 측정
        if view_cb is not None:
            view_cb()

    if verbose:
        print(f"    [정렬] {max_iter}회 반복 후에도 허용치 미달 — "
              f"현재 보정({dcol:+.2f}, {drow:+.2f}칸)으로 진행")
    return (dcol, drow)
