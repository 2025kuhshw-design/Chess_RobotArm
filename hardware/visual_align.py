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

# 허용 오차 (칸 단위). 0.0172칸 ≈ 0.5mm.
# ⚠️ 카메라 화소 하나가 탑뷰에서 CELL_SIZE_PX=50px/칸 ≈ 0.58mm/px 이다.
#    0.5mm 는 사실상 화소 하나 수준이라, 카메라 잡음·마커 중심 계산의
#    반올림만으로도 이 밑으로는 안 잡힐 수 있다. 그러면 ALIGN_MAX_ITER
#    를 다 쓰고 못 미친 채로 끝난다(위험하진 않다 — 마지막 오프셋으로
#    진행할 뿐이다). 실측해서 계속 못 미치면 이 값을 다시 올릴 것.
ALIGN_TOL_CELLS = 0.0172
# 최대 반복 횟수. 유격 때문에 완전히 수렴하지 않을 수 있으므로 상한을 둔다.
ALIGN_MAX_ITER = 4
# 보정 이득. 1.0이면 측정 오차만큼 그대로 되돌린다. 진동하면 0.6~0.8로 낮춘다.
ALIGN_GAIN = 0.8
# 측정 오차가 이보다 크면 '마커를 잘못 잡은 것'으로 보고 보정하지 않는다.
# 로봇의 실제 오차는 보통 1칸(29mm) 미만이다. 그보다 훨씬 큰 값이 나오면
# 흡착컵이 아니라 다른 물체(체스판·테이프·조명 반사)를 마커로 오인한 것이다.
# 그 값을 믿고 움직이면 팔이 엉뚱한 곳으로 크게 이동해 위험하다.
ALIGN_MAX_ERR_CELLS = 1.2

# ─────────────────────────────────────────
# 정렬 높이 — 이게 정확도를 좌우한다
# ─────────────────────────────────────────
# ⚠️ 카메라가 판을 정확히 수직으로 내려다보지 않으면, 판 위로 뜬 마커는
#    투영될 때 밀려 보인다(시차). 밀림은 **높이에 비례**한다.
#      흡착컵 4.5cm 높이 → 판 가장자리에서 약 25mm(0.9칸) 밀림
#      흡착컵 0.65cm     → 약 3mm
#    예전에는 안전높이(4cm)에서 정렬하고 그 뒤에 내려갔다. 그래서 정렬은
#    "맞았다"고 하는데 내려가는 순간 한 칸 가까이 어긋났다(실제로 겪음).
#
# 해결: **실제로 일할 높이에서 잰다.** 기물 윗면 바로 위까지 내려가서
#       측정하고, 거기서 마지막 하강만 수직으로 한다.
ALIGN_WORK_LIFT = 0.0205   # 기물 윗면에서 20.5mm 위(전체 높이 25mm) — 여기서 측정한다
                           # ⚠️ 8mm는 흡착컵이 기물을 밀어내고 판 바닥에 닿았다(실측)
# 보정하려고 옆으로 움직일 때 잠깐 올라갈 높이 (기물 위를 안전하게 넘어가려고).
# ⚠️ ALIGN_WORK_LIFT 보다 높아야 한다 — 안 그러면 옆으로 옮기는 동안
#    오히려 측정 높이보다 낮게 내려가 기물에 부딪힌다.
ALIGN_CLEAR_LIFT = 0.030   # 기물 윗면에서 30mm 위 (측정 높이보다 반드시 높게)


def offset_xy(col: int, row: int, dcol: float, drow: float) -> tuple:
    """칸 (col,row)의 좌표에 '칸 단위 오프셋'을 더한 월드 (x, y).

    chess_square_to_xyz 를 기준으로 삼고 거기에 델타만 얹는다.
      x = ... + (7-row)*CELL  →  drow 만큼 늘면 x 는 그만큼 준다
      y = ... + col*CELL      →  dcol 만큼 늘면 y 도 그만큼 는다
    이렇게 해야 실측 보정(board_fit)이 적용된 기준점 위에서 보정이 쌓인다.
    """
    x, y, _ = chess_square_to_xyz(col, row)
    return (x - drow * CELL_SIZE, y + dcol * CELL_SIZE)


def align_over_square(arm, detector, col: int, row: int,
                      lift: float = ALIGN_WORK_LIFT,
                      tol=ALIGN_TOL_CELLS, max_iter=ALIGN_MAX_ITER,
                      gain=ALIGN_GAIN, verbose=True, view_cb=None,
                      suction: bool = False, corrector=None,
                      clear_lift: float = ALIGN_CLEAR_LIFT) -> tuple:
    """목표 칸 위에서 카메라를 보며 흡착컵을 정렬한다.

    동작 (한 번 반복):
        1. 측정 높이(lift)로 **수직 하강**
        2. 마커를 보고 오차를 잰다
        3. 허용치 안이면 끝 — 팔은 이미 그 높이에 있으므로 바로 눌러 내리면 된다
        4. 아니면 clear_lift 로 올라가 옆으로 옮긴 뒤 1번으로

    ⚠️ 왜 이렇게 하나:
       · 측정을 **일할 높이에서** 해야 시차 밀림이 작다(위 상수 설명 참고).
       · 수평 이동은 **기물을 넘을 높이에서** 해야 옆 기물을 밀지 않는다.
       · 마지막 하강은 순수 수직이라, 그 사이 마커가 어떻게 보이든
         흡착컵의 실제 xy 는 안 변한다.

    arm      : RealArm
    detector : ChessBoardDetector (find_marker 사용)
    col,row  : 목표 칸
    lift     : 측정 높이 (기물 윗면 기준, m)
    clear_lift : 옆으로 옮길 때 올라갈 높이 (기물 윗면 기준, m)
    view_cb  : 매 반복마다 호출되는 콜백(화면 갱신용). None이면 생략
    suction  : 기물을 들고 있는 중이면 True. ⚠️ 빠뜨리면 정렬 도중
               흡착이 풀려 기물을 떨어뜨린다.
    corrector: f(x,y,z)->Δq (RL 보정). execute_move 가 쓰는 것과 **같은** 것을
               넘겨야 정렬할 때와 하강할 때의 자세가 어긋나지 않는다.

    반환: (dcol, drow) — 칸 단위 보정 오프셋. 호출한 쪽은 이후 하강에도
          **반드시 이 값을 적용**해야 한다(안 그러면 보정이 통째로 사라진다).
          끝났을 때 팔은 측정 높이에 있고, 그 자리에서 수직으로 내리면 된다.
    """
    import time
    import hardware.arm_controller as ac

    dcol, drow = 0.0, 0.0
    last_err = None

    def goto(dc, dr, z_lift):
        """오프셋 (dc,dr) 위치의 z_lift 높이로 이동."""
        x, y = offset_xy(col, row, dc, dr)
        z = PIECE_Z + z_lift
        q = list(inverse_kinematics(x, y, z))
        if corrector is not None:
            d = corrector(x, y, z)
            if d is not None:
                q = [q[i] + d[i] for i in range(3)]
        arm.move(*q, suction=suction)

    for i in range(max_iter):
        # ── 측정 높이로 하강 ──
        try:
            goto(dcol, drow, lift)
        except ValueError as e:
            if verbose:
                print(f"    [정렬] 측정 높이 도달 불가 — 중단 ({e})")
            return (dcol, drow)
        time.sleep(ac.SETTLE_WAIT)
        if view_cb is not None:
            view_cb()

        mk = detector.find_marker()
        if mk is None:
            if verbose:
                print("    [정렬] 마커를 못 찾음 → 보정 없이 진행")
                print("      · 팔이 마커를 가리고 있지 않은지 확인")
                print("      · 마커 색: python vision/pick_marker.py")
            return (dcol, drow)

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

        if err > ALIGN_MAX_ERR_CELLS:
            if verbose:
                print(f"    [정렬] 오차 {err*CELL_SIZE*1000:.0f}mm 는 너무 큽니다 "
                      f"(한도 {ALIGN_MAX_ERR_CELLS*CELL_SIZE*1000:.0f}mm)")
                print("      → 흡착컵이 아닌 다른 물체를 마커로 잡은 것으로 보입니다.")
                print("      → 보정하지 않고 진행합니다.")
            return (dcol, drow)

        if last_err is not None and err > last_err * 1.2:
            if verbose:
                print("    [정렬] 오차가 커짐 — 중단 (ALIGN_GAIN을 낮춰 보세요)")
            return (dcol, drow)
        last_err = err

        new_dcol = dcol - gain * err_col
        new_drow = drow - gain * err_row

        # ── 기물을 넘을 높이로 올라가서 옆으로 옮긴다 ──
        # (측정 높이에서 바로 옆으로 가면 옆 칸 기물을 밀어버린다)
        try:
            goto(dcol, drow, clear_lift)        # 제자리에서 상승
            goto(new_dcol, new_drow, clear_lift)  # 그 높이에서 수평 이동
        except ValueError as e:
            if verbose:
                print(f"    [정렬] 보정 위치가 도달 불가 — 이전 값 유지 ({e})")
            goto(dcol, drow, lift)
            return (dcol, drow)
        dcol, drow = new_dcol, new_drow
        if view_cb is not None:
            view_cb()

    # 반복을 다 썼으면 마지막 오프셋으로 측정 높이에 내려둔다
    try:
        goto(dcol, drow, lift)
    except ValueError:
        pass
    if verbose:
        print(f"    [정렬] {max_iter}회 반복 후에도 허용치 미달 — "
              f"현재 보정({dcol:+.2f}, {drow:+.2f}칸)으로 진행")
    return (dcol, drow)
