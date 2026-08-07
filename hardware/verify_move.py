"""
수를 둔 뒤 카메라로 확인하고, 실패했으면 다시 시도한다.

왜 필요한가:
  흡착이 빗나가면 기물을 못 집는다. 그런데 팔은 그걸 모르고 나머지 동작을
  끝까지 한다 — 빈 흡착컵을 목표 칸에 내리고, 흡착을 풀고, Z자로 돌아온다.
  그러면 엔진이 아는 판과 실제 판이 어긋난 채로 게임이 계속된다.
  → 팔이 판에서 물러난 뒤에 카메라로 실제 판을 보고, 기대와 다르면 다시 한다.

두 가지를 본다:
  ① 수가 실제로 반영됐는가 (출발 칸이 비었고 도착 칸이 찼는가)
  ② 놓인 기물이 칸 중심에서 얼마나 벗어났는가
     ②는 고칠 수 있는 정보이기도 하다 — 정렬은 측정 높이(기물 위 20.5mm)
     에서 끝나고, 거기서 누르는 높이까지 내려가는 동안 팔이 밀린다.
     그 밀림은 계산으로 못 잡는다(관절보간 탓이 아님을 시뮬레이션으로 확인:
     모형상 하강의 수평 이탈은 최대 1.1mm). 중력 처짐·유격·고무 변형이다.
     → 놓인 위치를 재서 그만큼을 다음 하강에 미리 빼준다(DESCENT_BIAS).
"""

import math

import hardware.arm_controller as ac

# 재시도 횟수. 같은 이유로 계속 실패하면 사람이 손대는 게 빠르다.
RETRY_MAX = 2

# 놓인 기물이 칸 중심에서 이보다 벗어나면 '치우쳤다'고 보고 보정에 반영한다.
# ⚠️ 이 값이 곧 수렴이 멈추는 지점이다. 3mm 로 뒀더니 잔여 오차 3mm 에서
#    딱 멈춰 버렸다(이득 0.5 × 밀림 6mm = 3mm 에서 불감대에 걸림).
#    측정 잡음(~0.5mm)보다는 크고, 남기고 싶은 오차보다는 작게 잡는다.
PLACE_TOL_MM = 1.2

# 밀림 추정치를 한 번에 얼마나 갱신할지. 1.0이면 마지막 한 번을 그대로 믿는다.
# 유격 때문에 매번 조금씩 다르므로 절반만 반영해 흔들림을 줄인다.
BIAS_GAIN = 0.5
# 추정치 상한 (칸). 0.30칸 ≈ 8.7mm. 측정이 잘못돼도 팔이 크게 어긋나지 않게.
BIAS_MAX_CELLS = 0.30

CELL_MM = 29.125


def reset_bias():
    ac.DESCENT_BIAS[0] = ac.DESCENT_BIAS[1] = 0.0


def update_bias(err_col: float, err_row: float, verbose: bool = True):
    """놓인 기물의 오차만큼 하강 밀림 추정치를 갱신한다.

    err_* = 측정한 기물 위치 - 목표 칸 (칸 단위).
    이 값이 곧 '하강하면서 밀린 양'이므로, 다음엔 그만큼 반대로 미리 뺀다.
    """
    for i, e in enumerate((err_col, err_row)):
        v = ac.DESCENT_BIAS[i] + BIAS_GAIN * e
        ac.DESCENT_BIAS[i] = max(-BIAS_MAX_CELLS, min(BIAS_MAX_CELLS, v))
    if verbose:
        print(f"  [하강보정] 밀림 추정 갱신 → 파일 "
              f"{ac.DESCENT_BIAS[0]*CELL_MM:+.1f}mm, "
              f"랭크 {ac.DESCENT_BIAS[1]*CELL_MM:+.1f}mm")


def _occupied(state, sq):
    col, row = sq
    return state[row][col] != "empty"


def check_move(detector, before, after, from_sq, to_sq, verbose: bool = True):
    """수가 실제로 반영됐는지 카메라로 확인한다.

    before / after : 수를 두기 **전**과 **후**의 기대 배치 (8x8)
    반환: (status, message)
        "ok"     — 기대대로 됐다
        "retry"  — 아무 일도 안 일어났다(집기 실패). 그대로 다시 시도하면 된다
        "manual" — 어중간하게 틀어졌다. 자동으로 못 고친다
        "unknown"— 판독이 안정되지 않아 판단 불가 (그냥 진행)
    """
    obs = detector.get_stable_board_state(expect=after)
    if obs is None:
        return "unknown", "판독이 안정되지 않아 확인하지 못했습니다"

    from_ok = not _occupied(obs, from_sq)          # 출발 칸이 비었어야 한다
    to_ok = _occupied(obs, to_sq) == _occupied(after, to_sq)
    if from_ok and to_ok:
        return "ok", ""

    # 아무 일도 안 일어난 경우 = 출발 칸에 그대로 있고, 도착 칸도 그대로.
    # 이때만 안전하게 재시도할 수 있다. 어중간하면 사람이 봐야 한다.
    nothing_happened = (_occupied(obs, from_sq) == _occupied(before, from_sq)
                        and _occupied(obs, to_sq) == _occupied(before, to_sq))
    f = f"{chr(97+from_sq[0])}{from_sq[1]+1}"
    t = f"{chr(97+to_sq[0])}{to_sq[1]+1}"
    if nothing_happened:
        return "retry", f"{f} 기물이 그대로 있습니다 — 집지 못한 것 같습니다"
    if not from_ok:
        return "manual", f"{f} 가 비어 있어야 하는데 아직 차 있습니다"
    return "manual", f"{t} 에 기물이 있어야 하는데 없습니다 (떨어뜨렸을 수 있음)"


def check_placement(detector, to_sq, verbose: bool = True):
    """놓인 기물이 칸 중심에서 얼마나 벗어났는지 재고, 보정에 반영한다.

    ⚠️ 팔이 판에서 물러난 뒤에 불러야 한다. 팔이 그 칸 위에 있으면
       흡착컵을 기물로 착각한다.
    반환: 벗어난 거리(mm) 또는 None(기물을 못 찾음)
    """
    detector.clear_piece_snapshot()          # 낡은 기록을 쓰지 않게
    c = detector.piece_center(to_sq[0], to_sq[1], use_snapshot=False)
    if c is None:
        return None
    ec, er = c[0] - to_sq[0], c[1] - to_sq[1]
    d = math.hypot(ec, er) * CELL_MM
    if verbose:
        t = f"{chr(97+to_sq[0])}{to_sq[1]+1}"
        print(f"  [놓임확인] {t} 기물이 칸 중심에서 {d:.1f}mm 벗어남 "
              f"(파일 {ec*CELL_MM:+.1f}, 랭크 {er*CELL_MM:+.1f})")
    if d > PLACE_TOL_MM:
        update_bias(ec, er, verbose=verbose)
    return d
