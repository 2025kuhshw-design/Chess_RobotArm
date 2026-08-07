"""
체스판·기물 인식 모듈
탑뷰 변환 → 8×8 셀 분할 → 기물 분류
"""

import cv2
import json
import math
import numpy as np
import os
import time

# ─────────────────────────────────────────
# 하이퍼파라미터 / 상수
# ─────────────────────────────────────────
CALIB_PATH   = os.path.join(os.path.dirname(__file__), "calibration.json")
CELL_SIZE_PX = 50                      # 한 칸 크기 (px)
BOARD_PX     = 8 * CELL_SIZE_PX        # 체스판 8x8 영역 (px)

# ─────────────────────────────────────────
# 판 바깥 여유 — 없으면 가장자리 칸에서 마커를 못 본다
# ─────────────────────────────────────────
# ⚠️ 예전에는 탑뷰를 **체스판 딱 그 영역만** 잘라냈다. 그런데 흡착컵 마커는
#    판보다 위에 떠 있어 시차로 바깥쪽으로 밀려 보인다. 그래서 a·h 파일이나
#    8랭크 같은 가장자리 칸에 가면 마커가 잘린 영역 밖으로 나가 **아예 안
#    보이고, 그러면 보정 자체가 안 됐다**(실제로 겪음).
# 판 둘레로 이만큼 더 펴서 마커가 화면 안에 남게 한다. 격자 번호와 칸 좌표는
# 그대로다 — 원점만 MARGIN_PX 만큼 밀린다.
TOP_MARGIN_CELLS = 1.5
MARGIN_PX    = int(round(TOP_MARGIN_CELLS * CELL_SIZE_PX))
TOP_SIZE     = BOARD_PX + 2 * MARGIN_PX    # 탑뷰 전체 크기 (px)


def cell_px(gx: float, gy: float) -> tuple:
    """격자 좌표 (gx,gy) 의 좌상단 픽셀. 여유(MARGIN_PX)를 더해 준다."""
    return (int(round(MARGIN_PX + gx * CELL_SIZE_PX)),
            int(round(MARGIN_PX + gy * CELL_SIZE_PX)))


def px_to_grid_uv(px: float, py: float) -> tuple:
    """탑뷰 픽셀 → 격자 좌표(칸 단위). cell_px 의 역변환."""
    return ((px - MARGIN_PX) / CELL_SIZE_PX,
            (py - MARGIN_PX) / CELL_SIZE_PX)

# 기물 분류 임계값
# 빈 칸일 때의 밝기와 이만큼(0~255) 넘게 차이 나면 기물이 있다고 본다.
# ⚠️ 예전에는 '셀 중앙의 분산'으로 판단했는데, 기물이 셀을 덮으면 오히려
#    균일해져 분산이 0에 가까워진다. 그래서 32개 기물이 전부 빈 칸으로
#    읽혔다. 지금은 '빈 칸 대비 얼마나 밝은가/어두운가'로 판단한다.
# 낮추면 기물을 더 잘 잡지만 그림자·얼룩도 기물로 오인한다.
# 'python vision/check_board.py' 로 실제 차이값을 보고 정할 것.
OCC_DIFF_THRESH = 18

# ─────────────────────────────────────────
# 빈 판 기준 영상 (권장 방식)
# ─────────────────────────────────────────
# ⚠️ 밝기만으로는 **검은 기물과 어두운 칸을 구분할 수 없다.** 둘 다 어둡기
#    때문이다. 실제로 검은 기물이 어두운 칸에서만 통째로 안 잡혔고,
#    임계값을 낮추면 빈 칸이 기물로 잡히기 시작했다(맞바꿈이라 해결 불가).
#
# 그래서 '기물을 다 치운 판'을 한 번 찍어두고, 그것과 **픽셀 단위로 비교**한다.
# 검은 기물이 어두운 칸에 있어도 나뭇결·테두리·그림자가 달라지므로 잡힌다.
# 기물이 작아도 된다 — 평균이 아니라 '달라진 픽셀의 비율'로 판단하기 때문.
EMPTY_REF_PATH = os.path.join(os.path.dirname(__file__), "empty_ref.npz")
# 픽셀 하나가 '달라졌다'고 볼 밝기 차.
# 기준 영상과 빼기 때문에 나뭇결·얼룩은 상쇄되고 카메라 잡음(±3~5)만 남는다.
# 그래서 밝기 방식보다 훨씬 낮게 잡을 수 있다. 낮추면 민감(그림자도 잡힘).
PIXEL_DIFF_THRESH = 12
# 셀 중앙에서 이 비율 이상 달라지면 기물이 있다고 본다.
# 기물이 작으면 낮추고, 빈 칸이 자꾸 잡히면 올린다.
OCC_AREA_FRAC = 0.18

# ─────────────────────────────────────────
# 기물 윗면 색으로 판정 (가장 확실한 방법)
# ─────────────────────────────────────────
# 카메라는 기물의 **윗면만** 본다. 그러니 윗면을 체스판에 없는 색으로 칠하면
# 판 색깔과 상관없이 확실하게 구분된다.
#   · 검은 기물 윗면에 검은 테이프 → 어두운 칸과 같은 색 → 영원히 구분 불가
#   · 검은 기물 윗면에 빨간 테이프 → 판에 빨강이 없으므로 100% 구분
#
# 양쪽 기물 윗면에 서로 다른 색 스티커/테이프를 붙이고 아래를 채운 뒤
# PIECE_COLOR_MODE = True 로 켠다. 색은 pick_pieces.py 로 실측한다.
#   python vision/pick_pieces.py --camera 1
PIECE_COLOR_MODE = False
PIECE_HSV_WHITE = []      # 흰쪽(사람) 기물 윗면 색 — [(lo, hi), ...]
PIECE_HSV_BLACK = []      # 검은쪽(로봇) 기물 윗면 색
# 셀 중앙에서 그 색이 이 비율 이상이면 기물이 있다고 본다.
# 기물이 작으면 낮춘다. 0.06 ≈ 셀 중앙(30x30px)의 54px.
PIECE_COLOR_MIN_FRAC = 0.08

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
#   "left"   → 로봇이 왼쪽(코너 1·4 변)
#   "right"  → 로봇이 오른쪽(코너 2·3 변)   ← 현재 배치
#
# 로봇 좌표계(utils/ik_solver.py): 랭크 1→8 은 사람 쪽에서 로봇 쪽으로,
# 파일 a→h 는 로봇 기준 왼쪽(+y). 위에서 내려다본 화면이므로 이 두 축의
# 화면상 방향이 ROBOT_SIDE 에 따라 결정된다.
# (사람은 로봇 맞은편에 앉는다고 가정 — 랭크1이 사람 쪽, 랭크8이 로봇 쪽)
ROBOT_SIDE = "right"

# ─────────────────────────────────────────
# 흡착컵 마커 (시각 피드백 보정용)
# ─────────────────────────────────────────
# 흡착컵 옆(카메라에서 보이는 면)에 눈에 띄는 색 스티커를 붙이고, 그 색을
# 여기에 등록한다. 카메라가 이 마커를 보고 "지금 실제로 어디에 가 있는지"를
# 알아내 오차만큼 다시 움직인다(폐루프 보정).
#   기계적 유격 탓에 오차가 매번 달라지므로 고정 보정식으로는 한계가 있다.
# HSV 범위. OpenCV의 H는 0~179.
# 현재 마커: **빨강** (점퍼선 색)
# ⚠️ 노랑은 체스판의 카페라떼색 칸과 색상(H)이 겹쳐 마커 인식이 안 됐다
#    (카페라떼 칸 H≈18, 노랑 마커 H≈20~38 — 사실상 같은 대역).
#    빨강은 H가 0 부근이라 카페라떼(H≈18)·검정 칸과 겹치지 않는다.
# 빨강은 OpenCV HSV에서 H=0 근처를 감싸므로 두 구간(0쪽, 179쪽)이 필요하다.
MARKER_HSV_RANGES = [
    ((  0, 110,  80), ( 10, 255, 255)),   # 빨강 (0쪽)
    ((170, 110,  80), (179, 255, 255)),   # 빨강 (179쪽)
]
# 참고 — 다른 색으로 바꿀 때 (기물 색과 겹치지 않는 것으로 고를 것):
#   노랑 : ((20,130,120),(38,255,255))   ⚠️ 카페라떼 칸과 겹침, 비추천
#   초록 : ((35,80,60),(85,255,255))
#   파랑 : ((100,120,60),(130,255,255))  ⚠️ 검은쪽 기물과 같은 색이면 겹침
#   분홍 : ((140,90,90),(175,255,255))
MARKER_MIN_AREA = 40      # 이보다 작은 덩어리는 잡음으로 무시 (탑뷰 픽셀)
# 한 칸이 50x50=2500px 이므로, 마커가 칸의 절반을 넘으면 보드나 조명 반사를
# 잘못 잡은 것으로 본다. 이 경우 마커 없음으로 처리해 엉뚱한 보정을 막는다.
MARKER_MAX_AREA = 1200
WARMUP_FRAMES   = 8       # 카메라 열자마자 버릴 프레임 수 (자동노출 안정화)
MARKER_RETRY    = 5       # 마커를 못 찾았을 때 새 프레임으로 재시도할 횟수

# 마커가 흡착컵 중심 바로 위에 있지 않을 때의 보정 (칸 단위).
# 예: 마커가 흡착컵보다 파일 방향으로 +0.3칸 치우쳐 보이면 (0.3, 0.0).
# test_square의 'markcal <칸>' 명령으로 자동 측정할 수 있다.
MARKER_OFFSET_COL = 0.0
MARKER_OFFSET_ROW = 0.0

# ─────────────────────────────────────────
# 시차(parallax) 보정 — 마커는 판 위로 떠 있다
# ─────────────────────────────────────────
# ⚠️ 카메라가 판을 정확히 수직으로 내려다보지 않으면, **판 위로 떠 있는**
#    마커는 판 평면에 투영될 때 밀려 보인다. 흡착컵이 e7 바로 위에 있는데
#    화면에는 e8 로 잡히는 식이다(실제로 겪음).
#
#    밀림 = h/(H-h) x (카메라 바로 아래 지점에서의 거리)
#      h=마커 높이, H=카메라 높이. 예: H=40cm, h=4.5cm, 가장자리 20cm →
#      약 0.87칸이 밀린다.
#
#    핵심은 이 밀림이 **위치에 비례해 선형으로 커진다**는 점이다. 그래서
#    상수 오프셋 하나로는 못 고치고(그 칸에서만 맞음), 어파인 변환이 필요하다.
#      실제칸 = A*측정칸 + B*측정랭크 + C
#      실제랭크 = D*측정칸 + E*측정랭크 + F
#    test_square 의 'markcal fit' 이 여러 칸에서 재서 이 6개를 구해준다.
# [A,B,C, D,E,F] 또는 None(보정 안 함)
MARKER_FIT = None


def apply_marker_fit(col: float, row: float) -> tuple:
    """마커가 보이는 위치 → 흡착컵의 실제 위치 (칸 단위)."""
    if MARKER_FIT is None:
        return (col - MARKER_OFFSET_COL, row - MARKER_OFFSET_ROW)
    a, b, c, d, e, f = MARKER_FIT
    return (a * col + b * row + c, d * col + e * row + f)

# ─────────────────────────────────────────
# 감마 — 흰 기물과 밝은 칸을 갈라놓는 장치
# ─────────────────────────────────────────
# 흰 기물 윗면과 카페라떼색 칸은 **색상(H)이 거의 같다**(둘 다 H≈18).
# 갈라놓는 축은 채도(S) 뿐이다: 흰 기물 S≈5, 카페라떼 S≈83.
#
# 감마 g>1 을 걸면(=어둡게) 채도 간격이 벌어진다. S = 1 - min/max 인데
# 각 채널을 g 제곱하면 (min/max)^g 가 작아져서 S가 커지고, 원래 채도가
# 높은 쪽이 더 많이 커지기 때문이다. 실측 근사값 기준:
#     g=1.0 → 흰 S=5,  라떼 S= 83   간격 78
#     g=2.0 → 흰 S=10, 라떼 S=139   간격 129
# 흰 기물이 노출 과다로 255에 붙어 있을 때도 되살려 준다.
#
# ⚠️ 감마를 바꾸면 빈 판 기준 영상(empty_ref.npz)도 다시 찍어야 한다.
#    기준 영상에 그때의 감마를 같이 저장해 두고, 다르면 경고한다.
GAMMA = 1.0
_GAMMA_LUT = None


def set_gamma(g: float):
    """감마를 바꾸고 LUT를 다시 만든다. 1.0 이면 아무것도 안 한다."""
    global GAMMA, _GAMMA_LUT
    GAMMA = float(g)
    if abs(GAMMA - 1.0) < 1e-6:
        _GAMMA_LUT = None
        return
    _GAMMA_LUT = np.array([((i / 255.0) ** GAMMA) * 255.0
                           for i in range(256)], dtype=np.uint8)


def apply_gamma(img: np.ndarray) -> np.ndarray:
    return img if _GAMMA_LUT is None else cv2.LUT(img, _GAMMA_LUT)


# ─────────────────────────────────────────
# 카메라 노출 고정 — 감마보다 먼저 해야 하는 것
# ─────────────────────────────────────────
# ⚠️ 감마는 **이미 찍힌 픽셀**에 거는 후처리다. 밝은 칸이 노출 과다로
#    255에 붙어(clipping) 있으면 (255/255)^g = 255 라서 감마를 아무리 올려도
#    영원히 흰색이다. 정보가 센서 단계에서 이미 지워졌기 때문이다.
#    → 이때는 **카메라 노출 자체를 낮춰야** 한다.
#
# 자동노출을 켜 두면 또 다른 문제가 있다. 노출이 계속 움직여서 방금 연
# 프로그램과 계속 돌던 프로그램이 서로 다른 색을 본다(실제로 겪음:
# --watch 는 64/64, 한 장만 읽으면 48/64).
# 노출을 고정하면 두 문제가 한 번에 사라진다.
#
# 값은 드라이버마다 뜻이 다르다(어떤 건 -6 같은 로그값, 어떤 건 78 같은 원값).
# 그래서 절대값을 정해두지 않고, pick_pieces 에서 눈으로 맞춘 값을
# colors.json 에 저장해 모든 도구가 같이 쓴다.
# ⚠️ 그리고 자동 **화이트밸런스**가 더 고약하다. AWB 는 화면 평균을 무채색으로
#    맞추려 한다. 그런데 이 판은 대부분이 베이지(카페라떼)색이라, AWB 가 그
#    색조를 통째로 상쇄해서 **밝은 칸을 흰색으로 만들어 버린다**.
#    실측 역산: 감마 3.0 에서 밝은 칸 HSV=(50,9,226) → 센서 원본 V≈245 S≈3.
#    나무색이라면 S≈83 이어야 하는데 S≈3, 즉 채널이 거의 같은 무채색이다.
#    이래서는 흰 기물과 원리상 구분할 수 없다. 감마로도 못 고친다(S를 곱해
#    키워도 3 은 거의 0 이다).
CAM_EXPOSURE = None       # None = 자동노출 그대로
CAM_GAIN = None
CAM_WB_TEMP = None        # None = 자동 화이트밸런스 그대로
# V4L2 계열은 '수동'을 나타내는 값이 드라이버마다 1 또는 0.25 다. 둘 다 시도한다.
_MANUAL_EXPOSURE_VALUES = (1, 0.25, 0)
AUTO_EXPOSURE_VALUES = (3, 0.75)


def apply_camera_controls(cap, exposure=None, gain=None, wb_temp=None,
                          verbose=True):
    """노출·게인·화이트밸런스를 고정한다. 카메라가 안 받아 주면 넘어간다.

    ⚠️ UVC 웹캠 중에는 이 속성들을 무시하는 것이 많다. 반영됐는지 되읽어
       확인하고, 안 되면 그렇다고 말한다(조용히 실패하면 원인을 못 찾는다).
    """
    done = []
    if exposure is not None:
        for v in _MANUAL_EXPOSURE_VALUES:
            if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v):
                break
        cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
        done.append(f"노출 {exposure:g} → 실제 {cap.get(cv2.CAP_PROP_EXPOSURE):g}")
    if gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, float(gain))
        done.append(f"게인 {gain:g}")
    if wb_temp is not None:
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(wb_temp))
        done.append(f"화이트밸런스 {wb_temp:g}K → 실제 "
                    f"{cap.get(cv2.CAP_PROP_WB_TEMPERATURE):g}")
    if done and verbose:
        print("[Detector] 카메라 고정: " + ", ".join(done))
    return bool(done)


# ─────────────────────────────────────────
# 실측한 색 설정 (vision/colors.json)
# ─────────────────────────────────────────
# ⚠️ 색은 조명·카메라·기물마다 다르므로 **설치 환경마다 다른 값**이다.
#    calibration.json 과 같은 성격이라 git에 올리지 않는다.
#
# 예전에는 pick_pieces.py / pick_marker.py 가 이 파일(detect.py)을 직접
# 고쳤다. 그런데 detect.py 는 git이 추적하는 파일이라, 색을 등록할 때마다
# `git pull` 이 "로컬 변경이 덮어써진다"며 실패했다. 그래서 값만 따로 뺐다.
# 위의 기본값은 그대로 두고, 파일이 있으면 그 값으로 덮어쓴다.
COLORS_PATH = os.path.join(os.path.dirname(__file__), "colors.json")


def _as_ranges(v):
    """[[[h,s,v],[h,s,v]], ...] → [((h,s,v),(h,s,v)), ...]"""
    return [(tuple(lo), tuple(hi)) for lo, hi in v]


def load_colors(path: str = None) -> bool:
    """colors.json 이 있으면 색 설정을 그 값으로 바꾼다. 반환: 로드했는가."""
    global MARKER_HSV_RANGES, PIECE_HSV_WHITE, PIECE_HSV_BLACK, PIECE_COLOR_MODE
    global MARKER_OFFSET_COL, MARKER_OFFSET_ROW, MARKER_FIT
    global CAM_EXPOSURE, CAM_GAIN, CAM_WB_TEMP
    path = path or COLORS_PATH
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        print(f"[Detector] colors.json 읽기 실패: {e} (기본 색 사용)")
        return False
    if d.get("marker"):
        MARKER_HSV_RANGES = _as_ranges(d["marker"])
    if "white" in d:
        PIECE_HSV_WHITE = _as_ranges(d["white"] or [])
    if "black" in d:
        PIECE_HSV_BLACK = _as_ranges(d["black"] or [])
    if "color_mode" in d:
        PIECE_COLOR_MODE = bool(d["color_mode"])
    if "marker_offset" in d:
        MARKER_OFFSET_COL, MARKER_OFFSET_ROW = (float(v) for v in d["marker_offset"])
    if "marker_fit" in d:
        MARKER_FIT = [float(v) for v in d["marker_fit"]] if d["marker_fit"] else None
    if "gamma" in d:
        set_gamma(d["gamma"])
    if "cam_exposure" in d:
        CAM_EXPOSURE = d["cam_exposure"]
    if "cam_gain" in d:
        CAM_GAIN = d["cam_gain"]
    if "cam_wb" in d:
        CAM_WB_TEMP = d["cam_wb"]
    return True


def save_colors(marker=None, white=None, black=None, color_mode=None,
                marker_offset=None, marker_fit=None, gamma=None,
                cam_exposure=..., cam_gain=..., cam_wb=...,
                path: str = None):
    """colors.json 에 색 설정을 저장한다. None 인 항목은 기존 값을 유지."""
    path = path or COLORS_PATH
    d = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            d = {}
    if marker is not None:
        d["marker"] = [[list(lo), list(hi)] for lo, hi in marker]
    if white is not None:
        d["white"] = [[list(lo), list(hi)] for lo, hi in white]
    if black is not None:
        d["black"] = [[list(lo), list(hi)] for lo, hi in black]
    if color_mode is not None:
        d["color_mode"] = bool(color_mode)
    if marker_offset is not None:
        d["marker_offset"] = [float(marker_offset[0]), float(marker_offset[1])]
    if marker_fit is not None:
        d["marker_fit"] = [float(v) for v in marker_fit] if marker_fit else None
    if gamma is not None:
        d["gamma"] = float(gamma)
    # ⚠️ 노출은 None(자동) 도 뜻이 있는 값이라, '안 건드림'을 ... 로 구분한다
    if cam_exposure is not ...:
        d["cam_exposure"] = None if cam_exposure is None else float(cam_exposure)
    if cam_gain is not ...:
        d["cam_gain"] = None if cam_gain is None else float(cam_gain)
    if cam_wb is not ...:
        d["cam_wb"] = None if cam_wb is None else float(cam_wb)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    load_colors(path)          # 방금 저장한 값을 바로 반영
    return path


load_colors()

# ─────────────────────────────────────────
# 사람 수 인식 (합법수 대조 방식)
# ─────────────────────────────────────────
# 1등 후보가 2등보다 이만큼 앞서야 '확실하다'고 본다.
# 64점 만점 채점이므로 1.0 = 칸 하나 차이. 낮추면 자동으로 넘어가는 대신
# 오인식이 늘고, 높이면 사람에게 묻는 횟수가 는다.
MOVE_MATCH_MARGIN = 1.0
# 색(흑/백)까지 맞았을 때 주는 추가 점수. 색 분류는 조명에 민감해 자주
# 틀리므로, 있다/없다(1.0)보다 낮게 둔다.
COLOR_MATCH_WEIGHT = 0.5


def board_to_state(board) -> list:
    """chess.Board → get_board_state 와 같은 형식의 8×8 배열."""
    import chess
    st = []
    for row in range(8):
        r = []
        for col in range(8):
            p = board.piece_at(chess.square(col, row))
            r.append("empty" if p is None
                     else ("white" if p.color == chess.WHITE else "black"))
        st.append(r)
    return st


def format_state(state, other=None) -> str:
    """8×8 배치를 사람이 읽을 수 있게. other 를 주면 다른 칸을 대문자로 표시."""
    sym = {"empty": ".", "white": "w", "black": "b"}
    lines = []
    for row in range(7, -1, -1):
        cells = []
        for col in range(8):
            c = sym[state[row][col]]
            if other is not None and state[row][col] != other[row][col]:
                c = c.upper() if c != "." else "X"
            cells.append(c)
        lines.append(f"  {row+1} " + " ".join(cells))
    lines.append("    " + " ".join("abcdefgh"))
    return "\n".join(lines)


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


def chess_to_grid_inv(gx: int, gy: int) -> tuple:
    """chess_to_grid 의 역변환 — 화면 격자 (gx,gy) → 체스 좌표 (col,row)."""
    if ROBOT_SIDE == "left":
        return (7 - gy, 7 - gx)
    if ROBOT_SIDE == "right":
        return (gy, gx)
    if ROBOT_SIDE == "top":
        return (gx, 7 - gy)
    if ROBOT_SIDE == "bottom":
        return (7 - gx, gy)
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
        self._last_marker_area = 0
        self._last_means = None
        self._last_frac = None
        self._load_empty_ref()
        apply_camera_controls(self.cap, CAM_EXPOSURE, CAM_GAIN,
                              CAM_WB_TEMP)
        self._warmup()
        print(f"[Detector] 카메라 {camera_index} 초기화 완료")

    def _warmup(self, max_frames: int = 90, settle: int = 4,
                tol: float = 0.4) -> int:
        """자동노출·화이트밸런스가 **안정될 때까지** 프레임을 버린다.

        ⚠️ 예전에는 무조건 8프레임만 버렸다. 그런데 자동노출이 잡히는 데는
           보통 1초(≈30프레임) 넘게 걸린다. 그래서 계속 돌던 라이브 뷰와
           방금 연 check_board 가 **서로 다른 색을 보고 다른 판정**을 냈다.
           밝게 뜬 프레임에서는 카페라떼 칸이 채도를 잃어 흰 기물 범위 안으로
           들어와 버린다(빈 랭크가 통째로 흰 기물이 되는 증상).
        여기서는 화면 평균 밝기가 연속으로 안 변할 때까지 기다린다.
        """
        prev, stable, used = None, 0, 0
        for _ in range(max_frames):
            ok, f = self.cap.read()
            if not ok:
                continue
            used += 1
            # ⚠️ 전체 평균만 보면 부족하다. 화이트밸런스는 평균 밝기가 멈춘
            #    뒤에도 계속 움직여서 색이 바뀐다. 채널별로 본다.
            m = f[::8, ::8].reshape(-1, 3).mean(axis=0)
            if prev is not None and float(np.abs(m - prev).max()) < tol:
                stable += 1
                if stable >= settle and used >= WARMUP_FRAMES:
                    break
            else:
                stable = 0
            prev = m
        if stable < settle:
            print(f"  ⚠️ 노출이 {used}프레임 동안 안 잡혔습니다 — 조명이 깜빡이거나"
                  " 카메라 자동노출이 계속 움직이는 중입니다.")
        return used

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
        # 판의 네 꼭짓점을 '여유를 뺀 안쪽 사각형'에 맞춘다.
        # 그러면 판 바깥도 MARGIN_PX 만큼 같이 펴져서 마커가 안 잘린다.
        lo, hi = MARGIN_PX, MARGIN_PX + BOARD_PX - 1
        dst = np.array([
            [lo, lo], [hi, lo], [hi, hi], [lo, hi],
        ], dtype=np.float32)
        self._M = cv2.getPerspectiveTransform(corners, dst)
        print(f"[Detector] 캘리브레이션 로드: {path}")

    # ─────────────────────────────────────────
    # 탑뷰 변환
    # ─────────────────────────────────────────
    def _get_top_view(self, frame: np.ndarray) -> np.ndarray:
        # 감마는 여기 한 곳에서만 건다. 색 판정·빈 판 비교·밝기·마커가 모두
        # 이 함수를 거치므로, 어디 한 군데만 감마가 다른 사고가 안 난다.
        return apply_gamma(cv2.warpPerspective(frame, self._M,
                                               (TOP_SIZE, TOP_SIZE)))

    # ─────────────────────────────────────────
    # 셀 밝기 측정
    # ─────────────────────────────────────────
    @staticmethod
    def _cell_mean(cell_img: np.ndarray) -> float:
        """셀 중앙 60%의 평균 밝기."""
        h, w = cell_img.shape[:2]
        m = int(h * 0.2)
        gray = cv2.cvtColor(cell_img[m:h-m, m:w-m], cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    @staticmethod
    def _cell_patch(top: np.ndarray, gx: int, gy: int) -> np.ndarray:
        """격자 (gx,gy) 셀 중앙 60% 의 흑백 패치."""
        x0, y0 = cell_px(gx, gy)
        cell = top[y0:y0+CELL_SIZE_PX, x0:x0+CELL_SIZE_PX]
        m = int(CELL_SIZE_PX * 0.2)
        return cv2.cvtColor(cell[m:-m, m:-m], cv2.COLOR_BGR2GRAY).astype(np.float32)

    def _cell_patches(self, top: np.ndarray) -> np.ndarray:
        """탑뷰 → (8,8,H,W) 셀별 흑백 패치. 인덱스는 [gy][gx]."""
        p0 = self._cell_patch(top, 0, 0)
        out = np.zeros((8, 8) + p0.shape, dtype=np.float32)
        for gy in range(8):
            for gx in range(8):
                out[gy, gx] = self._cell_patch(top, gx, gy)
        return out

    # ─────────────────────────────────────────
    # 빈 판 기준 영상
    # ─────────────────────────────────────────
    def _load_empty_ref(self):
        self._ref = None
        if not os.path.exists(EMPTY_REF_PATH):
            return
        try:
            z = np.load(EMPTY_REF_PATH)
            self._ref = z["cells"].astype(np.float32)
            print(f"[Detector] 빈 판 기준 영상 로드: {EMPTY_REF_PATH}")
            # 기준 영상은 찍을 당시의 감마로 밝기가 정해져 있다. 지금 감마가
            # 다르면 판 전체가 '달라진 픽셀'로 잡혀 전 칸이 기물이 된다.
            g_ref = float(z["gamma"]) if "gamma" in z.files else 1.0
            if abs(g_ref - GAMMA) > 1e-3:
                print(f"  ⚠️ 기준 영상은 감마 {g_ref:.2f} 로 찍혔는데 지금은 "
                      f"{GAMMA:.2f} 입니다.")
                print("     → python vision/check_board.py --capture-empty 로 "
                      "다시 찍으세요.")
        except Exception as e:
            print(f"[Detector] 빈 판 기준 영상 로드 실패: {e}")

    def capture_empty_reference(self, frames: int = 15, save: bool = True):
        """⚠️ 체스판을 **완전히 비운 상태**에서 호출할 것.

        여러 프레임을 평균해 잡음을 줄인다. 팔은 park 에 두어 그림자가
        게임 때와 같게 한다.
        """
        acc = None
        used = 0
        for _ in range(frames * 2):
            ok, f = self.cap.read()
            if not ok:
                continue
            p = self._cell_patches(self._get_top_view(f))
            acc = p if acc is None else acc + p
            used += 1
            if used >= frames:
                break
        if acc is None:
            raise RuntimeError("카메라 프레임을 못 읽었습니다")
        self._ref = acc / used
        if save:
            np.savez_compressed(EMPTY_REF_PATH, cells=self._ref,
                                gamma=np.float32(GAMMA))
            print(f"[Detector] 빈 판 기준 영상 저장 ({used}프레임 평균, "
                  f"감마 {GAMMA:.2f}) "
                  f"→ {EMPTY_REF_PATH}")
        return self._ref

    # ─────────────────────────────────────────
    # 기물 윗면 색으로 판정
    # ─────────────────────────────────────────
    @staticmethod
    def _color_mask(hsv, ranges):
        m = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        return m

    def _color_fracs(self, top):
        """셀 중앙에서 두 기물 색이 각각 차지하는 비율 (8,8) 두 장."""
        hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        k = np.ones((3, 3), np.uint8)
        mw = cv2.morphologyEx(self._color_mask(hsv, PIECE_HSV_WHITE),
                              cv2.MORPH_OPEN, k)
        mb = cv2.morphologyEx(self._color_mask(hsv, PIECE_HSV_BLACK),
                              cv2.MORPH_OPEN, k)
        m0 = int(CELL_SIZE_PX * 0.2)
        fw = np.zeros((8, 8), np.float32)
        fb = np.zeros((8, 8), np.float32)
        for gy in range(8):
            for gx in range(8):
                x0, y0 = cell_px(gx, gy)
                sl = (slice(y0 + m0, y0 + CELL_SIZE_PX - m0),
                      slice(x0 + m0, x0 + CELL_SIZE_PX - m0))
                fw[gy, gx] = (mw[sl] > 0).mean()
                fb[gy, gx] = (mb[sl] > 0).mean()
        return fw, fb

    @staticmethod
    def _side_from_color(fw_v, fb_v, fallback):
        """색 비율로 어느 편 기물인지 정한다.

        한쪽 색만 등록돼 있으면 '그 색이 보이면 그쪽, 아니면 반대쪽'으로 본다.
        (검은쪽만 파랗게 칠한 경우가 이에 해당)
        """
        w_ok = bool(PIECE_HSV_WHITE) and fw_v >= PIECE_COLOR_MIN_FRAC
        b_ok = bool(PIECE_HSV_BLACK) and fb_v >= PIECE_COLOR_MIN_FRAC
        if PIECE_HSV_WHITE and PIECE_HSV_BLACK:
            if w_ok and b_ok:
                return "white" if fw_v >= fb_v else "black"
            if w_ok:
                return "white"
            if b_ok:
                return "black"
            return fallback
        if PIECE_HSV_BLACK:            # 검은쪽만 칠한 경우
            return "black" if b_ok else "white"
        if PIECE_HSV_WHITE:            # 흰쪽만 칠한 경우
            return "white" if w_ok else "black"
        return fallback

    def _state_from_color(self, top):
        """셀 중앙에서 두 기물 색이 각각 몇 %인지 보고 판정한다.

        판 색깔과 무관하므로 '검은 기물 vs 어두운 칸' 문제가 원천적으로 없다.
        """
        fw, fb = self._color_fracs(top)
        board = []
        for row in range(8):
            line = []
            for col in range(8):
                gx, gy = chess_to_grid(col, row)
                w, b = float(fw[gy, gx]), float(fb[gy, gx])
                if max(w, b) < PIECE_COLOR_MIN_FRAC:
                    line.append("empty")
                else:
                    line.append("white" if w >= b else "black")
            board.append(line)
        self._last_frac = np.maximum(fw, fb)
        self._last_fw, self._last_fb = fw, fb
        self._last_means = None
        self._last_top = top          # 왜 그렇게 판정했는지 되짚기 위해 보관
        return board

    def _state_from_ref(self, top, expect=None):
        """빈 판 기준 영상과 픽셀 단위로 비교해 8×8 상태를 만든다."""
        cur = self._cell_patches(top)
        # ⚠️ 기준 영상이 다른 설정(TOP_SIZE 등)에서 찍혔으면 모양이 안 맞는다.
        #    그대로 빼면 브로드캐스트 오류로 죽으므로 미리 알리고 되돌린다.
        if cur.shape != self._ref.shape:
            print(f"[Detector] 빈 판 기준 영상 크기 불일치 "
                  f"{self._ref.shape} vs {cur.shape} → 무시하고 밝기 방식 사용")
            print("           다시 찍으세요: check_board.py --capture-empty")
            self._ref = None
            return None
        d = cur - self._ref

        # 전역 밝기 변화 보정 — 조명이 전체적으로 밝아/어두워진 만큼만 뺀다.
        # (빈 칸이어야 할 곳들의 중앙값을 0으로 맞춘다)
        samples = []
        for gy in range(8):
            for gx in range(8):
                if expect is not None:
                    c, r = chess_to_grid_inv(gx, gy)
                    if expect[r][c] != "empty":
                        continue
                samples.append(float(np.median(d[gy, gx])))
        if len(samples) >= 8:
            d -= float(np.median(samples))

        changed = np.abs(d) > PIXEL_DIFF_THRESH
        frac = changed.reshape(8, 8, -1).mean(axis=2)

        # ⚠️ 흑/백 구분을 '칸보다 밝은가'로 하면 안 된다. 같은 색 기물이라도
        #    어두운 칸에서는 밝게, 밝은 칸에서는 어둡게 보이므로 판정이
        #    칸 색을 그대로 따라가 격자무늬가 된다(실제로 겪음).
        #    기물 색이 등록돼 있으면 그것으로 정한다.
        use_color = bool(PIECE_HSV_WHITE or PIECE_HSV_BLACK)
        fw, fb = self._color_fracs(top) if use_color else (None, None)

        board, dmean = [], np.zeros((8, 8), dtype=np.float32)
        for row in range(8):
            line = []
            for col in range(8):
                gx, gy = chess_to_grid(col, row)
                ch = changed[gy, gx]
                dm = float(d[gy, gx][ch].mean()) if ch.any() else 0.0
                dmean[gy, gx] = dm
                if frac[gy, gx] < OCC_AREA_FRAC:
                    line.append("empty")
                    continue
                # 기물 색을 양쪽 다 등록했는데 그 칸에서 어느 색도 안 보이면
                # 기물이 없는 것이다. 이때 밝기로 추측하면 안 된다.
                # ⚠️ 빈 판 기준 영상이 낡으면(판·카메라가 움직이거나 조명이
                #    바뀌면) 빈 칸에서도 '변했다'가 나온다. 예전에는 그걸
                #    밝기로 흑/백 아무거나 찍어서 빈 랭크가 기물로 가득 찼다.
                if (use_color and PIECE_HSV_WHITE and PIECE_HSV_BLACK
                        and fw[gy, gx] < PIECE_COLOR_MIN_FRAC
                        and fb[gy, gx] < PIECE_COLOR_MIN_FRAC):
                    line.append("empty")
                    continue
                fallback = "white" if dm > 0 else "black"
                line.append(self._side_from_color(fw[gy, gx], fb[gy, gx], fallback)
                            if use_color else fallback)
            board.append(line)
        self._last_frac, self._last_dmean = frac, dmean
        self._last_fw, self._last_fb = fw, fb
        self._last_means = None          # 밝기 방식 표는 쓰지 않는다
        return board

    def _cell_means(self, top: np.ndarray) -> np.ndarray:
        """탑뷰 → 8×8 평균 밝기 배열. 인덱스는 [gy][gx] (화면 격자)."""
        out = np.zeros((8, 8), dtype=np.float32)
        for gy in range(8):
            for gx in range(8):
                x0, y0 = cell_px(gx, gy)
                cell = top[y0:y0+CELL_SIZE_PX, x0:x0+CELL_SIZE_PX]
                out[gy, gx] = self._cell_mean(cell)
        return out

    def _empty_levels(self, means: np.ndarray, expect=None) -> tuple:
        """빈 칸의 기준 밝기를 (밝은칸, 어두운칸) 두 값으로 추정한다.

        ⚠️ 예전에는 '셀 중앙의 분산이 낮으면 빈 칸'으로 판정했다. 이는 틀렸다.
           기물이 셀 중앙을 덮으면 그 부분은 오히려 **균일**해져 분산이 0에
           가까워진다. 그래서 32개 기물이 전부 empty 로 읽혔다.
           지금은 '빈 칸일 때의 밝기'와 얼마나 다른가로 판단한다.

        expect: 지금 있어야 할 배치(8×8). 주면 그중 **빈 칸**들만 표본으로
                써서 기준을 잡는다 — 조명이 바뀌어도 따라간다.
                없으면 전체 칸의 중앙값을 쓴다(기물이 많으면 부정확).
        """
        light, dark = [], []
        for gy in range(8):
            for gx in range(8):
                if expect is not None:
                    col, row = chess_to_grid_inv(gx, gy)
                    if expect[row][col] != "empty":
                        continue
                (light if (gx + gy) % 2 == 0 else dark).append(float(means[gy, gx]))
        # 표본이 너무 적으면 전체로 후퇴
        if len(light) < 4 or len(dark) < 4:
            light = [float(means[gy, gx]) for gy in range(8) for gx in range(8)
                     if (gx + gy) % 2 == 0]
            dark = [float(means[gy, gx]) for gy in range(8) for gx in range(8)
                    if (gx + gy) % 2 == 1]
        return float(np.median(light)), float(np.median(dark))

    def _classify(self, mean_val: float, ref: float) -> str:
        """빈 칸 기준 밝기와 비교해 판정.
        밝으면 흰 기물, 어두우면 검은 기물, 비슷하면 빈 칸."""
        d = mean_val - ref
        if abs(d) < OCC_DIFF_THRESH:
            return "empty"
        return "white" if d > 0 else "black"

    # ─────────────────────────────────────────
    # 메서드 1: 현재 보드 상태 반환
    # ─────────────────────────────────────────
    def detection_mode(self) -> str:
        """지금 실제로 쓰는 판정 방식 이름. 화면 표시가 실제와 어긋나지 않게
        get_board_state 의 분기와 **같은 조건**을 쓴다."""
        if PIECE_HSV_WHITE and PIECE_HSV_BLACK:
            return "기물 윗면 색 (양쪽 등록)"
        if PIECE_COLOR_MODE and (PIECE_HSV_WHITE or PIECE_HSV_BLACK):
            return "기물 윗면 색 (한쪽만 등록)"
        if self._ref is not None:
            return "빈 판 기준 영상"
        return "밝기 (기준 영상·색 모두 없음)"

    def get_board_state(self, frame=None, expect=None) -> list:
        """
        현재 프레임 캡처 → 8×8 보드 상태 반환.
        board[row][col] ∈ {"empty", "white", "black"}
        ⚠️ 인덱스는 **체스/로봇 좌표**다 (col=파일 a~h, row=랭크 1~8).
           화면 격자 위치는 chess_to_grid()로 변환해 읽는다.

        frame  을 주면 그 프레임으로 읽는다(라이브 뷰가 카메라를 점유 중일 때).
        expect 를 주면 그중 빈 칸들로 '빈 칸 기준 밝기'를 잡는다(조명 변화에 강함).
        """
        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("카메라 프레임 읽기 실패")

        top = self._get_top_view(frame)
        # 양쪽 기물 색이 다 등록돼 있으면 색만으로 확실히 판정된다.
        # 빈 판 기준 영상이 낡아도(판·카메라 이동, 조명 변화) 영향받지 않는다.
        if (PIECE_HSV_WHITE and PIECE_HSV_BLACK) or \
           (PIECE_COLOR_MODE and (PIECE_HSV_WHITE or PIECE_HSV_BLACK)):
            return self._state_from_color(top)
        if self._ref is not None:
            st = self._state_from_ref(top, expect)
            if st is not None:
                return st
            # 기준 영상이 못 쓰게 됐으면 아래 밝기 방식으로 이어간다

        means = self._cell_means(top)
        lv, dv = self._empty_levels(means, expect)
        self._last_means, self._last_levels = means, (lv, dv)

        board = []
        for row in range(8):
            board_row = []
            for col in range(8):
                gx, gy = chess_to_grid(col, row)
                ref = lv if (gx + gy) % 2 == 0 else dv
                board_row.append(self._classify(float(means[gy, gx]), ref))
            board.append(board_row)
        return board

    @staticmethod
    def _in_ranges(v, ranges) -> bool:
        return any(all(lo[i] <= v[i] <= hi[i] for i in range(3))
                   for lo, hi in ranges)

    def _explain_color_ranges(self, expect=None) -> list:
        """등록된 색 범위 안에 **판 색이 들어와 있는지**를 직접 보여준다.

        빈 랭크가 통째로 기물로 읽히는 사고는 거의 항상 이것 하나가 원인이다:
        기물 색 범위가 넓어서 판 색까지 삼킨 것. 표만 봐서는 알 수 없으니
        지금 화면의 빈 칸 색을 실제로 재서 범위 안인지 아닌지 적는다.
        """
        top = getattr(self, "_last_top", None)
        out = ["",
               f"  감마 {GAMMA:.2f}   노출 "
               f"{'자동' if CAM_EXPOSURE is None else CAM_EXPOSURE}"
               f"   화이트밸런스 "
               f"{'자동' if CAM_WB_TEMP is None else CAM_WB_TEMP}",
               f"  등록된 흰쪽 범위   {PIECE_HSV_WHITE}",
               f"  등록된 검은쪽 범위 {PIECE_HSV_BLACK}"]
        if top is None:
            return out
        hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        m0 = int(CELL_SIZE_PX * 0.2)
        # ⚠️ 밝은 칸/어두운 칸을 격자 홀짝으로 정하면 안 된다. 어느 홀짝이
        #    밝은 칸인지는 판을 어느 방향으로 놨느냐에 따라 뒤집힌다.
        #    실제로 재서 밝기 순으로 가른다.
        cells = []
        for gy in range(8):
            for gx in range(8):
                if expect is not None:
                    c, r = chess_to_grid_inv(gx, gy)
                    if expect[r][c] != "empty":
                        continue
                x0, y0 = cell_px(gx, gy)
                patch = hsv[y0+m0:y0+CELL_SIZE_PX-m0,
                            x0+m0:x0+CELL_SIZE_PX-m0].reshape(-1, 3)
                bgr = top[y0+m0:y0+CELL_SIZE_PX-m0, x0+m0:x0+CELL_SIZE_PX-m0]
                cells.append((np.median(patch, axis=0),
                              float((bgr.max(axis=2) >= 253).mean())))
        if not cells:
            return out
        cells.sort(key=lambda t: t[0][2])          # V 오름차순
        half = max(1, len(cells) // 2)
        groups = [("어두운 칸", cells[:half]), ("밝은 칸", cells[half:])]

        out.append("  지금 화면의 빈 칸 색 (중앙값):")
        bad = clipped = False
        light_sat = None
        for key, items in groups:
            v = np.median(np.array([c[0] for c in items]), axis=0)
            clip = float(np.mean([c[1] for c in items])) * 100
            hits = []
            if PIECE_HSV_WHITE and self._in_ranges(v, PIECE_HSV_WHITE):
                hits.append("흰쪽"); bad = True
            if PIECE_HSV_BLACK and self._in_ranges(v, PIECE_HSV_BLACK):
                hits.append("검은쪽"); bad = True
            mark = f"❌ {'/'.join(hits)} 기물 범위 안!" if hits else "✅ 범위 밖"
            note = ""
            if clip >= 2:
                note = f"   ⚠️ 픽셀 {clip:.0f}% 가 255에 붙음(노출 과다)"
                clipped = True
            out.append(f"    {key}  HSV=({v[0]:.0f},{v[1]:.0f},{v[2]:.0f})  "
                       f"{mark}{note}")
            if key == "밝은 칸":
                light_sat = float(v[1])
        # ⚠️ 밝은 칸의 채도가 바닥이면, 그 칸은 '흰색'으로 찍히고 있는 것이다.
        #    흰 기물과 같은 색이니 어떤 임계값으로도 못 가른다.
        if light_sat is not None and light_sat < 25:
            out += [
                f"  ⚠️ 밝은 칸의 채도가 {light_sat:.0f} 밖에 안 됩니다 "
                "(나무색이면 80 근처여야 합니다).",
                "     지금 카메라는 밝은 칸을 **무채색 흰색**으로 찍고 있습니다.",
                "     흰 기물과 색이 같으니 임계값을 어떻게 바꿔도 못 가릅니다.",
                "     거의 항상 원인은 자동 화이트밸런스입니다 — AWB 는 화면 평균을",
                "     무채색으로 맞추려 하는데, 이 판은 대부분이 베이지색이라",
                "     그 색조를 통째로 상쇄해 버립니다.",
                "     → 화이트밸런스와 노출을 고정하세요:",
                f"         python vision/lock_camera.py --camera {self.camera_index}",
            ]
        if clipped:
            out += [
                "  → 노출 과다입니다. 감마로는 **절대** 못 고칩니다:",
                "     255에 붙은 픽셀은 (255/255)^g = 255 라서 감마를 올려도 그대로입니다.",
                "     센서 단계에서 이미 정보가 지워졌습니다.",
                "     → 카메라 노출을 낮추세요:",
                "         python vision/pick_pieces.py --camera 1   에서 - / = 키",
                "       화면 위 clip 이 0% 가 될 때까지 낮춘 뒤 색을 다시 등록하세요.",
            ]
        elif bad:
            out += [
                "  → 빈 칸이 기물로 읽히는 이유가 바로 이것입니다. 색 범위가 판 색까지",
                "     덮고 있습니다. 다시 등록하세요 (판을 음성 표본으로 같이 모읍니다):",
                "       python vision/pick_pieces.py --camera 1",
                "       - / = 로 노출을 맞추고 → b(빈 칸 자동수집)",
                "       → 1/2 로 기물 클릭 → s",
            ]
        return out

    def explain_board_state(self, expect=None) -> str:
        """마지막 판독의 숫자를 표로 보여준다 — 임계값을 조정할 때 쓴다."""
        if PIECE_COLOR_MODE and getattr(self, "_last_fw", None) is not None:
            lines = [f"  기물 윗면 색으로 판정 (임계 {PIECE_COLOR_MIN_FRAC*100:.0f}%)",
                     "  칸별  흰쪽% / 검은쪽% :"]
            for row in range(7, -1, -1):
                cells = []
                for col in range(8):
                    gx, gy = chess_to_grid(col, row)
                    cells.append(f"{self._last_fw[gy,gx]*100:3.0f}/{self._last_fb[gy,gx]*100:<3.0f}")
                lines.append(f"   {row+1} " + " ".join(cells))
            lines.append("      " + "       ".join("abcdefgh"))
            lines.append("  둘 다 임계 미만이면 빈 칸. 기물이 작으면 "
                         "PIECE_COLOR_MIN_FRAC 을 낮춘다.")
            lines += self._explain_color_ranges(expect)
            return "\n".join(lines)
        if getattr(self, "_ref", None) is not None and self._last_frac is not None:
            lines = [f"  빈 판 기준 영상과 비교 (픽셀차 > {PIXEL_DIFF_THRESH} 인 비율)",
                     f"  판정 임계 OCC_AREA_FRAC = {OCC_AREA_FRAC:.2f}",
                     "  칸별 변화 비율 %:"]
            for row in range(7, -1, -1):
                cells = []
                for col in range(8):
                    gx, gy = chess_to_grid(col, row)
                    cells.append(f"{self._last_frac[gy,gx]*100:4.0f}")
                lines.append(f"   {row+1} " + " ".join(cells))
            lines.append("     " + "    ".join("abcdefgh"))
            lines.append(f"  {OCC_AREA_FRAC*100:.0f}% 넘으면 기물 있음. "
                         "기물이 작으면 OCC_AREA_FRAC 를 낮춘다.")
            if getattr(self, "_last_fb", None) is not None:
                lines.append("")
                lines.append(f"  흑/백은 색으로 판정 (임계 {PIECE_COLOR_MIN_FRAC*100:.0f}%)."
                             "  칸별 흰쪽%/검은쪽% :")
                for row in range(7, -1, -1):
                    cells = []
                    for col in range(8):
                        gx, gy = chess_to_grid(col, row)
                        cells.append(f"{self._last_fw[gy,gx]*100:3.0f}/"
                                     f"{self._last_fb[gy,gx]*100:<3.0f}")
                    lines.append(f"   {row+1} " + " ".join(cells))
                lines.append("      " + "       ".join("abcdefgh"))
                lines += self._explain_color_ranges(expect)
            else:
                lines.append("  ⚠️ 기물 색이 등록돼 있지 않아 흑/백을 '칸보다 밝은가'로"
                             " 판정합니다.")
                lines.append("     같은 색 기물도 칸 색에 따라 뒤집혀 격자무늬가 됩니다.")
                lines.append("     → python vision/pick_pieces.py 로 기물 색을 등록하세요.")
            return "\n".join(lines)
        if getattr(self, "_last_means", None) is None:
            return "  (아직 판독한 적이 없습니다)"
        lv, dv = self._last_levels
        lines = [f"  빈 칸 기준 밝기: 밝은칸 {lv:.0f} / 어두운칸 {dv:.0f}",
                 f"  판정 임계 OCC_DIFF_THRESH = {OCC_DIFF_THRESH}",
                 "  칸별 (밝기, 기준과의 차이):"]
        for row in range(7, -1, -1):
            cells = []
            for col in range(8):
                gx, gy = chess_to_grid(col, row)
                ref = lv if (gx + gy) % 2 == 0 else dv
                d = float(self._last_means[gy, gx]) - ref
                cells.append(f"{d:+5.0f}")
            lines.append(f"   {row+1} " + " ".join(cells))
        lines.append("     " + "     ".join("abcdefgh"))
        lines.append("  |차이| 가 임계보다 크면 기물 있음 (+흰, -검)")
        return "\n".join(lines)

    def get_stable_board_state(self, tries: int = 6, frame_src=None, expect=None):
        """연속 두 번 같게 읽힐 때만 결과를 돌려준다.

        팔·손이 지나가거나 조명이 흔들리는 순간에 읽으면 엉뚱한 배열이 나온다.
        같은 결과가 두 번 연달아 나와야 '안정된 상태'로 본다.
        끝내 안정되지 않으면 None.

        frame_src: 매번 새 프레임을 돌려주는 콜러블(라이브 뷰 공유용).
        """
        prev = None
        for _ in range(tries):
            cur = self.get_board_state(frame_src() if frame_src else None, expect)
            if prev is not None and cur == prev:
                return cur
            prev = cur
            time.sleep(0.15)
        return None

    # ─────────────────────────────────────────
    # 메서드 2-A: 규칙을 아는 상태에서의 이동 감지 (권장)
    # ─────────────────────────────────────────
    def detect_move_with_rules(self, board, frame_src=None):
        """합법수 중 '현재 화면과 가장 잘 맞는 수'를 고른다 → chess.Move.

        왜 이 방식인가:
          이전 방식은 '사라진 칸 1개 + 나타난 칸 1개'를 찾는 차분(diff)이었다.
          그런데 칸 하나만 잘못 읽혀도 조건이 깨져 매번 실패했고,
          기물을 잡는 수(도착칸이 원래 차 있음)는 원리적으로 못 읽었다.

          지금은 규칙을 이미 안다는 점을 이용한다. 합법수는 보통 20~40개뿐이고,
          각 수를 뒀을 때의 배치를 미리 알 수 있다. 그중 화면과 가장 비슷한
          것을 고르면, 몇 칸을 잘못 읽어도 정답이 살아남는다.

        반환: (chess.Move, 점수, 후보목록) 또는 (None, 0, 후보목록)
        """

        expect = board_to_state(board)
        observed = self.get_stable_board_state(frame_src=frame_src, expect=expect)
        if observed is None:
            return None, 0.0, []

        cands = []
        for mv in board.legal_moves:
            board.push(mv)
            cands.append((self._state_score(observed, board_to_state(board)), mv))
            board.pop()
        if not cands:
            return None, 0.0, []

        cands.sort(key=lambda t: -t[0])
        best, second = cands[0], (cands[1] if len(cands) > 1 else (-1e9, None))
        self._last_observed = observed

        # 1등이 2등보다 뚜렷하게 나아야 믿는다. 비슷하면 사람에게 묻는다.
        if best[0] - second[0] >= MOVE_MATCH_MARGIN:
            return best[1], best[0], cands
        return None, best[0], cands

    def _state_score(self, obs, exp) -> float:
        """관측 배치와 예상 배치가 얼마나 닮았는지 (64점 만점).

        '있다/없다'가 맞으면 큰 점수, 색까지 맞으면 추가 점수를 준다.
        색 분류는 조명에 민감해 자주 틀리므로 가중치를 낮게 둔다.
        """
        s = 0.0
        for row in range(8):
            for col in range(8):
                o, e = obs[row][col], exp[row][col]
                if (o == "empty") == (e == "empty"):
                    s += 1.0                       # 점유 여부 일치
                    if o != "empty" and o == e:
                        s += COLOR_MATCH_WEIGHT    # 색까지 일치
        return s

    # ─────────────────────────────────────────
    # 메서드 2-B: 규칙 없이 차분으로 감지 (예비용)
    # ─────────────────────────────────────────
    def detect_human_move(self) -> tuple:
        """
        엔터 대기 → 이전/현재 보드 비교 → (from_sq, to_sq) 반환
        from_sq, to_sq = (col, row) 형식
        ⚠️ 칸 하나만 잘못 읽혀도 실패한다. 가능하면 detect_move_with_rules 를 쓸 것.
        """
        print("  수를 두고 Enter를 누르세요...", end="", flush=True)
        input()   # 엔터 대기

        current = self.get_stable_board_state()
        if current is None:
            print("  [경고] 화면이 안정되지 않았습니다 "
                  "(팔이나 손이 보드를 가리는 중?). 다시 시도하세요.")
            return None

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
            gx0, gy0 = cell_px(i, i)
            cv2.line(top, (gx0, MARGIN_PX), (gx0, MARGIN_PX+BOARD_PX), (0,255,0), 1)
            cv2.line(top, (MARGIN_PX, gy0), (MARGIN_PX+BOARD_PX, gy0), (0,255,0), 1)

        # 기물 표시 (board는 체스 좌표 → 화면 위치로 변환)
        if board:
            for row in range(8):
                for col in range(8):
                    state = board[row][col]
                    gx, gy = chess_to_grid(col, row)
                    _x0, _y0 = cell_px(gx, gy)
                    cx = _x0 + CELL_SIZE_PX // 2
                    cy = _y0 + CELL_SIZE_PX // 2
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
                    x1, y1 = cell_px(gx, gy)
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
            self._last_marker_px = None
            self._last_marker_cnt = None
            mk = self.find_marker(frame)
            if mk is not None:
                # ⚠️ 칸 중앙이 아니라 **실제로 잡힌 픽셀 위치**에 그린다.
                px_py = getattr(self, "_last_marker_px", None)
                if px_py is not None:
                    px, py = int(round(px_py[0])), int(round(px_py[1]))
                    cnt = getattr(self, "_last_marker_cnt", None)
                    if cnt is not None:            # 잡힌 덩어리 윤곽
                        cv2.drawContours(top, [cnt], -1, (0, 140, 255), 1)
                    cv2.circle(top, (px, py), 7, (0, 0, 255), 2)
                    cv2.line(top, (px-11, py), (px+11, py), (0, 0, 255), 1)
                    cv2.line(top, (px, py-11), (px, py+11), (0, 0, 255), 1)
                # 소수점까지 보여준다 — 칸 안 어디쯤인지 알 수 있게
                cv2.putText(top,
                            f"MARK {chr(97+int(round(mk[0])))}{int(round(mk[1]))+1}"
                            f"  ({mk[0]:.2f},{mk[1]:.2f})",
                            (5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            else:
                why = getattr(self, "_last_marker_fail", None)
                cv2.putText(top, f"MARK: not found — {why}" if why
                            else "MARK: not found", (5, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            # 축 방향 안내 — 로봇이 있는 변을 표시
            side_txt = {"bottom": "ROBOT THIS SIDE (v)", "top": "ROBOT THIS SIDE (^)",
                        "left": "ROBOT <", "right": "> ROBOT"}[ROBOT_SIDE]
            pos = {"bottom": (5, TOP_SIZE-6), "top": (5, 16),
                   "left": (5, TOP_SIZE//2), "right": (TOP_SIZE-95, TOP_SIZE//2)}[ROBOT_SIDE]
            # 판 경계를 그려 '여유 영역'을 눈으로 구분할 수 있게
            cv2.rectangle(top, (MARGIN_PX, MARGIN_PX),
                          (MARGIN_PX+BOARD_PX-1, MARGIN_PX+BOARD_PX-1), (0,255,0), 2)
            cv2.putText(top, side_txt, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORIGIN_COLOR, 2)

        return top

    # ─────────────────────────────────────────
    # 메서드 4: 흡착컵 마커 위치 (시각 피드백 보정용)
    # ─────────────────────────────────────────
    def margin_report(self, frame=None) -> str:
        """판 **바깥 여유 영역**이 탑뷰에서 검게 나오는지 잰다.

        ⚠️ 여유(TOP_MARGIN_CELLS)를 늘려도, 카메라가 판 너머를 실제로 보고
           있지 않으면 그 자리는 그냥 검다. warpPerspective 는 원본 밖을
           검게 채우기 때문이다. 이건 소프트웨어로 못 고치고 카메라를
           옮겨야 한다 — 그러니 어느 쪽이 안 보이는지 숫자로 보여준다.
        """
        if frame is None:
            ok, frame = self.cap.read()
            if not ok:
                return "카메라 프레임을 못 읽었습니다"
        h, w = frame.shape[:2]
        # ⚠️ '검은 픽셀 비율'로 재면 안 된다 — 판 주위 책상이 어두우면
        #    화각 안인데도 안 보인다고 나온다. 기하로 정확히 따진다:
        #    탑뷰 좌표를 역변환해 원본 화면 안에 들어오는지 본다.
        Minv = np.linalg.inv(self._M)

        def src_ok(u, v):
            """칸 좌표 (u,v) 가 원본 화면 안에 들어오는가."""
            pt = np.array([[[MARGIN_PX + u * CELL_SIZE_PX,
                             MARGIN_PX + v * CELL_SIZE_PX]]], dtype=np.float32)
            s = cv2.perspectiveTransform(pt, Minv)[0, 0]
            return 0 <= s[0] < w and 0 <= s[1] < h

        def reach(side):
            """그 방향으로 판 밖 몇 칸까지 카메라가 보는가 (0.1칸 단위)."""
            best = 0.0
            for k in range(1, 41):                 # 최대 4칸까지 확인
                d = k * 0.1
                pts = {"left":   [(-d, t) for t in (0.5, 4, 7.5)],
                       "right":  [(8 + d, t) for t in (0.5, 4, 7.5)],
                       "top":    [(t, -d) for t in (0.5, 4, 7.5)],
                       "bottom": [(t, 8 + d) for t in (0.5, 4, 7.5)]}[side]
                if not all(src_ok(u, v) for u, v in pts):
                    break
                best = d
            return best

        probe = {"left": (0, 3), "right": (7, 3), "top": (3, 0), "bottom": (3, 7)}
        lines = [f"판 바깥을 카메라가 몇 칸까지 보는가 "
                 f"(지금 필요한 여유 = {TOP_MARGIN_CELLS}칸):"]
        blind = []
        for side in ("left", "right", "top", "bottom"):
            r = reach(side)
            gx, gy = probe[side]
            col, row = chess_to_grid_inv(gx, gy)
            name = (f"랭크 {row+1}" if side in ("left", "right")
                    else f"파일 {chr(97+col)}")
            mark = ("✅ 충분" if r >= TOP_MARGIN_CELLS else
                    ("△ 모자람" if r >= 0.5 else "❌ 거의 못 봄"))
            lines.append(f"    {name} 바깥: {r:.1f}칸까지 보임  {mark}")
            if r < 0.5:
                blind.append(name)
        if blind:
            lines += [
                f"  → {', '.join(blind)} 쪽은 카메라가 아예 못 보고 있습니다.",
                "     그 칸에 팔이 가면 마커가 화면 밖으로 나가 정렬이 안 됩니다.",
                "     TOP_MARGIN_CELLS 를 늘려도 소용없습니다 — 원본에 없는 화소입니다.",
                "     → 카메라를 뒤로/위로 옮겨 판 주위가 한 칸 이상 남게 한 뒤",
                "       python vision/auto_calibrate.py --camera N 을 다시 하세요.",
            ]
        return "\n".join(lines)

    def marker_debug(self, frame=None) -> str:
        """마커를 **원본 화면**에서도 찾아, 탑뷰에 왜 안 보이는지 가른다.

        세 가지 실패는 고치는 방법이 완전히 다른데, 탑뷰만 봐서는 구분이
        안 된다:
          ① 원본에도 마커 색이 없다        → 색/조명 문제 (pick_marker)
          ② 원본엔 있는데 탑뷰 밖으로 나간다 → 여유(TOP_MARGIN_CELLS) 부족
          ③ 원본엔 있고 탑뷰 안인데 검다    → 카메라가 그 영역을 아예 못 본다
                                            (판 너머가 화각 밖) → 카메라를
                                            뒤로/위로 옮겨야 한다
        """
        if frame is None:
            ok, frame = self.cap.read()
            if not ok:
                return "카메라 프레임을 못 읽었습니다"
        hsv = cv2.cvtColor(apply_gamma(frame), cv2.COLOR_BGR2HSV)
        mask = cv2.morphologyEx(self._color_mask(hsv, MARKER_HSV_RANGES),
                                cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 원본은 탑뷰와 배율이 달라 MARKER_MIN_AREA 를 그대로 쓸 수 없다
        cnts = [c for c in cnts if cv2.contourArea(c) >= 20]
        if not cnts:
            # ⚠️ '색이 안 맞아서 못 봄'과 '아예 화각 밖이라 못 봄'은 여기서
            #    구분이 안 된다. 여유 영역이 검은지 같이 보여 판단하게 한다.
            return ("① 원본 화면에도 마커 색이 없습니다.\n"
                    "     → 색 문제이거나, 마커가 카메라 화각 밖입니다.\n"
                    "     " + self.margin_report(frame).replace("\n", "\n     ") + "\n"
                    "     여유가 ✅ 인데도 못 찾으면 색 문제입니다:\n"
                    "       python vision/pick_marker.py --camera N  "
                    "(랭크8 자세에서도 클릭)")
        best = max(cnts, key=cv2.contourArea)
        M = cv2.moments(best)
        if M["m00"] == 0:
            return "① 마커 덩어리를 찾았지만 중심을 계산할 수 없습니다"
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        pt = cv2.perspectiveTransform(
            np.array([[[cx, cy]]], dtype=np.float32), self._M)[0, 0]
        u, v = px_to_grid_uv(float(pt[0]), float(pt[1]))
        inside = 0 <= pt[0] < TOP_SIZE and 0 <= pt[1] < TOP_SIZE
        head = (f"원본 화면에서는 마커를 찾았습니다 "
                f"(원본 픽셀 {cx:.0f},{cy:.0f}, 면적 {cv2.contourArea(best):.0f})\n"
                f"     탑뷰로 옮기면 픽셀 ({pt[0]:.0f},{pt[1]:.0f}) = 칸 좌표 "
                f"({u:.2f},{v:.2f})")
        if not inside:
            need = max(abs(min(u, v, 0.0)), max(u - 8, v - 8, 0.0))
            return (head + "\n"
                    f"     ② 탑뷰 밖입니다. 지금 여유는 {TOP_MARGIN_CELLS}칸인데 "
                    f"{need:.2f}칸이 필요합니다.\n"
                    f"        → detect.py 의 TOP_MARGIN_CELLS 를 "
                    f"{math.ceil((need + 0.3) * 2) / 2:g} 이상으로 올리세요.\n"
                    "        (단 카메라가 그 영역을 실제로 봐야 합니다 — 탑뷰가 "
                    "검게 나오면 화각 밖입니다)")
        top = self._get_top_view(frame)
        px, py = int(pt[0]), int(pt[1])
        patch = top[max(0, py-3):py+4, max(0, px-3):px+4]
        if patch.size and float(patch.max()) < 20:
            return (head + "\n"
                    "     ③ 탑뷰 안이지만 그 자리가 새까맣습니다 — 카메라가 판 너머를\n"
                    "        아예 못 보고 있습니다(화각 밖). 소프트웨어로는 못 고칩니다.\n"
                    "        → 카메라를 뒤로/위로 옮기고 "
                    "python vision/auto_calibrate.py 를 다시 하세요.")
        return (head + "\n"
                "     ③ 탑뷰 안에 정상적으로 들어옵니다 → 면적 필터"
                f"({MARKER_MIN_AREA}~{MARKER_MAX_AREA}px)에 걸린 것입니다.")

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
        # ⚠️ 왜 못 찾았는지 남긴다. "not found" 만 띄우면 색이 안 맞는 건지
        #    덩어리가 너무 큰/작은 건지 구분이 안 돼 엉뚱한 데를 고치게 된다.
        raw_max = max((cv2.contourArea(c) for c in cnts), default=0.0)
        cnts = [c for c in cnts if cv2.contourArea(c) >= MARKER_MIN_AREA]
        if not cnts:
            self._last_marker_area = raw_max
            self._last_marker_fail = (
                f"색이 안 맞음 (범위에 든 픽셀 0)" if raw_max == 0 else
                f"덩어리가 너무 작음 {raw_max:.0f} < {MARKER_MIN_AREA}px")
            return None
        best = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area > MARKER_MAX_AREA:
            # 체스판 밝은 칸이나 조명 반사를 잡은 것 — 채도(S) 최소값을 올릴 것
            self._last_marker_area = area
            self._last_marker_fail = (
                f"덩어리가 너무 큼 {area:.0f} > {MARKER_MAX_AREA}px "
                "(판·반사를 잡는 중)")
            return None
        self._last_marker_fail = None
        self._last_marker_area = area
        M = cv2.moments(best)
        if M["m00"] == 0:
            return None
        px, py = M["m10"] / M["m00"], M["m01"] / M["m00"]
        # 화면에 '진짜 잡힌 자리'를 그리려고 픽셀 좌표도 남긴다.
        # (예전에는 칸 중앙에 원을 그려서, 마커가 칸 구석에 있어도 중앙에
        #  있는 것처럼 보였다 — 디버깅할 때 오해를 부른다)
        self._last_marker_px = (px, py)
        self._last_marker_cnt = best
        c, r = grid_uv_to_colrow(*px_to_grid_uv(px, py))
        # 마커가 보이는 위치 → 흡착컵의 실제 위치 (시차·오프셋 보정)
        return apply_marker_fit(c, r)

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
