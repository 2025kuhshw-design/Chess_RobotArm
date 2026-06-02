"""
현장 배치 구조도 생성 스크립트
ik_solver.py의 실제 좌표 상수를 사용해 탑뷰 + 사이드뷰 배치도를 그린다.
실행: python docs/make_layout.py  →  docs/layout_diagram.png
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Wedge, Circle, Polygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.ik_solver import (BOARD_ORIGIN_X, BOARD_ORIGIN_Y, CELL_SIZE,
                             PIECE_Z, L1_DEFAULT, L2_DEFAULT, L3_DEFAULT)

BOARD = 8 * CELL_SIZE
REACH_MAX = L1_DEFAULT + L2_DEFAULT
# 기물 높이에서의 수평 도달 한계: sqrt(REACH^2 - (z+L3)^2)
R_PIECE = math.sqrt(max(REACH_MAX**2 - (PIECE_Z + L3_DEFAULT)**2, 0))

fig = plt.figure(figsize=(16, 8))

# ══════════════════════════════════════════════════════════
# (1) TOP VIEW  —  수평축=world Y(좌우, +Y=왼쪽), 수직축=world X(전방)
# ══════════════════════════════════════════════════════════
ax = fig.add_subplot(1, 2, 1)
ax.set_title("TOP VIEW  (bird's eye, units = m)", fontsize=13, fontweight="bold")

# 도달 범위 부채꼴 (베이스 중심, 전방 반원)
ax.add_patch(Wedge((0, 0), R_PIECE, 0, 180, width=R_PIECE,
                   facecolor="#d8ecff", edgecolor="none", alpha=0.5, zorder=0))
ax.add_patch(Wedge((0, 0), R_PIECE, 0, 180, width=0.001,
                   facecolor="none", edgecolor="#3a7bd5", lw=1.5,
                   linestyle="--", zorder=1))
ax.text(0, R_PIECE + 0.008, f"reach limit r={R_PIECE*100:.1f} cm",
        ha="center", color="#3a7bd5", fontsize=8)

# 8x8 보드 (world X=col? 주의: col→x, row→y)
for col in range(8):
    for row in range(8):
        cx = BOARD_ORIGIN_X + col * CELL_SIZE          # world x (forward)
        cy = BOARD_ORIGIN_Y + row * CELL_SIZE          # world y (left/right)
        # plot: 수평=-y(왼쪽이 +y), 수직=x
        px, py = -(cy + CELL_SIZE / 2) * 0 - cy, cx    # placeholder
        # 사각형 그리기 (plot좌표: X축=-world_y, Y축=world_x)
        light = (col + row) % 2 == 0
        ax.add_patch(Rectangle((-cy - CELL_SIZE, cx), CELL_SIZE, CELL_SIZE,
                               facecolor="#f0d9b5" if light else "#b58863",
                               edgecolor="#555", lw=0.5, zorder=2))
        notation = f"{chr(ord('a')+col)}{row+1}"
        ax.text(-cy - CELL_SIZE/2, cx + CELL_SIZE/2, notation,
                ha="center", va="center", fontsize=6, color="#333", zorder=3)

# a1 강조 (col0,row0)
a1x = BOARD_ORIGIN_X
a1y = BOARD_ORIGIN_Y
ax.add_patch(Rectangle((-a1y - CELL_SIZE, a1x), CELL_SIZE, CELL_SIZE,
                       facecolor="none", edgecolor="magenta", lw=2.5, zorder=4))
ax.text(-a1y - CELL_SIZE/2, a1x + CELL_SIZE/2, "a1",
        ha="center", va="center", fontsize=8, color="magenta",
        fontweight="bold", zorder=5)

# 로봇 베이스
ax.add_patch(Circle((0, 0), 0.012, facecolor="#222", zorder=6))
ax.text(0, -0.022, "ROBOT BASE\n(0, 0)", ha="center", va="top",
        fontsize=9, fontweight="bold")

# 캡처 기물 임시 구역 (far edge 너머 +x 방향)
dump_x0 = BOARD_ORIGIN_X + BOARD
ax.add_patch(Rectangle((-BOARD_ORIGIN_Y - BOARD, dump_x0), BOARD, 0.05,
                       facecolor="#ffe0e0", edgecolor="#cc6666",
                       lw=1, linestyle=":", zorder=2))
ax.text(-(BOARD_ORIGIN_Y + BOARD/2), dump_x0 + 0.025, "captured-piece zone",
        ha="center", va="center", fontsize=7, color="#aa4444")

# 카메라 (탑뷰: 보드 중앙 바로 위) — 마커는 중앙, 설명은 우측 여백
camx = BOARD_ORIGIN_X + BOARD/2
camy = BOARD_ORIGIN_Y + BOARD/2
ax.add_patch(Circle((-camy, camx), 0.011, facecolor="#2c8", edgecolor="k",
                    lw=1.2, zorder=7))
ax.text(-camy, camx, "CAM", ha="center", va="center", fontsize=6,
        fontweight="bold", color="white", zorder=8)
ax.annotate("camera directly\noverhead\n(looking straight down)",
            xy=(-camy, camx), xytext=(-0.165, 0.20),
            fontsize=7.5, color="#1a8a5a", ha="center",
            arrowprops=dict(arrowstyle="->", color="#2c8", lw=1.2))

# 축 화살표
ax.add_patch(FancyArrowPatch((0, 0), (0, 0.06),
             arrowstyle="->", mutation_scale=14, color="red", lw=1.5))
ax.text(0.004, 0.06, "+X (forward)", color="red", fontsize=8, va="center")
ax.add_patch(FancyArrowPatch((0, 0), (0.06, 0),
             arrowstyle="->", mutation_scale=14, color="green", lw=1.5))
ax.text(0.06, -0.006, "+Y (robot left)", color="green", fontsize=8, ha="left")

ax.set_xlim(0.16, -0.20)   # 수평축: 왼쪽이 +Y (반전)
ax.set_ylim(-0.05, 0.34)
ax.set_aspect("equal")
ax.set_xlabel("world  -Y  →   (left)")
ax.set_ylabel("world  X  (forward, away from robot)")
ax.grid(alpha=0.2)

# ══════════════════════════════════════════════════════════
# (2) SIDE VIEW  —  배치/장비 연결 개념도
# ══════════════════════════════════════════════════════════
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title("SIDE VIEW  (setup & wiring concept)", fontsize=13, fontweight="bold")

# 테이블
ax2.add_patch(Rectangle((0.0, 0.0), 1.0, 0.04, facecolor="#caa472",
                        edgecolor="#8a6d3b"))
ax2.text(0.5, -0.03, "TABLE", ha="center", fontsize=8, color="#8a6d3b")

# 보드 (테이블 위)
bx0, bw = 0.30, 0.34
ax2.add_patch(Rectangle((bx0, 0.04), bw, 0.012, facecolor="#e8d0a0",
                        edgecolor="#555"))
ax2.text(bx0 + bw - 0.01, 0.058, "chessboard 23.3 cm", ha="right", va="bottom",
         fontsize=7, color="#555")

# 로봇팔 (간단 모식: 베이스 기둥 + 2링크 + 흡착)
base_x = 0.20
ax2.add_patch(Rectangle((base_x-0.02, 0.04), 0.04, 0.14,
                        facecolor="#444", edgecolor="k"))   # 베이스 기둥
j1 = np.array([base_x, 0.18])
j2 = j1 + np.array([0.10, 0.06])
j3 = j2 + np.array([0.12, -0.10])
ee = j3 + np.array([0.0, -0.045])
for a, b, c in [(j1, j2, "#1f77b4"), (j2, j3, "#2ca02c")]:
    ax2.plot([a[0], b[0]], [a[1], b[1]], c, lw=5, solid_capstyle="round")
ax2.plot([j3[0], ee[0]], [j3[1], ee[1]], "#d62728", lw=4)
for jp in (j1, j2, j3):
    ax2.add_patch(Circle(jp, 0.012, facecolor="gold", edgecolor="k", zorder=5))
ax2.add_patch(Circle(ee, 0.010, facecolor="#d62728", edgecolor="k", zorder=5))
ax2.text(ee[0], ee[1]-0.02, "suction\ncup", ha="center", va="top", fontsize=6)
ax2.text(base_x, 0.20, "3x MG996R", ha="center", fontsize=6, color="#444")

# 카메라 붐(스탠드) + FOV
cam = np.array([bx0 + bw/2, 0.52])
ax2.add_patch(Rectangle((0.92, 0.04), 0.03, 0.50, facecolor="#888",
                        edgecolor="k"))   # 수직 스탠드 기둥
ax2.plot([0.935, cam[0]], [0.54, cam[1]], color="#888", lw=4)  # 붐
ax2.add_patch(Rectangle((cam[0]-0.03, cam[1]), 0.06, 0.035,
                        facecolor="#2c8", edgecolor="k"))
ax2.text(cam[0], cam[1]+0.05, "CAMERA", ha="center", fontsize=7,
         color="#2c8", fontweight="bold")
# FOV 삼각형 (아래로)
fov = Polygon([(cam[0], cam[1]), (bx0, 0.052), (bx0+bw, 0.052)],
              closed=True, facecolor="#2c8", alpha=0.12, edgecolor="#2c8",
              linestyle="--", lw=1)
ax2.add_patch(fov)
ax2.annotate("", xy=(cam[0]-0.16, 0.30), xytext=(cam[0]-0.16, 0.52),
             arrowprops=dict(arrowstyle="<->", color="#666"))
ax2.text(cam[0]-0.155, 0.41, "40~60 cm\n(all 4 corners\nin frame)",
         fontsize=6.5, color="#666", va="center")

# 조명
ax2.text(0.5, 0.62, "diffuse lighting from above (avoid glare/shadows)",
         ha="center", fontsize=7.5, color="#cc9900",
         bbox=dict(boxstyle="round", fc="#fff7d6", ec="#cc9900"))

# 장비 박스 (좌측 하단): 노트북 / 아두이노+PCA / 전원 / 펌프
boxes = [
    (0.01, 0.30, "LAPTOP\n(Windows)\nmain.py", "#dfe9ff"),
    (0.01, 0.20, "Arduino Uno\n+ PCA9685", "#e8ffe0"),
    (0.01, 0.10, "6V 8A PSU\n(servo power)", "#ffe8d0"),
    (0.01, 0.00, "vacuum pump\n+ valve", "#f0e0ff"),
]
for bxx, byy, txt, col in boxes:
    ax2.add_patch(Rectangle((bxx, byy), 0.13, 0.075, facecolor=col,
                            edgecolor="#555"))
    ax2.text(bxx+0.065, byy+0.037, txt, ha="center", va="center", fontsize=6)
# 연결선
ax2.annotate("USB", xy=(0.14, 0.235), xytext=(0.14, 0.31),
             arrowprops=dict(arrowstyle="->"), fontsize=6, ha="left")
ax2.annotate("I2C+5V", xy=(0.19, 0.18), xytext=(0.14, 0.22),
             arrowprops=dict(arrowstyle="->"), fontsize=6)
ax2.annotate("6V", xy=(0.16, 0.155), xytext=(0.14, 0.13),
             arrowprops=dict(arrowstyle="->"), fontsize=6)

ax2.set_xlim(-0.02, 1.0)
ax2.set_ylim(-0.06, 0.68)
ax2.set_aspect("equal")
ax2.axis("off")

fig.suptitle("Chess Robot Arm — On-site Layout", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])

out = os.path.join(os.path.dirname(__file__), "layout_diagram.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved: {out}")
