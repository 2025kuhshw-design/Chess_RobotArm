"""
기물 인식 점검 — 카메라가 판을 어떻게 읽고 있는지 숫자로 확인한다.

두 가지를 한 번에 본다:
  1. 기물이 제대로 잡히는가 (OCC_DIFF_THRESH 가 맞는가)
  2. 판 방향이 맞는가 (ROBOT_SIDE 가 맞는가)

시작 배치(기물을 처음 놓은 상태)로 두고 실행하면, 맞을 때 이렇게 나온다:
    랭크 1·2 = 흰 기물, 랭크 7·8 = 검은 기물, 나머지 = 빈 칸

실행:
  python vision/check_board.py --camera 1
  python vision/check_board.py --camera 1 --thresh 12    # 임계값 바꿔 시험
  python vision/check_board.py --camera 1 --all-sides    # 네 방향 모두 비교
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import vision.detect as vd
from vision.detect import ChessBoardDetector, format_state


def start_position_state(fen=None):
    """대조할 배치를 8×8 배열로. fen 을 주면 그 국면, 없으면 표준 시작 배치."""
    import chess
    from vision.detect import board_to_state
    return board_to_state(chess.Board(fen) if fen else chess.Board())


def agreement(a, b) -> int:
    return sum(a[r][c] == b[r][c] for r in range(8) for c in range(8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--thresh", type=float, default=None,
                    help="OCC_DIFF_THRESH 를 이 값으로 바꿔 시험")
    ap.add_argument("--all-sides", action="store_true",
                    help="ROBOT_SIDE 네 가지를 모두 시험해 가장 맞는 것을 찾는다")
    ap.add_argument("--capture-empty", action="store_true",
                    help="⭐ 체스판을 완전히 비운 뒤 실행 — 빈 판 기준 영상을 찍는다")
    ap.add_argument("--frac", type=float, default=None,
                    help="OCC_AREA_FRAC 을 이 값으로 바꿔 시험 (기준 영상 방식)")
    ap.add_argument("--fen", type=str, default=None,
                    help="대조할 배치 (FEN). 기물이 32개가 아니면 실제로 놓은 "
                         "배치를 지정한다")
    ap.add_argument("--gamma", type=float, default=None,
                    help="감마를 이 값으로 바꿔 시험 (저장은 안 함). 기준 영상은 "
                         "찍을 때의 감마로 고정되므로 --capture-empty 와 같이 쓸 것")
    args = ap.parse_args()

    if args.gamma is not None:
        vd.set_gamma(args.gamma)
    if args.thresh is not None:
        vd.OCC_DIFF_THRESH = args.thresh
    if args.frac is not None:
        vd.OCC_AREA_FRAC = args.frac

    det = ChessBoardDetector(camera_index=args.camera)
    try:
        expect = start_position_state(args.fen)

        if args.capture_empty:
            print("\n⚠️ 체스판에 기물이 하나도 없어야 합니다.")
            print("   로봇팔은 park(Z자) 자세로 두세요 — 그림자까지 같이 기록됩니다.")
            input("   준비됐으면 Enter > ")
            det.capture_empty_reference()
            print("\n이제 기물을 시작 배치로 놓고 다시 실행하세요:")
            print(f"   python vision/check_board.py --camera {args.camera}")
            return

        if args.all_sides:
            print("\n시작 배치라고 가정하고 네 방향을 모두 시험합니다.\n")
            orig = vd.ROBOT_SIDE
            best = None
            for side in ("left", "right", "top", "bottom"):
                vd.ROBOT_SIDE = side
                obs = det.get_board_state(expect=expect)
                n = agreement(obs, expect)
                print(f"  ROBOT_SIDE = {side:6s} → 64칸 중 {n}칸 일치")
                if best is None or n > best[0]:
                    best = (n, side, obs)
            vd.ROBOT_SIDE = orig
            print(f"\n가장 잘 맞는 방향: ROBOT_SIDE = \"{best[1]}\" ({best[0]}/64)")
            if best[0] < 56:
                print("  ⚠️ 최고점도 낮습니다 → 방향 문제가 아니라 기물 자체를")
                print("     못 읽고 있을 가능성이 큽니다. --thresh 를 낮춰 보세요.")
            elif best[1] != orig:
                print(f"  → vision/detect.py 의 ROBOT_SIDE 를 \"{best[1]}\" 로 고치세요.")
            else:
                print("  → 현재 설정이 맞습니다.")
            print("\n그 방향으로 읽은 결과 (대문자=시작배치와 다른 칸):")
            print(format_state(best[2], expect))
            return

        obs = det.get_board_state(expect=expect)
        print(f"\nROBOT_SIDE = \"{vd.ROBOT_SIDE}\"   판정 방식: {det.detection_mode()}")
        print("\n카메라가 읽은 배치 (대문자=시작배치와 다른 칸):")
        print(format_state(obs, expect))
        print(f"\n시작 배치와 {agreement(obs, expect)}/64 칸 일치")
        print()
        print(det.explain_board_state())
        print("\n판단 요령:")
        if "색" in det.detection_mode():
            print("  · 기물 있는 칸의 비율이 낮다 → detect.py 의 "
                  "PIECE_COLOR_MIN_FRAC 을 낮춘다")
            print("  · 실제로 놓은 배치와 --fen 이 같은지 먼저 확인할 것 "
                  "(32개면 --fen 을 아예 빼면 된다)")
        elif det._ref is not None:
            print("  · 기물 있는 칸의 비율이 낮다 → --frac 0.10 처럼 낮춘다")
            print("  · 빈 칸이 기물로 잡힌다      → --frac 0.25 처럼 올린다")
        else:
            print("  ⚠️ 빈 판 기준 영상이 없습니다. 밝기만으로는 검은 기물과")
            print("     어두운 칸을 구분할 수 없습니다. 먼저 이것부터 하세요:")
            print(f"       python vision/check_board.py --camera {args.camera} "
                  "--capture-empty")
        print("  · 기물은 잡히는데 엉뚱한 칸에 있다 → --all-sides 로 방향 확인")
    finally:
        det.close()


if __name__ == "__main__":
    main()
