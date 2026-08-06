"""
체스 유틸리티 모듈
- Stockfish 연동
- 이동 좌표 변환
- 기물 잡기 감지
"""

import chess
import chess.engine
import os

# ─────────────────────────────────────────
# 하이퍼파라미터 / 상수
# ─────────────────────────────────────────
DEFAULT_STOCKFISH_PATH = "/usr/games/stockfish"   # Linux 기본 경로
DEFAULT_TIME_LIMIT     = 1.0                       # Stockfish 탐색 시간 (초)


# ─────────────────────────────────────────
# 함수 1: Stockfish 최선의 수 반환
# ─────────────────────────────────────────
def stockfish_move(board: chess.Board,
                   stockfish_path: str = DEFAULT_STOCKFISH_PATH,
                   time_limit: float = DEFAULT_TIME_LIMIT) -> chess.Move:
    """
    현재 보드 상태 → Stockfish 최선의 수(chess.Move) 반환.
    stockfish_path: Stockfish 실행 파일 경로
    """
    if not os.path.exists(stockfish_path):
        raise FileNotFoundError(
            f"Stockfish 실행 파일을 찾을 수 없습니다: {stockfish_path}\n"
            "  Linux: sudo apt install stockfish\n"
            "  Windows: stockfishchess.org 에서 .exe 다운로드 후 경로 지정\n"
            "  실행 시: python main.py --stockfish /경로/stockfish"
        )

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        result = engine.play(board, chess.engine.Limit(time=time_limit))

    return result.move


# ─────────────────────────────────────────
# 함수 2: chess.Move → (from_col, from_row), (to_col, to_row)
# ─────────────────────────────────────────
def move_to_squares(move: chess.Move) -> tuple:
    """
    chess.Move → ((from_col, from_row), (to_col, to_row))
    chess 표기법: 파일 a-h → col 0-7 / 랭크 1-8 → row 0-7
    """
    from_sq = move.from_square
    to_sq   = move.to_square

    from_col = chess.square_file(from_sq)   # 0(a) ~ 7(h)
    from_row = chess.square_rank(from_sq)   # 0(1랭크) ~ 7(8랭크)
    to_col   = chess.square_file(to_sq)
    to_row   = chess.square_rank(to_sq)

    return (from_col, from_row), (to_col, to_row)


# ─────────────────────────────────────────
# 함수 3: 기물 잡기 여부 확인
# ─────────────────────────────────────────
def is_capture(board: chess.Board, move: chess.Move) -> bool:
    """
    해당 수가 기물 잡기인지 확인.
    True면 도착 칸 기물을 먼저 제거하는 동작 필요.
    앙파상(en passant)도 포함.
    """
    return board.is_capture(move)


# ─────────────────────────────────────────
# 함수 4: 수 하나 → 팔이 실제로 해야 할 동작 목록
# ─────────────────────────────────────────
def physical_ops(board: chess.Board, move: chess.Move) -> list:
    """체스 규칙상의 '한 수'를 **팔이 실제로 해야 할 동작들**로 분해한다.

    ⚠️ 왜 필요한가 — 규칙상 한 수라도 물리적으로는 여러 동작인 경우가 있다.
       예전에는 무조건 'from 칸 기물을 to 칸으로 옮긴다' 하나만 했다:
         · 캐슬링 : 킹만 옮기고 **룩이 그대로 남았다**
         · 앙파상 : 도착 칸이 비어 있는데 거기서 기물을 치우려 했고,
                    정작 잡히는 폰은 다른 칸이라 **그대로 남았다**
         · 승진   : 판 위엔 폰인데 엔진은 퀸으로 알아 이후가 전부 어긋난다
       그러면 엔진이 아는 판과 실제 판이 달라져 게임이 통째로 망가진다.

    반환: 순서대로 실행할 동작 목록
        ("clear", (col,row))            그 칸 기물을 보드 밖 캡처 구역으로
        ("move",  (col,row), (col,row)) 집어서 옮긴다
    두 번째 반환값: 사람에게 알릴 경고 문자열 목록 (승진 등)
    """
    def cr(sq):
        return (chess.square_file(sq), chess.square_rank(sq))

    ops, notes = [], []

    # 1) 잡히는 기물 먼저 치운다 — 앙파상은 도착 칸이 아니라 옆 칸이다
    if board.is_en_passant(move):
        cap_sq = chess.square(chess.square_file(move.to_square),
                              chess.square_rank(move.from_square))
        ops.append(("clear", cr(cap_sq)))
        notes.append(f"앙파상: {chess.square_name(cap_sq)} 의 폰을 치웁니다")
    elif board.is_capture(move):
        ops.append(("clear", cr(move.to_square)))

    # 2) 기물 이동
    ops.append(("move", cr(move.from_square), cr(move.to_square)))

    # 3) 캐슬링이면 룩도 옮겨야 한다
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            rook_from, rook_to = chess.square(7, rank), chess.square(5, rank)   # 킹사이드
        else:
            rook_from, rook_to = chess.square(0, rank), chess.square(3, rank)   # 퀸사이드
        ops.append(("move", cr(rook_from), cr(rook_to)))
        notes.append(f"캐슬링: 룩 {chess.square_name(rook_from)}"
                     f"→{chess.square_name(rook_to)} 도 함께 옮깁니다")

    # 4) 승진은 물리적으로 대신할 수 없다 — 사람이 기물을 바꿔줘야 한다
    if move.promotion:
        name = chess.piece_name(move.promotion)
        notes.append(f"⚠️ 승진: {chess.square_name(move.to_square)} 의 기물을 "
                     f"사람이 {name} 으로 바꿔주세요 (로봇은 못 바꿉니다)")

    return ops, notes


# ─────────────────────────────────────────
# 단독 실행: 기본 동작 테스트
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=== chess_utils 동작 테스트 ===")

    board = chess.Board()
    print(f"초기 보드:\n{board}\n")

    # move_to_squares 테스트 (e2→e4)
    move = chess.Move.from_uci("e2e4")
    from_sq, to_sq = move_to_squares(move)
    print(f"이동: e2→e4")
    print(f"  from_square: col={from_sq[0]}, row={from_sq[1]}  (e=4, 2랭크=row1)")
    print(f"  to_square:   col={to_sq[0]}, row={to_sq[1]}    (e=4, 4랭크=row3)")

    # is_capture 테스트
    print(f"\ne2→e4 기물 잡기 여부: {is_capture(board, move)}")   # False

    # Stockfish 테스트 (실행 파일 없으면 경고만)
    print("\nStockfish 연동 테스트...")
    sf_path = DEFAULT_STOCKFISH_PATH
    try:
        best = stockfish_move(board, sf_path, time_limit=0.5)
        from_sq2, to_sq2 = move_to_squares(best)
        print(f"  Stockfish 추천 수: {best} | from={from_sq2} to={to_sq2}")
    except FileNotFoundError as e:
        print(f"  [경고] {e}")
        print("  → main.py 실행 시 --stockfish 옵션으로 경로 지정 필요")

    print("\n✅ chess_utils 완료")
