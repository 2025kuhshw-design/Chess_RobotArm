"""
카메라 라이브 뷰 (백그라운드 스레드) + 프레임 공유 프록시.

왜 필요한가:
  · input() 은 블로킹이라 메인 스레드에서 OpenCV 이벤트 루프를 못 돌린다.
    그래서 "창을 띄워두고 명령을 계속 받는" 구조가 안 된다.
  · 카메라를 두 곳에서 read() 하면 프레임을 서로 뺏어간다.

해결:
  LiveView 스레드가 카메라 읽기를 **독점**하고 최신 프레임을 보관한다.
  기물 인식·마커 검출은 DetectorProxy 를 통해 그 프레임을 빌려 쓴다.
  뷰가 꺼져 있으면 프록시는 원래대로 detector 가 직접 읽게 넘긴다.

test_square.py 와 main.py 가 함께 쓴다.
"""

import threading
import time

DEFAULT_WINDOW = "Chess Vision"
FRAME_RETRY = 5          # 프록시가 최신 프레임으로 재시도할 횟수
RETRY_DELAY = 0.05


class LiveView:
    """카메라 창을 백그라운드로 계속 갱신한다."""

    def __init__(self, detector, window: str = DEFAULT_WINDOW):
        self.det = detector
        self.window = window
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._th = None

    # ── 수명 관리 ──
    def start(self):
        if self._th is not None:
            return
        self._stop.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        if self._th is None:
            return
        self._stop.set()
        self._th.join(timeout=2)
        self._th = None
        try:
            import cv2
            cv2.destroyWindow(self.window)
            cv2.waitKey(1)
        except Exception:
            pass

    @property
    def running(self) -> bool:
        return self._th is not None

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    # ── 스레드 본체 ──
    def _loop(self):
        import cv2
        while not self._stop.is_set():
            ok, f = self.det.cap.read()
            if not ok:
                continue
            with self._lock:
                self._frame = f
            try:
                cv2.imshow(self.window, self.det.overlay_debug(f))
                cv2.waitKey(30)
            except Exception:
                break      # GUI를 못 쓰는 환경이면 조용히 종료


class DetectorProxy:
    """라이브 뷰가 켜져 있으면 그 최신 프레임으로 인식하도록 감싼다.

    detector 의 나머지 속성/메서드는 그대로 통과시킨다(__getattr__).
    """

    def __init__(self, detector, view: LiveView = None):
        self.det = detector
        self.view = view

    def _borrow(self):
        """라이브 뷰의 최신 프레임. 뷰가 꺼져 있으면 None."""
        if self.view is not None and self.view.running:
            return self.view.latest()
        return None

    def find_marker(self):
        if self.view is not None and self.view.running:
            for _ in range(FRAME_RETRY):
                f = self._borrow()
                if f is not None:
                    r = self.det.find_marker(f)
                    if r is not None:
                        return r
                time.sleep(RETRY_DELAY)
            return None
        return self.det.find_marker()

    def get_board_state(self):
        return self.det.get_board_state(self._borrow())

    def piece_center(self, col, row, **kw):
        # __getattr__ 로 넘기면 detector 가 카메라를 직접 읽어 프레임을 뺏는다.
        # 라이브 뷰가 켜져 있을 땐 반드시 빌린 프레임을 넘겨야 한다.
        return self.det.piece_center(col, row, frame=self._borrow(), **kw)

    def snapshot_pieces(self, **kw):
        return self.det.snapshot_pieces(frame=self._borrow(), **kw)

    def get_stable_board_state(self, tries: int = 6):
        return self.det.get_stable_board_state(tries, frame_src=self._borrow_fresh)

    def _borrow_fresh(self):
        """안정화 판정용 — 매번 새 프레임을 준다(뷰가 계속 갱신하므로)."""
        return self._borrow()

    def detect_move_with_rules(self, board):
        # __getattr__ 로 넘기면 detector 가 카메라를 직접 읽어 프레임을 뺏는다.
        # 반드시 프록시의 프레임 공급자를 넘겨야 한다.
        return self.det.detect_move_with_rules(board, frame_src=self._borrow_fresh)

    def __getattr__(self, k):
        # ⚠️ __init__ 이전이나 det 가 없을 때 self.det 를 찾으면 다시
        #    __getattr__ 이 불려 무한 재귀에 빠진다. 명시적으로 끊는다.
        if k == "det":
            raise AttributeError(k)
        return getattr(self.det, k)
