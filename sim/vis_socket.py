"""
학습 시각화용 UDP 소켓 송신기
Unreal Engine (또는 다른 뷰어)으로 관절각·목표 위치를 실시간 전송.

프로토콜 (JSON over UDP, localhost:9001):
  {"q": [q1, q2, q3], "target": [tx, ty, tz], "step": N}

UE가 꺼져 있으면 sendto가 즉시 반환되므로 학습 속도에 영향 없음.
"""

import socket
import json
import numpy as np

VIS_PORT = 9001
VIS_HOST = "127.0.0.1"


class VisSocket:
    def __init__(self, host: str = VIS_HOST, port: int = VIS_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, port)

    def send(self, q: np.ndarray, target: np.ndarray, step: int = 0):
        payload = json.dumps({
            "q":      [round(float(v), 4) for v in q],
            "target": [round(float(v), 4) for v in target],
            "step":   step,
        }).encode()
        try:
            self._sock.sendto(payload, self._addr)
        except Exception:
            pass  # UE 미연결 시 무시

    def close(self):
        self._sock.close()
