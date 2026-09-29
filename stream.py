"""Video stream helpers for drone HTTP feeds and local cameras."""
from __future__ import annotations

import cv2


class FrameStream:
    def __init__(self, source: str | int):
        self.source = source
        self.capture = cv2.VideoCapture(source)
        self.index = 0
        if not self.capture.isOpened():
            self.close()
            raise ValueError(f"Не удалось открыть видеопоток: {source}")

    def read(self):
        ok, frame = self.capture.read()
        if not ok:
            return None
        self.index += 1
        return frame

    def skip(self, count: int) -> None:
        for _ in range(max(count, 0)):
            if not self.capture.grab():
                break
            self.index += 1

    def close(self) -> None:
        if getattr(self, "capture", None) is not None:
            self.capture.release()

    def __del__(self):
        self.close()


def parse_source(value: str) -> str | int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    return value
