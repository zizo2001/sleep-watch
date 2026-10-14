from __future__ import annotations

import cv2


class Camera:
    def __init__(self, index: int = 0, width: int = 1280,
                 height: int = 720, fps: int = 15):
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"camera index {self.index} did not open — is it connected? "
                f"try: sleep-watch test --camera 1")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Best effort — AVFoundation often ignores these; the lighting
        # guard + adaptive baseline absorb what remains.
        for prop, value in ((cv2.CAP_PROP_AUTO_EXPOSURE, 0.25),
                            (cv2.CAP_PROP_AUTO_WB, 0.0)):
            try:
                self.cap.set(prop, value)
            except cv2.error:
                pass

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
