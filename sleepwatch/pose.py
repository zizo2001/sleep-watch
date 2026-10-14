from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

JOINT_COLOR = (255, 200, 80)      # BGR — cool blue-ish joints
BONE_COLOR = (230, 150, 60)


class PoseOverlay:
    """MediaPipe Tasks pose skeleton, drawn onto BGR frames.

    Degrades gracefully: no model file -> available=False and annotate()
    returns the frame untouched. Inference runs every `every_n` calls;
    the last landmarks are reused in between.
    """

    def __init__(self, model_path: Path | None, every_n: int = 3):
        self.every_n = max(1, every_n)
        self._landmarker = None
        self._connections = None
        self._last_landmarks = None
        self._calls = 0
        if model_path is None or not Path(model_path).exists():
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1)
            self._landmarker = vision.PoseLandmarker.create_from_options(
                options)
            self._connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
            self._mp = mp
        except Exception:
            import traceback
            traceback.print_exc()
            self._landmarker = None

    @property
    def available(self) -> bool:
        return self._landmarker is not None

    @property
    def last_present(self) -> bool | None:
        """Person visible in the most recent annotated frame; None = no pose."""
        if self._landmarker is None:
            return None
        return self._last_landmarks is not None

    @property
    def last_box(self) -> tuple[float, float, float, float] | None:
        """Normalized (x, y, w, h) around the last-seen person, else None."""
        if self._landmarker is None or self._last_landmarks is None:
            return None
        xs = [lm.x for lm in self._last_landmarks]
        ys = [lm.y for lm in self._last_landmarks]
        x0, y0 = max(0.0, min(xs)), max(0.0, min(ys))
        x1, y1 = min(1.0, max(xs)), min(1.0, max(ys))
        return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))

    def annotate(self, frame_bgr: np.ndarray, ts_ms: int) -> np.ndarray:
        if self._landmarker is None:
            return frame_bgr
        self._calls += 1
        if self._calls % self.every_n == 1 or self.every_n == 1:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                   data=rgb)
            try:
                result = self._landmarker.detect_for_video(image, int(ts_ms))
                self._last_landmarks = (result.pose_landmarks[0]
                                        if result.pose_landmarks else None)
            except Exception:
                self._last_landmarks = None
        if self._last_landmarks is None:
            return frame_bgr
        return self._draw(frame_bgr, self._last_landmarks)

    def _draw(self, frame: np.ndarray, landmarks) -> np.ndarray:
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h),
                getattr(lm, "visibility", 1.0) or 1.0)
               for lm in landmarks]
        out = frame
        for c in self._connections:
            if c.start >= len(pts) or c.end >= len(pts):
                continue
            x1, y1, v1 = pts[c.start]
            x2, y2, v2 = pts[c.end]
            if min(v1, v2) < 0.5:
                continue
            cv2.line(out, (x1, y1), (x2, y2), BONE_COLOR, 2, cv2.LINE_AA)
        for x, y, v in pts:
            if v < 0.5:
                continue
            cv2.circle(out, (x, y), 3, JOINT_COLOR, -1, cv2.LINE_AA)
        return out

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
