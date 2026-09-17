"""Shared OpenCV video output used by radar and multi-sensor modes."""

from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    """Lazily open a fixed-size MP4 writer from its first frame."""

    def __init__(self, output_path, fps):
        self.output_path = Path(output_path).expanduser()
        self.fps = float(fps)
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {fps!r}")
        self._writer = None
        self._frame_size = None

    def write(self, frame):
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Video frame must have shape [H, W, 3], got {frame.shape}"
            )
        height, width = frame.shape[:2]
        frame_size = (int(width), int(height))
        if self._writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                frame_size,
            )
            if not self._writer.isOpened():
                self._writer.release()
                self._writer = None
                raise RuntimeError(
                    f"Cannot open video writer: {self.output_path}"
                )
            self._frame_size = frame_size
        elif frame_size != self._frame_size:
            raise ValueError(
                f"Video frame size changed from {self._frame_size} to "
                f"{frame_size}"
            )
        self._writer.write(frame)

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
