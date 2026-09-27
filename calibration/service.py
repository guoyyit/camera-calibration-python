"""Calibration task service: validation, locking, orchestration.

Concurrency model
-----------------
All operations on a task are serialised through a per-task re-entrant
lock, so concurrent create / append / read / recalibrate calls on the
same task are linearised: each operation observes and leaves the store
in a state produced by a single global order of operations.  Operations
on different tasks proceed in parallel.

Frame lifecycle
---------------
First submission wins: the first record stored for a ``frame_id``
determines its fate.  Re-submissions are reported as DUPLICATE and never
re-applied, which keeps retries idempotent and results deterministic.
Frames failing validation are recorded as CORRUPT and excluded from
calibration, as are ABANDONED frames.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from . import image_ops
from .models import (
    CalibrationConfig,
    CalibrationResult,
    FrameObservation,
    FrameStatus,
    TaskInfo,
    TaskStatus,
)
from .solver import calibrate as _solve
from .storage import TaskStore

_MIN_POINTS = 4
# Singular-value ratio below which a point set is considered degenerate
# (collinear), i.e. the frame is corrupt and cannot constrain a homography.
_DEGENERATE_SV_RATIO = 1e-8


def _validate_frame(
    frame_id: str,
    object_points: np.ndarray,
    image_points: np.ndarray,
    image_size: tuple[int, int],
) -> str | None:
    """Return None if the frame is valid, else a human-readable reason."""
    obj = np.asarray(object_points, dtype=np.float64)
    img = np.asarray(image_points, dtype=np.float64)
    if obj.ndim != 2 or obj.shape[1] != 2 or img.ndim != 2 or img.shape[1] != 2:
        return "points must have shape (N, 2)"
    if obj.shape[0] != img.shape[0]:
        return "object and image point counts differ"
    if obj.shape[0] < _MIN_POINTS:
        return f"need at least {_MIN_POINTS} points"
    if not (np.all(np.isfinite(obj)) and np.all(np.isfinite(img))):
        return "points contain NaN or infinite values"
    w, h = image_size
    if w <= 0 or h <= 0:
        return "image size must be positive"
    for pts, name in ((obj, "object"), (img, "image")):
        centered = pts - pts.mean(axis=0)
        sv = np.linalg.svd(centered, compute_uv=False)
        if sv[-1] <= _DEGENERATE_SV_RATIO * max(sv[0], 1e-300):
            return f"{name} points are collinear/degenerate"
    return None


class CalibrationService:
    """Thread-safe facade over the store and solver."""

    def __init__(self, db_path: str | Path) -> None:
        self._store = TaskStore(db_path)
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def close(self) -> None:
        self._store.close()

    def _lock_for(self, task_id: str) -> threading.RLock:
        with self._locks_guard:
            lock = self._locks.get(task_id)
            if lock is None:
                lock = threading.RLock()
                self._locks[task_id] = lock
            return lock

    # ------------------------------------------------------------- tasks
    def create_task(
        self, task_id: str, config: CalibrationConfig | None = None
    ) -> TaskInfo:
        """Create a task (idempotent: concurrent creates linearise)."""
        if config is None:
            config = CalibrationConfig()
        with self._lock_for(task_id):
            self._store.create_task(task_id, config)
            return self.get_task(task_id)

    def get_task(self, task_id: str) -> TaskInfo:
        with self._lock_for(task_id):
            if not self._store.task_exists(task_id):
                raise KeyError(f"unknown task {task_id!r}")
            result = self._store.load_result(task_id)
            n = len(self._store.load_valid_frames(task_id))
            return TaskInfo(
                task_id=task_id,
                status=self._store.task_status(task_id),
                frame_count=n,
                result=result,
            )

    # ------------------------------------------------------------- frames
    def submit_frame(
        self,
        task_id: str,
        frame_id: str,
        object_points: np.ndarray,
        image_points: np.ndarray,
        image_size: tuple[int, int],
    ) -> FrameStatus:
        """Validate and record one frame; returns its resulting status."""
        with self._lock_for(task_id):
            if not self._store.task_exists(task_id):
                raise KeyError(f"unknown task {task_id!r}")
            reason = _validate_frame(frame_id, object_points, image_points, image_size)
            if reason is not None:
                obs = None
                status = FrameStatus.CORRUPT
            else:
                obs = FrameObservation(
                    frame_id=frame_id,
                    object_points=np.asarray(object_points, dtype=np.float64),
                    image_points=np.asarray(image_points, dtype=np.float64),
                    image_size=(int(image_size[0]), int(image_size[1])),
                )
                status = FrameStatus.ACCEPTED
            inserted = self._store.add_frame(task_id, frame_id, status, obs)
            if not inserted:
                return FrameStatus.DUPLICATE
            return status

    def abandon_frame(self, task_id: str, frame_id: str) -> bool:
        """Exclude a frame from future calibrations. False if unknown."""
        with self._lock_for(task_id):
            if not self._store.task_exists(task_id):
                raise KeyError(f"unknown task {task_id!r}")
            return self._store.set_frame_status(
                task_id, frame_id, FrameStatus.ABANDONED
            )

    # --------------------------------------------------------- calibration
    def calibrate(
        self, task_id: str, config: CalibrationConfig | None = None
    ) -> CalibrationResult:
        """Run calibration on all currently ACCEPTED frames.

        The whole read-frames / solve / save-result sequence runs under
        the task lock, so it linearises against concurrent appends: the
        result reflects exactly the frames accepted at calibration time.
        """
        with self._lock_for(task_id):
            if not self._store.task_exists(task_id):
                raise KeyError(f"unknown task {task_id!r}")
            if config is None:
                config = self._store.load_config(task_id)
            frames = self._store.load_valid_frames(task_id)
            result = _solve(frames, config)
            self._store.save_result(task_id, result)
            return result

    def get_result(self, task_id: str) -> CalibrationResult:
        with self._lock_for(task_id):
            result = self._store.load_result(task_id)
            if result is None:
                raise KeyError(f"task {task_id!r} has no calibration result")
            return result

    # ------------------------------------------------------------ mapping
    def undistort_points(self, task_id: str, points: np.ndarray) -> np.ndarray:
        return image_ops.undistort_points(points, self.get_result(task_id))

    def distort_points(self, task_id: str, points: np.ndarray) -> np.ndarray:
        return image_ops.distort_points(points, self.get_result(task_id))

    def undistort_image(
        self, task_id: str, image: np.ndarray, border_value: float = 0.0
    ) -> np.ndarray:
        return image_ops.undistort_image(
            image, self.get_result(task_id), border_value=border_value
        )
