"""Calibration task service: validation, dedup, calibration, concurrency.

All public operations on a task are serialised through a per-task lock, so
concurrent create/append/read/recalibrate calls are linearisable and the
final state is deterministic: the persisted frame set is exactly the set of
valid, first-committed frames, and the persisted result is always computed
from that full set (incremental recalibration is a full solve over the
current valid frames, hence identical to a from-scratch calibration).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .models import (
    CalibrationResult,
    FrameObservation,
    RejectedFrame,
    TaskConfig,
)
from .solver import calibrate
from .store import TaskExistsError, TaskStore

_MIN_POINTS_PER_FRAME = 4


def validate_frame(frame: FrameObservation) -> str | None:
    """Return a rejection reason, or ``None`` if the frame is valid."""
    if not frame.frame_id:
        return "empty frame_id"
    n_obj = len(frame.object_points)
    n_img = len(frame.image_points)
    if n_obj != n_img:
        return f"point count mismatch: {n_obj} object vs {n_img} image points"
    if n_obj < _MIN_POINTS_PER_FRAME:
        return f"too few points: {n_obj} < {_MIN_POINTS_PER_FRAME}"
    for p in frame.object_points:
        if len(p) != 3 or not all(math.isfinite(c) for c in p):
            return "non-finite or malformed object point"
    for p in frame.image_points:
        if len(p) != 2 or not all(math.isfinite(c) for c in p):
            return "non-finite or malformed image point"
    obj = np.asarray(frame.object_points, dtype=np.float64)
    # Board points must span a 2D plane area (not collinear) for the
    # homography initialisation to be well conditioned.
    centred = obj[:, :2] - obj[:, :2].mean(axis=0)
    if float(np.linalg.matrix_rank(centred, tol=1e-12)) < 2:
        return "object points are collinear"
    return None


class CalibrationService:
    """High-level API used by the HTTP layer and tests."""

    def __init__(self, root: str | Path):
        self.store = TaskStore(root)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # -- concurrency ---------------------------------------------------------

    def _lock_for(self, task_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(task_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[task_id] = lock
            return lock

    # -- task lifecycle --------------------------------------------------------

    def create_task(self, task_id: str, config: TaskConfig) -> None:
        """Create a task; concurrent duplicates raise :class:`TaskExistsError`."""
        with self._lock_for(task_id):
            self.store.create_task(task_id, config)

    # -- observations ------------------------------------------------------------

    def add_observations(
        self, task_id: str, frames: Sequence[FrameObservation]
    ) -> list[RejectedFrame]:
        """Validate and commit frames; returns the rejected ones.

        Out-of-order, duplicated or corrupted frames never corrupt the task
        state: duplicates (same ``frame_id`` already committed, in this batch
        or earlier) are ignored, invalid frames are rejected with a reason.
        The call is atomic with respect to other operations on the task.
        """
        rejected: list[RejectedFrame] = []
        with self._lock_for(task_id):
            committed = {f.frame_id for f in self.store.load_frames(task_id)}
            for frame in frames:
                reason = validate_frame(frame)
                if reason is not None:
                    rejected.append(RejectedFrame(frame.frame_id, reason))
                    continue
                if frame.frame_id in committed:
                    continue  # duplicate: applied exactly once, silently
                self.store.append_frame(task_id, frame)
                committed.add(frame.frame_id)
        return rejected

    # -- calibration ---------------------------------------------------------------

    def calibrate(self, task_id: str) -> CalibrationResult:
        """Run calibration on all valid committed frames and persist it.

        Recalibration after appending more frames is a full solve over the
        current valid set, so its result equals a from-scratch calibration
        of the same set.  Abandoned (rejected/duplicate) frames never take
        part because they were never committed.
        """
        with self._lock_for(task_id):
            config = self.store.load_config(task_id)
            frames = self.store.load_frames(task_id)
            result = calibrate(
                frames,
                config.image_width,
                config.image_height,
                config.solver,
            )
            self.store.save_result(task_id, result)
            return result

    def get_result(self, task_id: str) -> CalibrationResult | None:
        with self._lock_for(task_id):
            return self.store.load_result(task_id)

    def get_frames(self, task_id: str) -> list[FrameObservation]:
        with self._lock_for(task_id):
            return self.store.load_frames(task_id)
