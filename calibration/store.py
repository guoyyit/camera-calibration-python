"""Local persistent storage for calibration tasks.

Layout on disk (one directory per task under the store root)::

    <root>/<task_id>/meta.json          task configuration (immutable)
    <root>/<task_id>/frames/NNNNNN.json one committed frame per file
    <root>/<task_id>/result.json        last committed calibration result

Durability model
----------------
* Every mutation is written to a temporary file and atomically renamed into
  place (``os.replace``), so a crash can never leave a partially written
  file — either the old or the new state is visible.
* Frame files are named with a zero-padded sequence number; recovery scans
  the directory and replays committed files in sequence order, ignoring
  leftover temporary files.
* Appending a frame that is already stored (same ``frame_id``) is a no-op,
  so observations are never applied twice.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .models import (
    CalibrationResult,
    CameraIntrinsics,
    FrameError,
    FrameObservation,
    SolverConfig,
    TaskConfig,
)


class TaskNotFoundError(KeyError):
    """Raised when a task id does not exist in the store."""


class TaskExistsError(RuntimeError):
    """Raised when creating a task that already exists."""


# ---------------------------------------------------------------------------
# (de)serialisation helpers
# ---------------------------------------------------------------------------


def _config_to_dict(config: TaskConfig) -> dict:
    return {
        "image_width": config.image_width,
        "image_height": config.image_height,
        "solver": {
            "max_iterations": config.solver.max_iterations,
            "convergence_tolerance": config.solver.convergence_tolerance,
            "initial_damping": config.solver.initial_damping,
        },
    }


def _config_from_dict(data: dict) -> TaskConfig:
    solver = data["solver"]
    return TaskConfig(
        image_width=int(data["image_width"]),
        image_height=int(data["image_height"]),
        solver=SolverConfig(
            max_iterations=int(solver["max_iterations"]),
            convergence_tolerance=float(solver["convergence_tolerance"]),
            initial_damping=float(solver["initial_damping"]),
        ),
    )


def _frame_to_dict(frame: FrameObservation) -> dict:
    return {
        "frame_id": frame.frame_id,
        "object_points": [list(p) for p in frame.object_points],
        "image_points": [list(p) for p in frame.image_points],
    }


def _frame_from_dict(data: dict) -> FrameObservation:
    return FrameObservation(
        frame_id=str(data["frame_id"]),
        object_points=tuple(
            (float(p[0]), float(p[1]), float(p[2])) for p in data["object_points"]
        ),
        image_points=tuple(
            (float(p[0]), float(p[1])) for p in data["image_points"]
        ),
    )


def _result_to_dict(result: CalibrationResult) -> dict:
    intr = result.intrinsics
    return {
        "intrinsics": {
            "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
            "k1": intr.k1, "k2": intr.k2, "p1": intr.p1, "p2": intr.p2,
        },
        "frame_errors": [
            {"frame_id": fe.frame_id, "mean_error": fe.mean_error,
             "max_error": fe.max_error}
            for fe in result.frame_errors
        ],
        "mean_reprojection_error": result.mean_reprojection_error,
        "iterations": result.iterations,
        "converged": result.converged,
        "used_frame_ids": list(result.used_frame_ids),
    }


def _result_from_dict(data: dict) -> CalibrationResult:
    intr = data["intrinsics"]
    return CalibrationResult(
        intrinsics=CameraIntrinsics(
            fx=float(intr["fx"]), fy=float(intr["fy"]),
            cx=float(intr["cx"]), cy=float(intr["cy"]),
            k1=float(intr["k1"]), k2=float(intr["k2"]),
            p1=float(intr["p1"]), p2=float(intr["p2"]),
        ),
        frame_errors=tuple(
            FrameError(frame_id=str(fe["frame_id"]),
                       mean_error=float(fe["mean_error"]),
                       max_error=float(fe["max_error"]))
            for fe in data["frame_errors"]
        ),
        mean_reprojection_error=float(data["mean_reprojection_error"]),
        iterations=int(data["iterations"]),
        converged=bool(data["converged"]),
        used_frame_ids=tuple(str(i) for i in data["used_frame_ids"]),
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` as JSON atomically: temp file + fsync + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TaskStore:
    """Filesystem-backed store for calibration tasks."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    # -- paths ---------------------------------------------------------------

    def _task_dir(self, task_id: str) -> Path:
        if not task_id or task_id in (".", "..") or "/" in task_id or "\\" in task_id:
            raise ValueError(f"invalid task id: {task_id!r}")
        return self.root / task_id

    def _frames_dir(self, task_id: str) -> Path:
        return self._task_dir(task_id) / "frames"

    # -- task lifecycle --------------------------------------------------------

    def create_task(self, task_id: str, config: TaskConfig) -> None:
        """Persist a new task; raises :class:`TaskExistsError` if present."""
        task_dir = self._task_dir(task_id)
        meta = task_dir / "meta.json"
        if meta.exists():
            raise TaskExistsError(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        self._frames_dir(task_id).mkdir(parents=True, exist_ok=True)
        _atomic_write_json(meta, _config_to_dict(config))

    def task_exists(self, task_id: str) -> bool:
        return (self._task_dir(task_id) / "meta.json").exists()

    def load_config(self, task_id: str) -> TaskConfig:
        meta = self._task_dir(task_id) / "meta.json"
        if not meta.exists():
            raise TaskNotFoundError(task_id)
        with meta.open("r", encoding="utf-8") as fh:
            return _config_from_dict(json.load(fh))

    # -- observations ----------------------------------------------------------

    def append_frame(self, task_id: str, frame: FrameObservation) -> bool:
        """Commit one frame; returns False if the id was already stored."""
        frames_dir = self._frames_dir(task_id)
        if not frames_dir.is_dir():
            raise TaskNotFoundError(task_id)
        existing = {f.frame_id for f in self.load_frames(task_id)}
        if frame.frame_id in existing:
            return False
        seq = self._next_sequence(frames_dir)
        _atomic_write_json(frames_dir / f"{seq:06d}.json", _frame_to_dict(frame))
        return True

    def load_frames(self, task_id: str) -> list[FrameObservation]:
        """Load all committed frames (canonical order, duplicates removed).

        Recovery rule: only fully committed ``NNNNNN.json`` files are
        replayed, in sequence order; temporary files from interrupted
        writes are ignored.  The first commit of a ``frame_id`` wins, so a
        frame is never applied twice.
        """
        frames_dir = self._frames_dir(task_id)
        if not frames_dir.is_dir():
            raise TaskNotFoundError(task_id)
        frames: list[FrameObservation] = []
        seen: set[str] = set()
        for path in sorted(frames_dir.glob("*.json")):
            name = path.stem
            if not name.isdigit():
                continue  # skip anything that is not a committed frame file
            with path.open("r", encoding="utf-8") as fh:
                frame = _frame_from_dict(json.load(fh))
            if frame.frame_id in seen:
                continue
            seen.add(frame.frame_id)
            frames.append(frame)
        frames.sort(key=lambda f: f.frame_id)
        return frames

    @staticmethod
    def _next_sequence(frames_dir: Path) -> int:
        highest = 0
        for path in frames_dir.glob("*.json"):
            if path.stem.isdigit():
                highest = max(highest, int(path.stem))
        return highest + 1

    # -- results -----------------------------------------------------------------

    def save_result(self, task_id: str, result: CalibrationResult) -> None:
        if not self.task_exists(task_id):
            raise TaskNotFoundError(task_id)
        _atomic_write_json(
            self._task_dir(task_id) / "result.json", _result_to_dict(result)
        )

    def load_result(self, task_id: str) -> CalibrationResult | None:
        path = self._task_dir(task_id) / "result.json"
        if not path.exists():
            if not self.task_exists(task_id):
                raise TaskNotFoundError(task_id)
            return None
        with path.open("r", encoding="utf-8") as fh:
            return _result_from_dict(json.load(fh))
