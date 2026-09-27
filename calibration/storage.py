"""SQLite-backed persistent store with crash recovery.

Durability model
----------------
- Every mutation runs inside a single SQLite transaction; a crash either
  commits the whole transaction or rolls it back, so the store is always
  at a fully committed state — no partial writes are ever visible.
- The database runs in WAL mode with ``synchronous=FULL``; after a crash
  SQLite automatically recovers to the last committed transaction on the
  next open.
- Frames are keyed by ``(task_id, frame_id)``; re-submitting an existing
  ``frame_id`` is an idempotent no-op (``INSERT OR IGNORE``), so retried
  or duplicated observations are never applied twice.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

from .models import (
    CalibrationConfig,
    CalibrationResult,
    FrameObservation,
    FrameStatus,
    TaskStatus,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id     TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    config_json TEXT NOT NULL,
    result_json TEXT,
    version     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS frames (
    task_id       TEXT NOT NULL,
    frame_id      TEXT NOT NULL,
    status        TEXT NOT NULL,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    n_points      INTEGER NOT NULL,
    object_points BLOB NOT NULL,
    image_points  BLOB NOT NULL,
    seq           INTEGER NOT NULL,
    PRIMARY KEY (task_id, frame_id)
);
"""


class TaskStore:
    """Durable store for tasks, frames and calibration results."""

    def __init__(self, db_path: str | Path) -> None:
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._conn:
            self._conn.executescript(_SCHEMA)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------- tasks
    def create_task(self, task_id: str, config: CalibrationConfig) -> bool:
        """Create a task; returns False if it already exists (idempotent)."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO tasks (task_id, status, config_json)"
                " VALUES (?, ?, ?)",
                (task_id, TaskStatus.CREATED.value, json.dumps(_config_to_dict(config))),
            )
            return cur.rowcount > 0

    def task_exists(self, task_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            return row is not None

    def load_config(self, task_id: str) -> CalibrationConfig:
        with self._lock:
            row = self._conn.execute(
                "SELECT config_json FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown task {task_id!r}")
        return _config_from_dict(json.loads(row["config_json"]))

    # ------------------------------------------------------------- frames
    def add_frame(
        self,
        task_id: str,
        frame_id: str,
        status: FrameStatus,
        obs: FrameObservation | None,
    ) -> bool:
        """Record a frame submission; False if frame_id already stored.

        The insert is idempotent: an existing ``frame_id`` is left
        untouched so a retried observation is never applied twice.
        ``obs`` may be None for corrupt frames that failed validation.
        """
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM frames"
                " WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            seq = row["next_seq"]
            if obs is None:
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO frames"
                    " (task_id, frame_id, status, width, height, n_points,"
                    "  object_points, image_points, seq)"
                    " VALUES (?, ?, ?, 0, 0, 0, ?, ?, ?)",
                    (task_id, frame_id, status.value, b"", b"", seq),
                )
            else:
                obj = np.ascontiguousarray(obs.object_points, dtype=np.float64)
                img = np.ascontiguousarray(obs.image_points, dtype=np.float64)
                w, h = obs.image_size
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO frames"
                    " (task_id, frame_id, status, width, height, n_points,"
                    "  object_points, image_points, seq)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (task_id, frame_id, status.value, w, h, len(obj),
                     obj.tobytes(), img.tobytes(), seq),
                )
            return cur.rowcount > 0

    def set_frame_status(self, task_id: str, frame_id: str, status: FrameStatus) -> bool:
        """Update a frame's status (e.g. mark ABANDONED). False if missing."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE frames SET status = ? WHERE task_id = ? AND frame_id = ?",
                (status.value, task_id, frame_id),
            )
            return cur.rowcount > 0

    def frame_statuses(self, task_id: str) -> dict[str, FrameStatus]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT frame_id, status FROM frames WHERE task_id = ?"
                " ORDER BY frame_id",
                (task_id,),
            ).fetchall()
        return {r["frame_id"]: FrameStatus(r["status"]) for r in rows}

    def load_valid_frames(self, task_id: str) -> list[FrameObservation]:
        """Load ACCEPTED frames sorted by frame_id (deterministic order)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT frame_id, width, height, n_points, object_points,"
                " image_points FROM frames"
                " WHERE task_id = ? AND status = ? ORDER BY frame_id",
                (task_id, FrameStatus.ACCEPTED.value),
            ).fetchall()
        out = []
        for r in rows:
            n = r["n_points"]
            obj = np.frombuffer(r["object_points"], dtype=np.float64).reshape(n, 2).copy()
            img = np.frombuffer(r["image_points"], dtype=np.float64).reshape(n, 2).copy()
            out.append(
                FrameObservation(
                    frame_id=r["frame_id"],
                    object_points=obj,
                    image_points=img,
                    image_size=(int(r["width"]), int(r["height"])),
                )
            )
        return out

    # ------------------------------------------------------------- results
    def save_result(self, task_id: str, result: CalibrationResult) -> None:
        """Persist a calibration result atomically (single transaction)."""
        payload = json.dumps(_result_to_dict(result))
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE tasks SET result_json = ?, status = ?,"
                " version = version + 1 WHERE task_id = ?",
                (payload, TaskStatus.CALIBRATED.value, task_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"unknown task {task_id!r}")

    def load_result(self, task_id: str) -> CalibrationResult | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT result_json FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown task {task_id!r}")
        if row["result_json"] is None:
            return None
        return _result_from_dict(json.loads(row["result_json"]))

    def task_status(self, task_id: str) -> TaskStatus:
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown task {task_id!r}")
        return TaskStatus(row["status"])


# ----------------------------------------------------------- serialization
def _config_to_dict(c: CalibrationConfig) -> dict:
    return {
        "max_iterations": c.max_iterations,
        "convergence_tolerance": c.convergence_tolerance,
        "use_tangential": c.use_tangential,
        "use_k3": c.use_k3,
    }


def _config_from_dict(d: dict) -> CalibrationConfig:
    return CalibrationConfig(
        max_iterations=int(d["max_iterations"]),
        convergence_tolerance=float(d["convergence_tolerance"]),
        use_tangential=bool(d["use_tangential"]),
        use_k3=bool(d["use_k3"]),
    )


def _result_to_dict(r: CalibrationResult) -> dict:
    return {
        "fx": r.fx,
        "fy": r.fy,
        "cx": r.cx,
        "cy": r.cy,
        "distortion": [float(v) for v in r.distortion],
        "rms_error": r.rms_error,
        "per_frame_error": {k: float(v) for k, v in sorted(r.per_frame_error.items())},
        "iterations": r.iterations,
        "converged": r.converged,
        "frame_ids": list(r.frame_ids),
        "config": _config_to_dict(r.config),
    }


def _result_from_dict(d: dict) -> CalibrationResult:
    return CalibrationResult(
        fx=float(d["fx"]),
        fy=float(d["fy"]),
        cx=float(d["cx"]),
        cy=float(d["cy"]),
        distortion=np.array(d["distortion"], dtype=np.float64),
        rms_error=float(d["rms_error"]),
        per_frame_error={k: float(v) for k, v in d["per_frame_error"].items()},
        iterations=int(d["iterations"]),
        converged=bool(d["converged"]),
        frame_ids=list(d["frame_ids"]),
        config=_config_from_dict(d["config"]),
    )
