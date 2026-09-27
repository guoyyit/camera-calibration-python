"""Persistence, crash recovery and idempotency of the store."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np

from calibration.models import CalibrationConfig, CalibrationResult, FrameStatus
from calibration.storage import TaskStore
from tests.conftest import make_frames


def test_persistence_across_reopen(db_path):
    frames = make_frames(3)
    store = TaskStore(db_path)
    store.create_task("t", CalibrationConfig(max_iterations=42))
    for f in frames:
        store.add_frame("t", f.frame_id, FrameStatus.ACCEPTED, f)
    result = CalibrationResult(
        fx=1.0, fy=2.0, cx=3.0, cy=4.0,
        distortion=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        rms_error=0.01, per_frame_error={f.frame_id: 0.01 for f in frames},
        iterations=5, converged=True,
        frame_ids=[f.frame_id for f in frames],
        config=CalibrationConfig(max_iterations=42),
    )
    store.save_result("t", result)
    store.close()

    reopened = TaskStore(db_path)
    loaded = reopened.load_result("t")
    assert loaded is not None
    assert loaded.fx == 1.0 and loaded.cy == 4.0
    np.testing.assert_array_equal(loaded.distortion, result.distortion)
    assert loaded.config.max_iterations == 42
    valid = reopened.load_valid_frames("t")
    assert [f.frame_id for f in valid] == sorted(f.frame_id for f in frames)
    reopened.close()


def test_duplicate_frame_insert_is_idempotent(db_path):
    frames = make_frames(2)
    store = TaskStore(db_path)
    store.create_task("t", CalibrationConfig())
    assert store.add_frame("t", frames[0].frame_id, FrameStatus.ACCEPTED, frames[0])
    # second insert of the same frame_id is a no-op
    assert not store.add_frame("t", frames[0].frame_id, FrameStatus.ACCEPTED, frames[0])
    valid = store.load_valid_frames("t")
    assert len(valid) == 1
    store.close()


def test_uncommitted_transaction_leaves_no_partial_writes(db_path):
    frames = make_frames(2)
    store = TaskStore(db_path)
    store.create_task("t", CalibrationConfig())
    # Simulate an interrupted multi-statement write: insert inside a manual
    # transaction that is rolled back instead of committed.
    conn = store._conn
    conn.execute("BEGIN")
    f = frames[0]
    obj = np.ascontiguousarray(f.object_points, dtype=np.float64)
    img = np.ascontiguousarray(f.image_points, dtype=np.float64)
    conn.execute(
        "INSERT INTO frames (task_id, frame_id, status, width, height,"
        " n_points, object_points, image_points, seq)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("t", f.frame_id, "accepted", 640, 480, len(obj),
         obj.tobytes(), img.tobytes(), 1),
    )
    conn.rollback()  # crash before commit
    assert store.load_valid_frames("t") == []
    store.close()

    reopened = TaskStore(db_path)
    assert reopened.load_valid_frames("t") == []
    reopened.close()


def _run_crash_subprocess(db_path: str, script: str) -> None:
    code = textwrap.dedent(script).format(db_path=db_path)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert proc.returncode != 0, f"subprocess should have crashed: {proc.stdout}{proc.stderr}"


def test_crash_after_commit_recovers_last_committed_state(db_path):
    """A hard crash (os._exit) after commits must keep all committed data."""
    _run_crash_subprocess(db_path, """
        import os, sys
        sys.path.insert(0, os.getcwd())
        import numpy as np
        from calibration.storage import TaskStore
        from calibration.models import CalibrationConfig, FrameStatus
        from tests.conftest import make_frames
        store = TaskStore({db_path!r})
        store.create_task("t", CalibrationConfig())
        for f in make_frames(3):
            store.add_frame("t", f.frame_id, FrameStatus.ACCEPTED, f)
        os._exit(1)  # simulate power loss: no close, no cleanup
    """)
    store = TaskStore(db_path)
    valid = store.load_valid_frames("t")
    assert len(valid) == 3  # all committed frames survived
    assert store.task_exists("t")
    store.close()


def test_crash_mid_sequence_keeps_only_committed_frames(db_path):
    """Frames committed before the crash survive; nothing is half-written."""
    _run_crash_subprocess(db_path, """
        import os, sys
        sys.path.insert(0, os.getcwd())
        from calibration.storage import TaskStore
        from calibration.models import CalibrationConfig, FrameStatus
        from tests.conftest import make_frames
        store = TaskStore({db_path!r})
        store.create_task("t", CalibrationConfig())
        frames = make_frames(5)
        store.add_frame("t", frames[0].frame_id, FrameStatus.ACCEPTED, frames[0])
        store.add_frame("t", frames[1].frame_id, FrameStatus.ACCEPTED, frames[1])
        os._exit(1)  # crash before the remaining frames are submitted
    """)
    store = TaskStore(db_path)
    valid = store.load_valid_frames("t")
    assert [f.frame_id for f in valid] == ["f000", "f001"]
    store.close()
