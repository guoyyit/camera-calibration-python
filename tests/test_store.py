"""Persistence and crash-recovery behaviour of the task store."""

from __future__ import annotations

import json
import os

import pytest

from calibration.models import FrameObservation
from calibration.service import CalibrationService
from calibration.store import TaskExistsError, TaskNotFoundError, TaskStore

from .conftest import board_points


def _frame(frame_id: str) -> FrameObservation:
    obj = board_points()
    return FrameObservation(
        frame_id,
        tuple(map(tuple, obj.tolist())),
        tuple(map(tuple, obj[:, :2].tolist())),
    )


def test_create_append_recover(tmp_path, task_config):
    store = TaskStore(tmp_path)
    store.create_task("t", task_config)
    assert store.task_exists("t")
    assert store.append_frame("t", _frame("b")) is True
    assert store.append_frame("t", _frame("a")) is True
    assert store.append_frame("t", _frame("a")) is False  # duplicate

    recovered = TaskStore(tmp_path)  # fresh instance = process restart
    assert recovered.load_config("t") == task_config
    frames = recovered.load_frames("t")
    assert [f.frame_id for f in frames] == ["a", "b"]  # canonical order


def test_duplicate_create_rejected(tmp_path, task_config):
    store = TaskStore(tmp_path)
    store.create_task("t", task_config)
    with pytest.raises(TaskExistsError):
        store.create_task("t", task_config)


def test_missing_task_raises(tmp_path):
    store = TaskStore(tmp_path)
    assert not store.task_exists("nope")
    with pytest.raises(TaskNotFoundError):
        store.load_config("nope")
    with pytest.raises(TaskNotFoundError):
        store.load_frames("nope")
    with pytest.raises(TaskNotFoundError):
        store.append_frame("nope", _frame("x"))


def test_crash_during_append_leaves_no_partial_state(tmp_path, task_config, monkeypatch):
    """A crash mid-write must not commit the frame nor leave temp files."""
    store = TaskStore(tmp_path)
    store.create_task("t", task_config)
    store.append_frame("t", _frame("committed"))

    def exploding_replace(src, dst):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(os, "replace", exploding_replace)
    with pytest.raises(RuntimeError):
        store.append_frame("t", _frame("victim"))
    monkeypatch.undo()

    frames_dir = tmp_path / "t" / "frames"
    leftovers = [p for p in frames_dir.iterdir() if p.name.startswith(".tmp-")]
    assert leftovers == [], "temporary file was not cleaned up"
    recovered = TaskStore(tmp_path)
    assert [f.frame_id for f in recovered.load_frames("t")] == ["committed"]

    # store still usable afterwards; sequence numbers do not collide
    assert recovered.append_frame("t", _frame("after-crash")) is True
    assert [f.frame_id for f in TaskStore(tmp_path).load_frames("t")] == [
        "after-crash", "committed",
    ]


def test_leftover_temp_files_ignored_on_recovery(tmp_path, task_config):
    """Interrupted writes (orphan temp files) are invisible after recovery."""
    store = TaskStore(tmp_path)
    store.create_task("t", task_config)
    store.append_frame("t", _frame("ok"))
    frames_dir = tmp_path / "t" / "frames"
    (frames_dir / ".tmp-partial").write_text('{"frame_id": "ghost"')
    assert [f.frame_id for f in TaskStore(tmp_path).load_frames("t")] == ["ok"]


def test_result_persistence_is_atomic(tmp_path, synthetic_frames, task_config, monkeypatch):
    svc = CalibrationService(tmp_path)
    svc.create_task("t", task_config)
    svc.add_observations("t", synthetic_frames)
    result = svc.calibrate("t")
    assert svc.get_result("t") == result

    # crash while saving the next result keeps the previous committed one
    def exploding_replace(src, dst):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(os, "replace", exploding_replace)
    with pytest.raises(RuntimeError):
        svc.store.save_result("t", result)
    monkeypatch.undo()
    assert CalibrationService(tmp_path).get_result("t") == result


def test_no_double_application_across_restart(tmp_path, synthetic_frames, task_config):
    """Frames committed before a crash are not applied twice after it."""
    svc = CalibrationService(tmp_path)
    svc.create_task("t", task_config)
    svc.add_observations("t", synthetic_frames[:5])
    r1 = svc.calibrate("t")

    # "restart": new service instance on the same directory
    svc2 = CalibrationService(tmp_path)
    svc2.add_observations("t", synthetic_frames[:5])  # resubmitted after restart
    svc2.add_observations("t", synthetic_frames[5:])
    r2 = svc2.calibrate("t")
    assert len(svc2.get_frames("t")) == len(synthetic_frames)
    full = CalibrationService(tmp_path)
    assert r2.used_frame_ids == tuple(sorted(f.frame_id for f in synthetic_frames))
    assert r1.used_frame_ids == tuple(
        sorted(f.frame_id for f in synthetic_frames[:5])
    )


def test_meta_json_is_valid_json(tmp_path, task_config):
    store = TaskStore(tmp_path)
    store.create_task("t", task_config)
    with (tmp_path / "t" / "meta.json").open() as fh:
        data = json.load(fh)
    assert data["image_width"] == task_config.image_width
    assert data["solver"]["max_iterations"] == task_config.solver.max_iterations
