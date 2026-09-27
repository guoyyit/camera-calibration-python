"""Concurrent create / append / read / recalibrate must linearise and
produce a deterministic final state."""

from __future__ import annotations

import threading
from collections import Counter

import numpy as np
import pytest

from calibration.models import FrameStatus
from calibration.service import CalibrationService
from calibration.solver import calibrate
from tests.conftest import make_frames


@pytest.fixture()
def svc(db_path):
    service = CalibrationService(db_path)
    yield service
    service.close()


def _run_threads(fn, n):
    threads = [threading.Thread(target=fn, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_concurrent_create_same_task(svc):
    errors = []

    def create(i):
        try:
            svc.create_task("shared")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    _run_threads(create, 16)
    assert errors == []
    assert svc.get_task("shared").task_id == "shared"


def test_concurrent_duplicate_submission_applied_once(svc):
    svc.create_task("t")
    frame = make_frames(1)[0]
    outcomes = []
    lock = threading.Lock()

    def submit(i):
        status = svc.submit_frame(
            "t", frame.frame_id, frame.object_points, frame.image_points, frame.image_size
        )
        with lock:
            outcomes.append(status)

    _run_threads(submit, 16)
    counts = Counter(outcomes)
    assert counts[FrameStatus.ACCEPTED] == 1
    assert counts[FrameStatus.DUPLICATE] == 15
    assert len(svc._store.load_valid_frames("t")) == 1


def test_concurrent_append_and_calibrate_linearises(svc):
    frames = make_frames(8)
    svc.create_task("t")
    errors = []
    calibrate_results = []

    def submit(i):
        try:
            f = frames[i]
            svc.submit_frame("t", f.frame_id, f.object_points, f.image_points, f.image_size)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def run_calibrate(i):
        try:
            res = svc.calibrate("t")
            calibrate_results.append(res)
        except ValueError:
            pass  # too few frames at that linearisation point: acceptable
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def read(i):
        try:
            svc.get_task("t")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = (
        [threading.Thread(target=submit, args=(i,)) for i in range(8)]
        + [threading.Thread(target=run_calibrate, args=(i,)) for i in range(4)]
        + [threading.Thread(target=read, args=(i,)) for i in range(4)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # every accepted frame stored exactly once
    assert len(svc._store.load_valid_frames("t")) == 8
    # each completed calibration saw a consistent prefix of the frame set
    for res in calibrate_results:
        assert res.frame_ids == sorted(res.frame_ids)
        assert set(res.frame_ids) <= {f.frame_id for f in frames}
    # final calibration is deterministic and equals the serial reference
    final = svc.calibrate("t")
    ref = calibrate(frames)
    assert final.fx == ref.fx and final.fy == ref.fy
    assert final.cx == ref.cx and final.cy == ref.cy
    np.testing.assert_array_equal(final.distortion, ref.distortion)
    assert final.per_frame_error == ref.per_frame_error


def test_concurrent_workload_is_repeatable(svc, db_path):
    """Two identical concurrent workloads yield identical final results."""
    frames = make_frames(6)

    def workload(service):
        service.create_task("t")

        def submit(i):
            f = frames[i]
            service.submit_frame("t", f.frame_id, f.object_points, f.image_points, f.image_size)

        _run_threads(submit, len(frames))
        return service.calibrate("t")

    first = workload(svc)
    svc2 = CalibrationService(db_path.replace(".db", "2.db"))
    try:
        second = workload(svc2)
    finally:
        svc2.close()
    assert first.fx == second.fx and first.fy == second.fy
    assert first.cx == second.cx and first.cy == second.cy
    np.testing.assert_array_equal(first.distortion, second.distortion)
    assert first.rms_error == second.rms_error
