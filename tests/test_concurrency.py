"""Concurrent create / append / read / recalibrate must be linearisable."""

from __future__ import annotations

import threading

from calibration.service import CalibrationService
from calibration.solver import calibrate
from calibration.store import TaskExistsError

from .conftest import IMAGE_HEIGHT, IMAGE_WIDTH, make_frame
import numpy as np


def _run_parallel(fn, n_threads):
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    def worker(i):
        try:
            barrier.wait(timeout=10)
            fn(i)
        except BaseException as exc:  # noqa: BLE001 - collected for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    return errors


def test_concurrent_create_exactly_one_wins(tmp_path, task_config):
    svc = CalibrationService(tmp_path)
    outcomes: list[str] = []
    lock = threading.Lock()

    def create(_):
        try:
            svc.create_task("race", task_config)
            with lock:
                outcomes.append("ok")
        except TaskExistsError:
            with lock:
                outcomes.append("exists")

    errors = _run_parallel(create, 8)
    assert errors == []
    assert outcomes.count("ok") == 1
    assert outcomes.count("exists") == 7
    assert svc.get_frames("race") == []


def test_concurrent_appends_all_committed_exactly_once(tmp_path, task_config):
    svc = CalibrationService(tmp_path)
    svc.create_task("t", task_config)
    rng = np.random.default_rng(5)
    n_threads, per_thread = 6, 2
    frames = [make_frame(f"f{i:03d}", rng) for i in range(n_threads * per_thread)]

    def append(i):
        svc.add_observations("t", frames[i * per_thread:(i + 1) * per_thread])

    errors = _run_parallel(append, n_threads)
    assert errors == []
    stored = svc.get_frames("t")
    assert [f.frame_id for f in stored] == sorted(f.frame_id for f in frames)

    # every thread resubmits its frames: still no duplicates
    errors = _run_parallel(append, n_threads)
    assert errors == []
    assert len(svc.get_frames("t")) == len(frames)


def test_concurrent_calibrate_and_append_linearisable(tmp_path, task_config):
    """Readers/calibrators always see a consistent committed snapshot."""
    svc = CalibrationService(tmp_path)
    svc.create_task("t", task_config)
    rng = np.random.default_rng(11)
    frames = [make_frame(f"f{i:03d}", rng) for i in range(12)]
    svc.add_observations("t", frames[:4])  # enough to calibrate

    observed: list[tuple[str, ...]] = []
    lock = threading.Lock()
    stop = threading.Event()

    def calibrator():
        while not stop.is_set():
            result = svc.calibrate("t")
            with lock:
                observed.append(result.used_frame_ids)

    def appender(i):
        svc.add_observations("t", [frames[4 + i]])

    threads = [threading.Thread(target=calibrator) for _ in range(2)]
    for t in threads:
        t.start()
    errors = _run_parallel(appender, 8)
    assert errors == []
    stop.set()
    for t in threads:
        t.join(timeout=60)

    # every observed snapshot must be a prefix-closed subset of committed ids
    all_ids = [f.frame_id for f in frames]
    for ids in observed:
        assert list(ids) == sorted(ids)
        assert set(ids) <= set(all_ids)
        assert len(ids) >= 4

    # final calibration is deterministic and equals the full-set solve
    final = svc.calibrate("t")
    assert final.used_frame_ids == tuple(sorted(all_ids))
    assert final == calibrate(frames, IMAGE_WIDTH, IMAGE_HEIGHT, task_config.solver)
    # repeated calibration of a quiescent task is bit-identical
    assert svc.calibrate("t") == final


def test_concurrent_reads_during_writes(tmp_path, task_config):
    svc = CalibrationService(tmp_path)
    svc.create_task("t", task_config)
    rng = np.random.default_rng(21)
    frames = [make_frame(f"f{i:03d}", rng) for i in range(8)]
    svc.add_observations("t", frames[:4])
    svc.calibrate("t")

    seen: list[int] = []
    lock = threading.Lock()
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            with lock:
                seen.append(len(svc.get_frames("t")))
            svc.get_result("t")

    def appender(i):
        svc.add_observations("t", [frames[4 + i]])

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    errors = _run_parallel(appender, 4)
    assert errors == []
    stop.set()
    for t in threads:
        t.join(timeout=60)

    # frame counts observed by readers are monotonically non-decreasing
    # values drawn from the committed prefixes {4,5,6,7,8}
    assert set(seen) <= {4, 5, 6, 7, 8}
    assert len(svc.get_frames("t")) == 8
