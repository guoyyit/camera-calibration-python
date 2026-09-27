"""Service-level behaviour: ordering, duplicates, corrupt frames,
incremental recalibration and abandoned frames."""

from __future__ import annotations

import numpy as np
import pytest

from calibration.models import CalibrationConfig, FrameStatus
from calibration.service import CalibrationService
from calibration.solver import calibrate
from tests.conftest import make_frames


@pytest.fixture()
def svc(db_path):
    service = CalibrationService(db_path)
    yield service
    service.close()


def _submit(svc, task, frame):
    return svc.submit_frame(
        task, frame.frame_id, frame.object_points, frame.image_points, frame.image_size
    )


def test_out_of_order_duplicate_and_corrupt_frames(svc, frames):
    svc.create_task("t")
    # out of order
    for i in [3, 0, 5, 2, 4, 1]:
        assert _submit(svc, "t", frames[i]) == FrameStatus.ACCEPTED
    # duplicates are ignored
    assert _submit(svc, "t", frames[3]) == FrameStatus.DUPLICATE
    assert _submit(svc, "t", frames[0]) == FrameStatus.DUPLICATE
    # corrupt: NaN coordinate
    bad = frames[0].image_points.copy()
    bad[0, 0] = np.nan
    assert svc.submit_frame("t", "nan-frame", frames[0].object_points, bad, (640, 480)) == FrameStatus.CORRUPT
    # corrupt: too few points
    assert svc.submit_frame("t", "few", frames[0].object_points[:3], frames[0].image_points[:3], (640, 480)) == FrameStatus.CORRUPT
    # corrupt: collinear object points
    col = np.column_stack([np.arange(6), np.arange(6) * 2.0])
    assert svc.submit_frame("t", "collinear", col, frames[0].image_points[:6], (640, 480)) == FrameStatus.CORRUPT
    # corrupt: mismatched counts
    assert svc.submit_frame("t", "mismatch", frames[0].object_points, frames[0].image_points[:-1], (640, 480)) == FrameStatus.CORRUPT

    res = svc.calibrate("t")
    ref = calibrate(frames)  # only the six valid frames
    assert res.fx == ref.fx and res.fy == ref.fy
    assert res.cx == ref.cx and res.cy == ref.cy
    np.testing.assert_array_equal(res.distortion, ref.distortion)
    assert res.frame_ids == sorted(f.frame_id for f in frames)


def test_incremental_recalibration_matches_full(svc, frames):
    extra = make_frames(3, seed=99, prefix="g")
    svc.create_task("t")
    for f in frames[:4]:
        _submit(svc, "t", f)
    svc.calibrate("t")
    # append more observations later (incremental)
    for f in frames[4:] + extra:
        _submit(svc, "t", f)
    res_inc = svc.calibrate("t")
    ref_full = calibrate(frames + extra)
    assert res_inc.fx == ref_full.fx
    assert res_inc.fy == ref_full.fy
    assert res_inc.cx == ref_full.cx
    assert res_inc.cy == ref_full.cy
    np.testing.assert_array_equal(res_inc.distortion, ref_full.distortion)
    assert res_inc.per_frame_error == ref_full.per_frame_error


def test_abandoned_frames_excluded(svc, frames):
    svc.create_task("t")
    for f in frames:
        _submit(svc, "t", f)
    svc.calibrate("t")
    assert svc.abandon_frame("t", frames[5].frame_id)
    res = svc.calibrate("t")
    ref = calibrate(frames[:5])
    assert res.fx == ref.fx
    np.testing.assert_array_equal(res.distortion, ref.distortion)
    assert frames[5].frame_id not in res.per_frame_error


def test_unknown_task_raises(svc):
    with pytest.raises(KeyError):
        svc.get_task("nope")
    with pytest.raises(KeyError):
        svc.calibrate("nope")


def test_config_stored_and_used(svc, frames):
    cfg = CalibrationConfig(max_iterations=3, convergence_tolerance=1e-4)
    svc.create_task("t", cfg)
    for f in frames:
        _submit(svc, "t", f)
    res = svc.calibrate("t")
    assert res.config.max_iterations == 3
    assert res.iterations <= 3
