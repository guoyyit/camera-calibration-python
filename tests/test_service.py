"""Out-of-order / duplicate / corrupted frames and incremental recalibration."""

from __future__ import annotations

import numpy as np

from calibration.models import FrameObservation
from calibration.service import CalibrationService
from calibration.solver import calibrate

from .conftest import IMAGE_HEIGHT, IMAGE_WIDTH, board_points


def _corrupted_frames() -> list[FrameObservation]:
    obj = board_points()
    good_img = tuple(map(tuple, obj[:, :2].tolist()))
    good_obj = tuple(map(tuple, obj.tolist()))
    return [
        # NaN in image points
        FrameObservation("bad-nan", good_obj,
                         tuple((float("nan"), p[1]) for p in good_img)),
        # Inf in object points
        FrameObservation("bad-inf", tuple((float("inf"),) * 3 for _ in good_obj),
                         good_img),
        # mismatched point counts
        FrameObservation("bad-count", good_obj[:-1], good_img),
        # too few points
        FrameObservation("bad-few", good_obj[:3], good_img[:3]),
        # empty id
        FrameObservation("", good_obj, good_img),
        # collinear object points
        FrameObservation("bad-collinear",
                         tuple((float(i), 0.0, 0.0) for i in range(len(good_obj))),
                         good_img),
    ]


def test_out_of_order_duplicate_and_corrupted(tmp_path, synthetic_frames, task_config):
    svc = CalibrationService(tmp_path)
    svc.create_task("t", task_config)

    shuffled = [synthetic_frames[i] for i in [7, 2, 9, 0, 4, 1, 8, 3, 6, 5]]
    batch = shuffled + [synthetic_frames[3], synthetic_frames[3]] + _corrupted_frames()
    rejected = svc.add_observations("t", batch)

    assert {r.frame_id for r in rejected} == {
        "bad-nan", "bad-inf", "bad-count", "bad-few", "", "bad-collinear"
    }
    stored = svc.get_frames("t")
    assert [f.frame_id for f in stored] == sorted(f.frame_id for f in synthetic_frames)

    result = svc.calibrate("t")
    # identical to calibration over only the valid frames
    reference = calibrate(synthetic_frames, IMAGE_WIDTH, IMAGE_HEIGHT, task_config.solver)
    assert result == reference


def test_duplicate_frames_do_not_change_result(tmp_path, synthetic_frames, task_config):
    svc = CalibrationService(tmp_path)
    svc.create_task("t", task_config)
    svc.add_observations("t", synthetic_frames)
    r1 = svc.calibrate("t")
    # resubmit every frame twice, in reverse order
    svc.add_observations("t", list(reversed(synthetic_frames)) * 2)
    r2 = svc.calibrate("t")
    assert r1 == r2
    assert len(svc.get_frames("t")) == len(synthetic_frames)


def test_incremental_recalibration_matches_full(tmp_path, synthetic_frames, task_config):
    svc = CalibrationService(tmp_path)
    svc.create_task("t", task_config)

    svc.add_observations("t", synthetic_frames[:5])
    svc.calibrate("t")
    # append the rest incrementally, out of order, with a duplicate
    svc.add_observations("t", [synthetic_frames[8], synthetic_frames[5]])
    svc.add_observations("t", [synthetic_frames[9], synthetic_frames[7],
                               synthetic_frames[6], synthetic_frames[5]])
    inc = svc.calibrate("t")

    full = calibrate(synthetic_frames, IMAGE_WIDTH, IMAGE_HEIGHT, task_config.solver)
    assert inc == full
    assert inc.used_frame_ids == tuple(sorted(f.frame_id for f in synthetic_frames))


def test_rejected_frames_never_participate(tmp_path, synthetic_frames, task_config):
    svc = CalibrationService(tmp_path)
    svc.create_task("t", task_config)
    svc.add_observations("t", synthetic_frames[:6] + _corrupted_frames())
    result = svc.calibrate("t")
    assert set(result.used_frame_ids).isdisjoint(
        {"bad-nan", "bad-inf", "bad-count", "bad-few", "", "bad-collinear"}
    )
    assert len(result.used_frame_ids) == 6


def test_calibrate_persists_result(tmp_path, synthetic_frames, task_config):
    svc = CalibrationService(tmp_path)
    svc.create_task("t", task_config)
    svc.add_observations("t", synthetic_frames)
    result = svc.calibrate("t")
    # a fresh service on the same root sees the committed result
    svc2 = CalibrationService(tmp_path)
    assert svc2.get_result("t") == result
