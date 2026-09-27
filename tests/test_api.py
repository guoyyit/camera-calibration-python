"""End-to-end tests of the FastAPI surface."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from calibration.solver import project_points

from .conftest import IMAGE_HEIGHT, IMAGE_WIDTH, TRUE_INTRINSICS, board_points


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient backed by a fresh store in a temp directory."""
    monkeypatch.setenv("CALIBRATION_STORE", str(tmp_path))
    import main
    importlib.reload(main)
    with TestClient(main.app) as test_client:
        yield test_client


def _frame_payloads(n: int, seed: int = 3) -> list[dict]:
    rng = np.random.default_rng(seed)
    obj = board_points()
    frames = []
    for i in range(n):
        rvec = rng.uniform(-0.4, 0.4, 3)
        tvec = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.08, 0.08),
                         rng.uniform(0.4, 0.8)])
        proj = project_points(obj, rvec, tvec, TRUE_INTRINSICS)
        frames.append({
            "frame_id": f"f{i:03d}",
            "object_points": obj.tolist(),
            "image_points": proj.tolist(),
        })
    return frames


def test_full_workflow(client):
    assert client.post("/tasks", json={
        "task_id": "t", "image_width": IMAGE_WIDTH, "image_height": IMAGE_HEIGHT,
        "solver": {"max_iterations": 100, "convergence_tolerance": 1e-12},
    }).status_code == 201
    # duplicate create -> 409
    assert client.post("/tasks", json={
        "task_id": "t", "image_width": IMAGE_WIDTH, "image_height": IMAGE_HEIGHT,
    }).status_code == 409

    frames = _frame_payloads(6)
    r = client.post("/tasks/t/observations",
                    json={"frames": frames[::-1] + [frames[0]]})
    assert r.status_code == 200
    assert r.json()["rejected"] == []

    r = client.post("/tasks/t/calibrate")
    assert r.status_code == 200
    body = r.json()
    assert abs(body["intrinsics"]["fx"] - TRUE_INTRINSICS[0]) < 1.0
    assert body["mean_reprojection_error"] < 1e-6
    assert len(body["frame_errors"]) == 6
    assert body["used_frame_ids"] == sorted(f["frame_id"] for f in frames)

    assert client.get("/tasks/t/result").json() == body


def test_corrupted_frame_rejected_via_api(client):
    client.post("/tasks", json={"task_id": "t", "image_width": IMAGE_WIDTH,
                                "image_height": IMAGE_HEIGHT})
    frames = _frame_payloads(4)
    # corrupted: fewer image points than object points
    frames[1]["image_points"] = frames[1]["image_points"][:10]
    # corrupted: too few points to be usable
    frames[2]["object_points"] = frames[2]["object_points"][:3]
    frames[2]["image_points"] = frames[2]["image_points"][:3]
    r = client.post("/tasks/t/observations", json={"frames": frames})
    assert r.status_code == 200
    assert {f["frame_id"] for f in r.json()["rejected"]} == {"f001", "f002"}


def test_unknown_task_404(client):
    assert client.post("/tasks/nope/observations", json={"frames": []}).status_code == 404
    assert client.post("/tasks/nope/calibrate").status_code == 404
    assert client.get("/tasks/nope/result").status_code == 404


def test_mapping_endpoints_round_trip(client):
    intrinsics = {
        "fx": 800.0, "fy": 810.0, "cx": 320.0, "cy": 240.0,
        "k1": -0.2, "k2": 0.05, "p1": 0.001, "p2": -0.002,
    }
    pts = [[100.0, 100.0], [320.0, 240.0], [639.0, 479.0]]
    distorted = client.post("/distort/points",
                            json={"intrinsics": intrinsics, "points": pts}).json()["points"]
    recovered = client.post("/undistort/points",
                            json={"intrinsics": intrinsics, "points": distorted}).json()["points"]
    assert np.allclose(recovered, pts, atol=1e-6)
