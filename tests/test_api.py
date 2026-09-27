"""HTTP API end-to-end tests."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from calibration.api import create_app
from tests.conftest import TRUE_FX, make_frames


@pytest.fixture()
def client(db_path):
    app = create_app(db_path)
    with TestClient(app) as c:
        yield c


def _post_frame(client, task, frame):
    return client.post(
        f"/tasks/{task}/frames",
        json={
            "frame_id": frame.frame_id,
            "object_points": frame.object_points.tolist(),
            "image_points": frame.image_points.tolist(),
            "image_size": list(frame.image_size),
        },
    )


def test_full_workflow(client):
    frames = make_frames(5)
    r = client.post("/tasks/wf", json={"config": {"max_iterations": 100, "convergence_tolerance": 1e-12}})
    assert r.status_code == 200
    assert r.json()["status"] == "created"

    # out of order + duplicate + corrupt
    for i in [2, 0, 4, 1, 3]:
        assert _post_frame(client, "wf", frames[i]).json()["status"] == "accepted"
    assert _post_frame(client, "wf", frames[2]).json()["status"] == "duplicate"
    # corrupt: mismatched point counts
    r = client.post(
        "/tasks/wf/frames",
        json={
            "frame_id": "broken",
            "object_points": frames[0].object_points.tolist(),
            "image_points": frames[0].image_points[:-1].tolist(),
            "image_size": [640, 480],
        },
    )
    assert r.json()["status"] == "corrupt"

    r = client.post("/tasks/wf/calibrate")
    assert r.status_code == 200
    res = r.json()
    assert res["fx"] == pytest.approx(TRUE_FX, abs=1e-6)
    assert res["converged"]
    assert set(res["per_frame_error"]) == {f.frame_id for f in frames}

    # stored result retrievable
    r = client.get("/tasks/wf/result")
    assert r.status_code == 200
    assert r.json()["fx"] == res["fx"]

    # task info
    r = client.get("/tasks/wf")
    assert r.json()["status"] == "calibrated"
    assert r.json()["frame_count"] == 5

    # point mappings round-trip
    pts = [[320.0, 240.0], [100.0, 80.0]]
    r = client.post("/tasks/wf/undistort-points", json={"points": pts})
    assert r.status_code == 200
    und = r.json()["points"]
    r = client.post("/tasks/wf/distort-points", json={"points": und})
    back = np.array(r.json()["points"])
    np.testing.assert_allclose(back, pts, atol=1e-4)


def test_unknown_task_404(client):
    assert client.get("/tasks/nope").status_code == 404
    assert client.post("/tasks/nope/calibrate").status_code == 404
    r = client.post(
        "/tasks/nope/frames",
        json={"frame_id": "f", "object_points": [[0, 0]], "image_points": [[0, 0]], "image_size": [1, 1]},
    )
    assert r.status_code == 404


def test_abandon_frame_endpoint(client):
    frames = make_frames(4)
    client.post("/tasks/ab")
    for f in frames:
        _post_frame(client, "ab", f)
    r = client.post(f"/tasks/ab/frames/{frames[3].frame_id}/abandon")
    assert r.status_code == 200
    r = client.post("/tasks/ab/calibrate")
    assert frames[3].frame_id not in r.json()["per_frame_error"]
    r = client.post("/tasks/ab/frames/ghost/abandon")
    assert r.status_code == 404


def test_config_overridable_per_calibrate(client):
    frames = make_frames(4)
    client.post("/tasks/cfg")
    for f in frames:
        _post_frame(client, "cfg", f)
    r = client.post("/tasks/cfg/calibrate", json={"max_iterations": 2, "convergence_tolerance": 1e-3})
    assert r.status_code == 200
    assert r.json()["iterations"] <= 2
