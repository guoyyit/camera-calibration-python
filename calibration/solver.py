"""Intrinsic calibration solver.

Pipeline (Zhang's method):
1. Per-frame homography (normalised DLT) from the planar target to the image.
2. Closed-form intrinsics from the homographies (skew = 0 constraint).
3. Per-frame extrinsics from each homography.
4. Joint Levenberg-Marquardt refinement of intrinsics, distortion and
   extrinsics minimising reprojection error.

Determinism: frames are processed in sorted ``frame_id`` order and all
numerical steps are deterministic, so identical inputs always yield
bit-identical outputs.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .models import CalibrationConfig, CalibrationResult, FrameObservation

_MIN_FRAMES = 2
_MIN_POINTS_PER_FRAME = 4


def _homography_dlt(obj: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Normalised DLT homography mapping obj (N,2) -> img (N,2)."""

    def _norm(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = pts.mean(axis=0)
        scale = np.sqrt(2.0) / (np.linalg.norm(pts - mean, axis=1).mean() + 1e-12)
        t = np.array(
            [[scale, 0.0, -scale * mean[0]],
             [0.0, scale, -scale * mean[1]],
             [0.0, 0.0, 1.0]]
        )
        return t, np.column_stack([pts, np.ones(len(pts))]) @ t.T

    tobj, op = _norm(obj)
    timg, ip = _norm(img)
    a = []
    for (x, y, _), (u, v, _) in zip(op, ip):
        a.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        a.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(a))
    h = vt[-1].reshape(3, 3)
    h = np.linalg.inv(timg) @ h @ tobj
    return h / (h[2, 2] if abs(h[2, 2]) > 1e-12 else 1.0)


def _intrinsics_from_homographies(homographies: list[np.ndarray]) -> np.ndarray:
    """Zhang's closed-form intrinsics from >= 2 homographies (skew = 0)."""

    def v(h: np.ndarray, i: int, j: int) -> np.ndarray:
        hi, hj = h[:, i], h[:, j]
        return np.array([
            hi[0] * hj[0],
            hi[0] * hj[1] + hi[1] * hj[0],
            hi[1] * hj[1],
            hi[2] * hj[0] + hi[0] * hj[2],
            hi[2] * hj[1] + hi[1] * hj[2],
            hi[2] * hj[2],
        ])

    rows = []
    for h in homographies:
        rows.append(v(h, 0, 1))
        rows.append(v(h, 0, 0) - v(h, 1, 1))
    _, _, vt = np.linalg.svd(np.asarray(rows))
    b = vt[-1]
    # B is only defined up to scale; the physical solution is positive
    # definite, so flip the sign if SVD returned the negated vector.
    if b[0] < 0:
        b = -b
    b11, b12, b22, b13, b23, b33 = b
    d = b11 * b22 - b12 * b12
    if abs(d) < 1e-18:
        raise ValueError("degenerate homography set: cannot recover intrinsics")
    cy = (b12 * b13 - b11 * b23) / d
    lam = b33 - (b13 * b13 + cy * (b12 * b13 - b11 * b23)) / b11
    if lam <= 0:
        raise ValueError("degenerate homography set: negative scale")
    fx = np.sqrt(lam / b11)
    fy = np.sqrt(lam * b11 / d)
    cx = -b13 * fx * fx / lam
    return np.array([fx, fy, cx, cy])


def _extrinsics_from_homography(k: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover (rvec, tvec) from a homography and intrinsics K."""
    k_inv = np.linalg.inv(k)
    h1, h2, h3 = h[:, 0], h[:, 1], h[:, 2]
    lam = 1.0 / (np.linalg.norm(k_inv @ h1) + 1e-12)
    r1 = lam * (k_inv @ h1)
    r2 = lam * (k_inv @ h2)
    r3 = np.cross(r1, r2)
    t = lam * (k_inv @ h3)
    u, _, vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    # Rodrigues: R -> rvec
    theta = np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-12:
        rvec = np.zeros(3)
    else:
        rx = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
        rvec = theta / (2.0 * np.sin(theta)) * rx
    return rvec, t


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rvec)
    if theta < 1e-14:
        return np.eye(3)
    k = rvec / theta
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)


def _project(
    obj: np.ndarray,
    intr: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """Project planar object points (z=0) to pixels with distortion."""
    fx, fy, cx, cy = intr
    k1, k2, p1, p2, k3 = dist
    r = _rodrigues(rvec)
    pts3 = np.column_stack([obj, np.zeros(len(obj))])
    cam = pts3 @ r.T + tvec
    z = np.where(np.abs(cam[:, 2]) < 1e-12, 1e-12, cam[:, 2])
    xu = cam[:, 0] / z
    yu = cam[:, 1] / z
    r2 = xu * xu + yu * yu
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = xu * radial + 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu)
    yd = yu * radial + p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu
    return np.column_stack([fx * xd + cx, fy * yd + cy])


def calibrate(
    frames: list[FrameObservation],
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    """Estimate intrinsics + distortion from valid frames.

    Frames are sorted by ``frame_id`` internally, so input ordering does not
    affect the result.  Raises ``ValueError`` if there are too few frames or
    points, or if the views are degenerate.
    """
    if config is None:
        config = CalibrationConfig()
    frames = sorted(frames, key=lambda f: f.frame_id)
    if len(frames) < _MIN_FRAMES:
        raise ValueError(f"need at least {_MIN_FRAMES} valid frames, got {len(frames)}")
    for f in frames:
        if len(f.object_points) < _MIN_POINTS_PER_FRAME:
            raise ValueError(
                f"frame {f.frame_id!r} has {len(f.object_points)} points, "
                f"need >= {_MIN_POINTS_PER_FRAME}"
            )

    homographies = [
        _homography_dlt(f.object_points, f.image_points) for f in frames
    ]
    intr0 = _intrinsics_from_homographies(homographies)
    extrinsics = [
        _extrinsics_from_homography(
            np.array([[intr0[0], 0.0, intr0[2]],
                      [0.0, intr0[1], intr0[3]],
                      [0.0, 0.0, 1.0]]),
            h,
        )
        for h in homographies
    ]

    dist0 = np.zeros(5)
    if not config.use_tangential:
        dist0[2] = dist0[3] = 0.0
    if not config.use_k3:
        dist0[4] = 0.0

    x0 = np.concatenate(
        [intr0, dist0] + [np.concatenate([rv, tv]) for rv, tv in extrinsics]
    )

    def unpack(x: np.ndarray):
        intr = x[0:4]
        dist = x[4:9].copy()
        if not config.use_tangential:
            dist[2] = dist[3] = 0.0
        if not config.use_k3:
            dist[4] = 0.0
        ext = [x[9 + 6 * i: 15 + 6 * i] for i in range(len(frames))]
        return intr, dist, ext

    def residuals(x: np.ndarray) -> np.ndarray:
        intr, dist, ext = unpack(x)
        out = []
        for f, e in zip(frames, ext):
            proj = _project(f.object_points, intr, dist, e[0:3], e[3:6])
            out.append((proj - f.image_points).ravel())
        return np.concatenate(out)

    result = least_squares(
        residuals,
        x0,
        method="trf",
        max_nfev=config.max_iterations,
        xtol=config.convergence_tolerance,
        ftol=config.convergence_tolerance,
        gtol=config.convergence_tolerance,
    )

    intr, dist, _ = unpack(result.x)
    per_frame: dict[str, float] = {}
    _, dist_f, ext_f = unpack(result.x)
    sq_sum = 0.0
    count = 0
    for f, e in zip(frames, ext_f):
        proj = _project(f.object_points, intr, dist_f, e[0:3], e[3:6])
        err = proj - f.image_points
        sse = float(np.sum(err * err))
        per_frame[f.frame_id] = float(np.sqrt(sse / len(f.object_points)))
        sq_sum += sse
        count += len(f.object_points)
    rms = float(np.sqrt(sq_sum / count))

    return CalibrationResult(
        fx=float(intr[0]),
        fy=float(intr[1]),
        cx=float(intr[2]),
        cy=float(intr[3]),
        distortion=dist_f.astype(np.float64),
        rms_error=rms,
        per_frame_error=per_frame,
        iterations=int(result.nfev),
        converged=bool(result.status > 0),
        frame_ids=[f.frame_id for f in frames],
        config=config,
    )
