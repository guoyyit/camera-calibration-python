"""Intrinsic calibration solver (Levenberg-Marquardt bundle adjustment).

The solver estimates ``fx, fy, cx, cy, k1, k2, p1, p2`` jointly with one
6-DoF pose per frame by minimising the sum of squared reprojection errors.

Pipeline
--------
1. Frames are sorted by ``frame_id`` (canonical order) so the result does
   not depend on submission order.
2. Per-frame homographies are estimated with a normalised DLT.
3. Intrinsics are initialised with Zhang's closed-form solution.
4. All parameters are refined with Levenberg-Marquardt.  The Jacobian is
   computed with central finite differences that exploit the block-sparse
   structure (pose parameters of one frame only affect that frame's
   residuals).

Determinism
-----------
For a fixed set of input frames (in any submission order), fixed solver
configuration and the same platform/NumPy build, the output is bit-for-bit
reproducible: the frame order, initial guesses, damping schedule and finite
difference steps are all deterministic functions of the input.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .models import (
    CalibrationResult,
    CameraIntrinsics,
    FrameError,
    FrameObservation,
    SolverConfig,
    sorted_frames,
)

_MIN_FRAMES = 3  # Zhang's closed-form needs >= 3 views of the plane
_MIN_POINTS_PER_FRAME = 4  # homography DLT needs >= 4 correspondences


class CalibrationError(ValueError):
    """Raised when calibration cannot be performed on the given frames."""


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-14:
        return np.eye(3) + _skew(rvec)
    k = rvec / theta
    kx = _skew(k)
    return np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)


def _rodrigues_inv(rot: np.ndarray) -> np.ndarray:
    cos_theta = (float(np.trace(rot)) - 1.0) / 2.0
    theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    if theta < 1e-14:
        return np.zeros(3)
    factor = theta / (2.0 * np.sin(theta))
    return factor * np.array(
        [rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]
    )


def _normalise_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Similarity normalisation: centroid at origin, mean distance sqrt(2)."""
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_dist = float(np.linalg.norm(shifted, axis=1).mean())
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 0 else 1.0
    transform = np.array(
        [[scale, 0.0, -scale * centroid[0]],
         [0.0, scale, -scale * centroid[1]],
         [0.0, 0.0, 1.0]]
    )
    return shifted * scale, transform


def _homography_dlt(obj_xy: np.ndarray, img_xy: np.ndarray) -> np.ndarray:
    """Normalised DLT homography from board plane (Z=0) to image."""
    obj_n, t_obj = _normalise_2d(obj_xy)
    img_n, t_img = _normalise_2d(img_xy)
    n = obj_xy.shape[0]
    a = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = obj_n[i]
        u, v = img_n[i]
        a[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        a[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]
    _, _, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    homography = np.linalg.inv(t_img) @ h @ t_obj
    return homography / homography[2, 2]


def _initial_intrinsics(
    homographies: list[np.ndarray], width: int, height: int
) -> np.ndarray:
    """Zhang's closed-form intrinsics from homographies (skew fixed to 0)."""
    v_rows: list[np.ndarray] = []

    def v_vec(h: np.ndarray, i: int, j: int) -> np.ndarray:
        return np.array([
            h[0, i] * h[0, j],
            h[0, i] * h[1, j] + h[1, i] * h[0, j],
            h[1, i] * h[1, j],
            h[2, i] * h[0, j] + h[0, i] * h[2, j],
            h[2, i] * h[1, j] + h[1, i] * h[2, j],
            h[2, i] * h[2, j],
        ])

    for h in homographies:
        v_rows.append(v_vec(h, 0, 1))
        v_rows.append(v_vec(h, 0, 0) - v_vec(h, 1, 1))
    v_mat = np.stack(v_rows)
    try:
        _, _, vt = np.linalg.svd(v_mat)
        b = vt[-1]
        b11, b12, b22, b13, b23, b33 = b
        denom = b11 * b22 - b12 * b12
        cy = (b12 * b13 - b11 * b23) / denom
        lam = b33 - (b13 * b13 + cy * (b12 * b13 - b11 * b23)) / b11
        fx = np.sqrt(lam / b11)
        fy = np.sqrt(lam * b11 / denom)
        cx = -b13 * fx * fx / lam
        if not (np.isfinite(fx) and np.isfinite(fy) and fx > 0 and fy > 0):
            raise ValueError("degenerate closed-form intrinsics")
    except (ValueError, ZeroDivisionError, FloatingPointError):
        fx = fy = float(max(width, height))
        cx, cy = width / 2.0, height / 2.0
    return np.array([fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0])


def _pose_from_homography(
    homography: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intrinsics[:4]
    k_inv = np.array(
        [[1.0 / fx, 0.0, -cx / fx],
         [0.0, 1.0 / fy, -cy / fy],
         [0.0, 0.0, 1.0]]
    )
    h = k_inv @ homography
    norm = float(np.linalg.norm(h[:, 0]))
    if norm < 1e-12:
        norm = 1.0
    r1 = h[:, 0] / norm
    r2 = h[:, 1] / norm
    r3 = np.cross(r1, r2)
    t = h[:, 2] / norm
    rot = np.column_stack([r1, r2, r3])
    u, _, vt = np.linalg.svd(rot)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vt
    return _rodrigues_inv(rot), t


# ---------------------------------------------------------------------------
# Projection model
# ---------------------------------------------------------------------------


def project_points(
    obj_pts: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Project board-frame 3D points to pixels (with distortion)."""
    fx, fy, cx, cy, k1, k2, p1, p2 = intrinsics
    rot = _rodrigues(rvec)
    cam = obj_pts @ rot.T + tvec
    x = cam[:, 0] / cam[:, 2]
    y = cam[:, 1] / cam[:, 2]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.column_stack([fx * x_d + cx, fy * y_d + cy])


# ---------------------------------------------------------------------------
# Levenberg-Marquardt refinement
# ---------------------------------------------------------------------------


def _residuals(
    theta: np.ndarray,
    obj_list: list[np.ndarray],
    img_list: list[np.ndarray],
) -> np.ndarray:
    intrinsics = theta[:8]
    parts = []
    for i, (obj, img) in enumerate(zip(obj_list, img_list)):
        rvec = theta[8 + 6 * i: 8 + 6 * i + 3]
        tvec = theta[8 + 6 * i + 3: 8 + 6 * i + 6]
        projected = project_points(obj, rvec, tvec, intrinsics)
        parts.append((projected - img).ravel())
    return np.concatenate(parts)


def _frame_residuals(
    theta: np.ndarray,
    obj: np.ndarray,
    img: np.ndarray,
    frame_index: int,
) -> np.ndarray:
    intrinsics = theta[:8]
    rvec = theta[8 + 6 * frame_index: 8 + 6 * frame_index + 3]
    tvec = theta[8 + 6 * frame_index + 3: 8 + 6 * frame_index + 6]
    return (project_points(obj, rvec, tvec, intrinsics) - img).ravel()


def _jacobian(
    theta: np.ndarray,
    obj_list: list[np.ndarray],
    img_list: list[np.ndarray],
    base_residual: np.ndarray,
    row_offsets: list[int],
) -> np.ndarray:
    """Central-difference Jacobian exploiting block sparsity."""
    n_params = theta.size
    n_residuals = base_residual.size
    jac = np.zeros((n_residuals, n_params))
    eps = 1e-6

    # Intrinsics columns: dense, affect every residual.
    for j in range(8):
        h = eps * max(1.0, abs(theta[j]))
        theta_p = theta.copy(); theta_p[j] += h
        theta_m = theta.copy(); theta_m[j] -= h
        r_p = _residuals(theta_p, obj_list, img_list)
        r_m = _residuals(theta_m, obj_list, img_list)
        jac[:, j] = (r_p - r_m) / (2.0 * h)

    # Pose columns: frame i's pose only affects frame i's residual block.
    for i in range(len(obj_list)):
        start = row_offsets[i]
        stop = row_offsets[i + 1]
        for k in range(6):
            j = 8 + 6 * i + k
            h = eps * max(1.0, abs(theta[j]))
            theta_p = theta.copy(); theta_p[j] += h
            theta_m = theta.copy(); theta_m[j] -= h
            r_p = _frame_residuals(theta_p, obj_list[i], img_list[i], i)
            r_m = _frame_residuals(theta_m, obj_list[i], img_list[i], i)
            jac[start:stop, j] = (r_p - r_m) / (2.0 * h)
    return jac


def _lm_refine(
    obj_list: list[np.ndarray],
    img_list: list[np.ndarray],
    theta: np.ndarray,
    config: SolverConfig,
) -> tuple[np.ndarray, int, bool]:
    row_offsets = [0]
    for img in img_list:
        row_offsets.append(row_offsets[-1] + 2 * img.shape[0])

    residual = _residuals(theta, obj_list, img_list)
    cost = 0.5 * float(residual @ residual)
    damping = config.initial_damping
    iterations = 0
    converged = False

    for iteration in range(config.max_iterations):
        jac = _jacobian(theta, obj_list, img_list, residual, row_offsets)
        a_mat = jac.T @ jac
        grad = jac.T @ residual
        if float(np.abs(grad).max(initial=0.0)) < config.convergence_tolerance:
            converged = True
            iterations = iteration
            break

        accepted = False
        new_theta = theta
        new_residual = residual
        new_cost = cost
        for _ in range(30):  # damping-adjustment loop
            damped = a_mat + damping * np.diag(np.diag(a_mat) + 1e-12)
            try:
                delta = np.linalg.solve(damped, -grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(damped, -grad, rcond=None)[0]
            candidate = theta + delta
            candidate_residual = _residuals(candidate, obj_list, img_list)
            candidate_cost = 0.5 * float(candidate_residual @ candidate_residual)
            if np.isfinite(candidate_cost) and candidate_cost < cost:
                new_theta, new_residual, new_cost = (
                    candidate, candidate_residual, candidate_cost)
                accepted = True
                damping = max(damping / 10.0, 1e-12)
                break
            damping *= 10.0
            if damping > 1e12:
                break

        iterations = iteration + 1
        if not accepted:
            break
        relative_drop = (cost - new_cost) / max(1.0, cost)
        theta, residual, cost = new_theta, new_residual, new_cost
        if relative_drop < config.convergence_tolerance:
            converged = True
            break

    return theta, iterations, converged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def calibrate(
    frames: Sequence[FrameObservation],
    image_width: int,
    image_height: int,
    config: SolverConfig | None = None,
) -> CalibrationResult:
    """Estimate intrinsics from corner observations.

    Parameters
    ----------
    frames:
        Valid, deduplicated observations.  Order does not matter; frames are
        processed sorted by ``frame_id``.
    image_width, image_height:
        Sensor size in pixels (used for the initial principal-point guess).
    config:
        Solver configuration (iteration limit, convergence tolerance).

    Raises
    ------
    CalibrationError
        If there are fewer than 3 frames or a frame has fewer than 4 points.
    """
    config = config or SolverConfig()
    ordered = sorted_frames(frames)
    if len(ordered) < _MIN_FRAMES:
        raise CalibrationError(
            f"need at least {_MIN_FRAMES} frames, got {len(ordered)}"
        )
    for frame in ordered:
        if len(frame.object_points) < _MIN_POINTS_PER_FRAME:
            raise CalibrationError(
                f"frame {frame.frame_id!r} has "
                f"{len(frame.object_points)} points, "
                f"need at least {_MIN_POINTS_PER_FRAME}"
            )

    obj_list = [np.asarray(f.object_points, dtype=np.float64) for f in ordered]
    img_list = [np.asarray(f.image_points, dtype=np.float64) for f in ordered]

    homographies = [
        _homography_dlt(obj[:, :2], img) for obj, img in zip(obj_list, img_list)
    ]
    intrinsics = _initial_intrinsics(homographies, image_width, image_height)

    theta_parts = [intrinsics]
    for homography in homographies:
        rvec, tvec = _pose_from_homography(homography, intrinsics)
        theta_parts.extend([rvec, tvec])
    theta = np.concatenate(theta_parts)

    theta, iterations, converged = _lm_refine(obj_list, img_list, theta, config)

    intr = CameraIntrinsics(
        fx=float(theta[0]), fy=float(theta[1]),
        cx=float(theta[2]), cy=float(theta[3]),
        k1=float(theta[4]), k2=float(theta[5]),
        p1=float(theta[6]), p2=float(theta[7]),
    )

    frame_errors: list[FrameError] = []
    total_sq = 0.0
    total_count = 0
    for i, frame in enumerate(ordered):
        rvec = theta[8 + 6 * i: 8 + 6 * i + 3]
        tvec = theta[8 + 6 * i + 3: 8 + 6 * i + 6]
        projected = project_points(
            obj_list[i], rvec, tvec, np.array([
                intr.fx, intr.fy, intr.cx, intr.cy,
                intr.k1, intr.k2, intr.p1, intr.p2,
            ])
        )
        errors = np.linalg.norm(projected - img_list[i], axis=1)
        frame_errors.append(FrameError(
            frame_id=frame.frame_id,
            mean_error=float(errors.mean()),
            max_error=float(errors.max()),
        ))
        total_sq += float(errors @ errors)
        total_count += errors.size

    return CalibrationResult(
        intrinsics=intr,
        frame_errors=tuple(frame_errors),
        mean_reprojection_error=float(np.sqrt(total_sq / total_count)),
        iterations=iterations,
        converged=converged,
        used_frame_ids=tuple(f.frame_id for f in ordered),
    )
