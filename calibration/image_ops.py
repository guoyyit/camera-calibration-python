"""Distortion / undistortion mappings built on a calibration result.

Pixel convention: origin at the top-left pixel centre, x right, y down.

Boundary policy for :func:`undistort_image`: output pixels whose source
location falls outside the source image are filled with the constant
``border_value`` (documented constant-padding policy).
"""

from __future__ import annotations

import numpy as np

from .models import CalibrationResult

# Fixed iteration count for the inverse-distortion fixed-point solver.
# A fixed count keeps the mapping deterministic and reproducible.
_UNDISTORT_ITERATIONS = 20


def _distort_normalized(xu: np.ndarray, yu: np.ndarray, dist: np.ndarray):
    k1, k2, p1, p2, k3 = dist
    r2 = xu * xu + yu * yu
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = xu * radial + 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu)
    yd = yu * radial + p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu
    return xd, yd


def distort_points(points: np.ndarray, result: CalibrationResult) -> np.ndarray:
    """Forward map ideal (undistorted) pixel points to observed pixels.

    ``points`` is an (N, 2) array of pixel coordinates in the rectified
    image; returns their distorted counterparts in the original image.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    xu = (pts[:, 0] - result.cx) / result.fx
    yu = (pts[:, 1] - result.cy) / result.fy
    xd, yd = _distort_normalized(xu, yu, result.distortion)
    return np.column_stack(
        [result.fx * xd + result.cx, result.fy * yd + result.cy]
    )


def undistort_points(points: np.ndarray, result: CalibrationResult) -> np.ndarray:
    """Map observed (distorted) pixel points to ideal pixel coordinates.

    Uses a fixed number of fixed-point iterations on the distortion model,
    so the output is deterministic.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    xd = (pts[:, 0] - result.cx) / result.fx
    yd = (pts[:, 1] - result.cy) / result.fy
    xu, yu = xd.copy(), yd.copy()
    k1, k2, p1, p2, k3 = result.distortion
    for _ in range(_UNDISTORT_ITERATIONS):
        r2 = xu * xu + yu * yu
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        dx = 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu)
        dy = p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu
        denom = np.where(np.abs(radial) < 1e-12, 1e-12, radial)
        xu = (xd - dx) / denom
        yu = (yd - dy) / denom
    return np.column_stack(
        [result.fx * xu + result.cx, result.fy * yu + result.cy]
    )


def undistort_image(
    image: np.ndarray,
    result: CalibrationResult,
    border_value: float = 0.0,
) -> np.ndarray:
    """Rectify ``image`` using the calibration result.

    For every output (ideal) pixel the corresponding source location is
    computed with :func:`distort_points` and sampled with bilinear
    interpolation.  Any of the four bilinear neighbour pixels that falls
    outside the source image contributes the constant ``border_value``
    instead (constant-padding policy, matching OpenCV ``BORDER_CONSTANT``
    semantics).  Supports (H, W) grayscale and (H, W, C) colour images.
    """
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    dst = np.column_stack([xs.ravel(), ys.ravel()])
    src = distort_points(dst, result)
    sx = src[:, 0].reshape(h, w)
    sy = src[:, 1].reshape(h, w)

    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = sx - x0
    wy = sy - y0
    fill = np.asarray(border_value, dtype=np.float64)

    def sample(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        """Bilinear neighbour with constant padding outside the image."""
        inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        vals = img[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)]
        if img.ndim == 2:
            return np.where(inside, vals, fill)
        return np.where(inside[..., None], vals, fill)

    ia = sample(y0, x0)
    ib = sample(y0, x1)
    ic = sample(y1, x0)
    id_ = sample(y1, x1)

    if img.ndim == 2:
        wxe = wx
        wye = wy
    else:
        wxe = wx[..., None]
        wye = wy[..., None]

    return (
        ia * (1 - wxe) * (1 - wye)
        + ib * wxe * (1 - wye)
        + ic * (1 - wxe) * wye
        + id_ * wxe * wye
    )
